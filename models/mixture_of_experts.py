import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    
    def __init__(self, experts, k=1, temperature=1e-1, _lambda=1e-2, context_dim=512, alpha=None):
        """Mixture of Experts module with top-k gating."""
        
        super().__init__()

        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.k = k
        self.temperature = temperature
        self.alpha = alpha
        assert self.k <= self.num_experts, "k must be less than or equal to the number of experts"

        self.dim = next(experts[0].children()).in_features
        self.gate = nn.Linear(self.dim, self.num_experts)

        self.rr_gate = False
        self.A, self.b = None, None
        self._lambda = _lambda
        self.context_dim = context_dim
        self.routing_count = None

    @property
    def device(self):
        """Return the device of this module (based on its parameters/buffers)."""
        p = next(self.parameters(), None)
        if p is not None:
            return p.device
        b = next(self.buffers(), None)
        return b.device if b is not None else torch.device("cpu")

    def reset_routing_count(self):
        """Reset routing statistics. Called on train/eval switches."""
        self.routing_count = None

    def eval(self):
        """Override nn.Module.eval to also reset routing stats.

        Calling .eval() invokes .train(False), so this covers both.
        """
        super().eval()
        self.reset_routing_count()
        return self

    def forward(self, x, expert_id=None):

        if self.rr_gate:

            """
            Collection mode. Collect the A and b matrices for RR gating.
            """
            
            assert expert_id is not None, "expert_id must be provided in rr_gate mode"
            assert isinstance(self.A, torch.Tensor) and isinstance(self.b, torch.Tensor), "A and b matrices must be initialized in rr_gate mode"
            
            X = x[:, :self.context_dim, :]  # use only context_dim tokens for RR gate stats
            flattened_X = X.reshape(-1, X.size(-1))
            X_with_bias = torch.cat((flattened_X, torch.ones((flattened_X.shape[0], 1), dtype=torch.float32).to(flattened_X.device)), dim=1)
            
            one_hot = F.one_hot(
                torch.as_tensor(expert_id, device=X_with_bias.device),
                num_classes=self.num_experts
            ).float()
            rows = X_with_bias.size(0)
            Y = one_hot.unsqueeze(0).expand(rows, -1)

            self.A += X_with_bias.T @ X_with_bias
            self.b += X_with_bias.T @ Y

        if expert_id is not None:
            """
            If the expert id is provided, use only that expert for all tokens.
            """
            return self.experts[expert_id](x)
        
        """
        Standard MoE forward with top-k gating.
        """

        logits = self.gate(x)
        normalized_logits = logits / self.temperature

        scores, idx = torch.topk(normalized_logits, k=self.k, dim=-1)
        topk_weights = F.softmax(scores, dim=-1, dtype=torch.bfloat16)    # [batch, tokens, k]
        weights = torch.zeros_like(logits)                                # [batch, tokens, num_experts]
        weights.scatter_(dim=-1, index=idx, src=topk_weights)             # fill only top-k

        self.update_routing_stats(weights)

        out = None
        for i, expert in enumerate(self.experts):
            if out is None:
                out = expert(x) * torch.unsqueeze(weights[:, :, i], -1)
            else:
                out = out + expert(x) * torch.unsqueeze(weights[:, :, i], -1)

        p = None
        if self.alpha is not None:
            p = F.softmax(normalized_logits, dim=-1, dtype=torch.bfloat16)
            return out, p, weights
        
        return out

    def update_routing_stats(self, weights):
        """
        Count how many times each expert was selected (weight > 0) across all samples and tokens.
        """
        if self.routing_count is None:
            self.routing_count = torch.sum(weights > 0, dim=(0, 1), dtype=torch.int32)
        else:
            self.routing_count += torch.sum(weights > 0, dim=(0, 1), dtype=torch.int32)

    def set_rr_gate_mode(self, mode: bool):

        """
        If mode is set to True, reset A and b matrices. Otherwhise, compute the optimal RR gating weights and set them to the gate layer.
        """
            
        self.rr_gate = mode

        if self.rr_gate:
            self.A = torch.zeros((self.dim + 1, self.dim + 1), dtype=torch.float32, device=self.device)
            self.b = torch.zeros((self.dim + 1, self.num_experts), dtype=torch.float32, device=self.device)
        
        else:
            W = torch.linalg.solve(self.A + self._lambda * torch.eye(self.dim + 1, device=self.device), self.b)  # type: ignore
            bias = W[-1, :]
            W = W[:-1, :]
            norm = torch.norm(W, dim=0, keepdim=True)
            if torch.any(norm == 0.0):
                print("WARNING: 0 encountered in norm, substituting with 1e-6")
                norm[norm == 0.0] = 1e-6
            W = W / norm
            bias = bias / norm[0, :]
            self.gate.weight.data = W.T
            self.gate.bias.data = bias.T
