from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    
    def __init__(self, experts, layer_id, k=1, temperature=1e-1, _lambda=1e-2, context_dim=512, alpha=None):
        """Mixture of Experts module with top-k gating."""
        
        super().__init__()

        self.experts = nn.ModuleList(experts)
        self.layer_id = layer_id
        self.num_experts = len(experts)

        self.k = k
        self.temperature = temperature
        self.alpha = alpha
        assert self.k <= self.num_experts, "k must be less than or equal to the number of experts"

        self.dim = next(experts[0].children()).in_features
        self.router = nn.Linear(self.dim, self.num_experts, dtype=torch.bfloat16)

        self.rr_gate = False
        self.A, self.b = None, None
        self.A_d, self.b_d = [], []  # collection of A and b matrices per expert
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

    def forward(self, x, expert_id=None, layer_id=None):

        if self.rr_gate:

            if layer_id is None or layer_id == self.layer_id:

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

                A = X_with_bias.T @ X_with_bias
                b = X_with_bias.T @ Y
                self.A += A
                self.b += b
                self.A_d[expert_id] += A  # type: ignore
                self.b_d[expert_id] += b  # type: ignore

            if layer_id is not None:
                expert_id = None  # reset expert_id to None to forward using the computed routers. expert_id was needed only for the onehot matrix.

        if expert_id is not None:
            """
            If the expert id is provided, use only that expert for all tokens.
            """
            return self.experts[expert_id](x)
        
        """
        Standard MoE forward with top-k gating.
        """

        logits = self.router(x)
        normalized_logits = logits / self.temperature

        soft_weights = F.softmax(normalized_logits, dim=-1, dtype=torch.bfloat16)

        scores, idx = torch.topk(normalized_logits, k=self.k, dim=-1)
        topk_weights = F.softmax(scores, dim=-1, dtype=torch.bfloat16)

        hard_weights = torch.zeros_like(logits, dtype=torch.bfloat16)
        hard_weights = hard_weights.scatter(dim=-1, index=idx, src=topk_weights)

        # Trick for gradients to flow through the hard weights while still having a one-hot like behavior in the forward pass.
        weights = hard_weights + (soft_weights - soft_weights.detach())

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

    def set_rr_gate_mode(self, mode: bool, reset: bool = False, layer_id: int | None = None):

        """
        If reset is set to True, force the reset A and b matrices. Otherwhise, initialize them only if they are None.
        Additionally, compute the optimal RR gating weights and set them to the gate layer when switching off the rr_gate mode.
        """
            
        self.rr_gate = mode

        if self.rr_gate:
            if layer_id is None or layer_id == self.layer_id:
                if self.A is None or self.b is None or reset:
                    self.A = torch.zeros((self.dim + 1, self.dim + 1), dtype=torch.float32, device=self.device)
                    self.b = torch.zeros((self.dim + 1, self.num_experts), dtype=torch.float32, device=self.device)
                    self.A_d = [torch.zeros((self.dim + 1, self.dim + 1), dtype=torch.float32, device=self.device) for i in range(self.num_experts)]
                    self.b_d = [torch.zeros((self.dim + 1, self.num_experts), dtype=torch.float32, device=self.device) for i in range(self.num_experts)]
                else:
                    num_new_experts = self.num_experts - self.b.shape[1]
                    if num_new_experts > 0:  # enter this if when a new expert has been added
                        self.b = torch.cat((self.b, torch.zeros((self.dim + 1, num_new_experts), dtype=torch.float32, device=self.device)), dim=1)
                        if self.routing_count is not None:  # this is None if no evaluation has been run yet
                            self.routing_count = torch.cat((self.routing_count, torch.zeros((num_new_experts,), dtype=torch.int32, device=self.device)), dim=0)  # type: ignore
                        self.A_d.append(torch.zeros((self.dim + 1, self.dim + 1), dtype=torch.float32, device=self.device))
                        self.b_d.append(torch.zeros((self.dim + 1, self.num_experts), dtype=torch.float32, device=self.device))
                    elif num_new_experts < 0: # enter this if when the last expert is removed
                        self.A -= self.A_d[self.num_experts]
                        self.b = self.b[:, :self.num_experts]
                        if self.routing_count is not None:
                            self.routing_count = self.routing_count[:self.num_experts]
                        self.A_d = self.A_d[:self.num_experts]
                        self.b_d = self.b_d[:self.num_experts]
        else:
            if layer_id is None or layer_id == self.layer_id:
                W = torch.linalg.solve(self.A + self._lambda * torch.eye(self.dim + 1, device=self.device), self.b)  # type: ignore
                bias = W[-1, :]
                W = W[:-1, :]
                norm = torch.norm(W, dim=0, keepdim=True)
                if torch.any(norm == 0.0):
                    print("WARNING: 0 encountered in norm, substituting with 1e-6")
                    norm[norm == 0.0] = 1e-6
                W = W / norm
                bias = bias / norm[0, :]
                W = W.to(torch.bfloat16)
                bias = bias.to(torch.bfloat16)
                self.router.weight.data = W.T
                self.router.bias.data = bias.T

    def add_expert(self, expert_state_dict):
        """Add a new expert to the MoE."""

        last_expert = self.experts[-1]
        new_expert = deepcopy(last_expert)
        new_expert.load_state_dict(expert_state_dict)
        self.experts.append(new_expert)
        self.num_experts += 1

        # Update gate layer to accommodate new expert
        new_gate = nn.Linear(self.dim, self.num_experts, dtype=torch.bfloat16).to(self.device)
        with torch.no_grad():    
            new_gate.weight[:-1, :].data = self.router.weight.data
            new_gate.bias[:-1] = self.router.bias.data
            new_gate.weight[-1, :] = torch.randn((new_gate.weight.shape[1],), device=self.device) * 0.01
            new_gate.bias[-1] = 0.0
        self.router = new_gate
    
    def remove_last_expert(self):
        """Remove the last expert from the MoE."""
        self.experts = self.experts[:-1]
        self.num_experts -= 1

        # Update gate layer to remove the last expert
        new_gate = nn.Linear(self.dim, self.num_experts).to(self.device)
        with torch.no_grad():
            new_gate.weight.data = self.router.weight.data[:-1, :]
            new_gate.bias.data = self.router.bias.data[:-1]
        self.router = new_gate