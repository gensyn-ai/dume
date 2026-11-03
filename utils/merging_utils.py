from copy import deepcopy

import torch
from torch.nn.functional import cross_entropy


def ties_merging(task_vectors, base_model_state_dict, ties_merging_k):

    # Step 1: Trimming - Keep only top-k% parameters in each task vector
        trimmed_task_vectors = {}
        
        for domain, task_vector in task_vectors.items():
            trimmed_vector = {}
            
            # Flatten all parameters and get their magnitudes
            all_magnitudes = []
            for key, param in task_vector.items():
                all_magnitudes.append(param.abs().flatten())
            all_magnitudes = torch.cat(all_magnitudes)
            
            # Calculate threshold for top TIES_K%
            total_params = len(all_magnitudes)
            k = int(total_params * ties_merging_k / 100)
            
            if k >= total_params:
                threshold = 0.0
            else:
                threshold, _ = torch.topk(all_magnitudes, k, largest=True, sorted=True)
                threshold = threshold[-1]
            
            # Apply trimming
            for key, param in task_vector.items():
                mask = (param.abs() >= threshold).float()
                trimmed_vector[key] = param * mask
            
            trimmed_task_vectors[domain] = trimmed_vector
            print(f"Domain {domain}: trimmed to top {ties_merging_k}% parameters (threshold: {threshold:.6f})")
        
        # Step 2: Elect Sign - For each parameter position, elect the most common sign
        final_merged = {}
        
        # Get all parameter names
        param_names = list(next(iter(trimmed_task_vectors.values())).keys())
        
        for param_name in param_names:
            # Collect all non-zero values for this parameter across all task vectors
            param_values = []
            for domain_vector in trimmed_task_vectors.values():
                param_values.append(domain_vector[param_name])
            
            # Stack the parameter tensors
            stacked_params = torch.stack(param_values, dim=0)  # Shape: [num_domains, ...]
            
            # Get signs of non-zero parameters
            sign_matrix = torch.sign(stacked_params)  # Shape: [num_domains, ...]
            non_zero_mask = stacked_params != 0  # Shape: [num_domains, ...]
            
            # For each position, count positive and negative signs
            positive_votes = (sign_matrix == 1) & non_zero_mask
            negative_votes = (sign_matrix == -1) & non_zero_mask
            
            positive_count = positive_votes.sum(dim=0).float()
            negative_count = negative_votes.sum(dim=0).float()
            
            # Elect the sign with more votes (break ties by keeping positive)
            elected_sign = torch.where(positive_count >= negative_count, 
                                    torch.ones_like(positive_count), 
                                    -torch.ones_like(positive_count))
            
            # Step 3: Disentangle - Keep only parameters that agree with elected sign and average them
            agreed_values = []
            for i, domain_vector in enumerate(trimmed_task_vectors.values()):
                param = domain_vector[param_name]
                # Keep values that have the same sign as elected and are non-zero
                param_sign = torch.sign(param)
                agreement_mask = (param_sign == elected_sign) & non_zero_mask[i]
                agreed_param = torch.where(agreement_mask, param, torch.zeros_like(param))
                agreed_values.append(agreed_param)
            
            # Average the agreed parameters
            agreed_stack = torch.stack(agreed_values, dim=0)
            non_zero_count = (agreed_stack != 0).sum(dim=0).float()
            non_zero_count = torch.clamp(non_zero_count, min=1)  # Avoid division by zero
            
            merged_param = agreed_stack.sum(dim=0) / non_zero_count
            
            final_merged[param_name] = merged_param
        
        # Step 4: Merge - Add merged task vector to original parameters
        merged_state_dict = {}
        for param_name in param_names:
            merged_state_dict[param_name] = base_model_state_dict[param_name] + final_merged[param_name]
            
        print("TIES merging completed successfully!")

        return merged_state_dict


def random_drop_and_rescale(task_vector, p):
        assert p is not None and 0 <= p < 1, "p must be in the range [0, 1)."
        # Generate a binary mask: 1 with probability (1-p) and 0 with probability p.
        mask = torch.bernoulli(torch.full(task_vector.shape, 1 - p, device=task_vector.device))
        # Apply the mask and rescale the kept values by 1/(1-p)
        return task_vector * mask / (1 - p)


# ============================================================
# STEP 1:
# Compute the diagonal Fisher information matrix
#
# Paper connection:
# Section 3 of the paper approximates each model posterior
# with a Gaussian whose precision matrix is the Fisher.
#
# The paper uses a diagonal Fisher approximation:
#
#   F ≈ diag(E[g^2])
#
# where:
#   g = ∇θ log p(y | x, θ)
#
# ============================================================

def compute_diagonal_fisher(model, dataloader, device, num_steps=None):

    """
    Computes a diagonal empirical Fisher estimate.
    Returns:
        fisher_dict[param_name] = diagonal Fisher tensor
    """

    model.eval()

    # fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.state_dict().items()}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):

        if num_steps is not None and batch_idx >= num_steps:
            break
        
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}...")

        inputs = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        model.zero_grad()
        outputs = model(input_ids=inputs)

        # ----------------------------------------------------
        # Paper:
        # Fisher = E[g g^T]
        #
        # Diagonal approximation:
        # diag(F) = E[g^2]
        #
        # So we accumulate squared gradients.
        # ----------------------------------------------------

        loss = cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.detach() ** 2

        num_batches += 1

    # --------------------------------------------------------
    # Monte Carlo expectation estimate:
    #
    # E[g^2] ≈ (1/N) Σ g_i^2
    # --------------------------------------------------------

    for name in fisher:
        fisher[name] /= max(num_batches, 1)

    return fisher

# ============================================================
# STEP 2:
# Fisher merge task vectors
#
# Paper connection:
#
# Equation:
#
#   θ* = (Σ F_i)^(-1) (Σ F_i θ_i)
#
# Since:
#
#   θ_i = θ_base + τ_i
#
# we merge:
#
#   θ* = θ_base + merged_task_vector
#
# where:
#
#   merged_task_vector =
#       (Σ F_i τ_i) / (Σ F_i)
#
# under the diagonal Fisher approximation.
#
# ============================================================

def fisher_merge_task_vectors(base_state_dict, task_vectors, fisher_matrices, device, fisher_floor=1e-12):

    """
    Fisher merge multiple task vectors.
    Args:
        base_state_dict:
            Base pretrained model state_dict
        task_vectors:
            dict:
                domain_name -> task_vector_state_dict
            Each task vector is:
                task_vector = finetuned - base
        fisher_matrices:
            dict:
                domain_name -> fisher_state_dict
        fisher_floor:
            Numerical stability epsilon

    Returns:
        merged_state_dict

    """

    merged_state = deepcopy(base_state_dict)
    domains = list(task_vectors.keys())
    for param_name in merged_state.keys():
        base_param = base_state_dict[param_name].to(device)

        # ----------------------------------------------------
        # Numerator:
        #
        # Σ_i F_i τ_i
        # ----------------------------------------------------

        weighted_task_sum = torch.zeros_like(base_param).to(device)

        # ----------------------------------------------------
        # Denominator:
        #
        # Σ_i F_i
        # ----------------------------------------------------

        fisher_sum = torch.zeros_like(base_param).to(device)

        for domain in domains:
            task_vec = task_vectors[domain][param_name].to(device)
            # Fisher importance for this parameter
            fisher = fisher_matrices[domain][param_name].to(device)
            weighted_task_sum += fisher * task_vec
            fisher_sum += fisher

        # ----------------------------------------------------
        # Paper:
        #
        # θ* = (Σ F_i θ_i) / (Σ F_i)
        #
        # In task-vector form:
        #
        # τ* = (Σ F_i τ_i) / (Σ F_i)
        # ----------------------------------------------------

        merged_task_vector = weighted_task_sum / (fisher_sum + fisher_floor)
        merged_state[param_name] = base_param + merged_task_vector

    return merged_state


def fisher_merging(task_vectors, base_state_dict, model, state_dicts, dataloaders, device, num_steps=100):

    """
    Fisher merge task vectors into a single merged model state dict.
    
    ===============================================================
    INPUTS
    ===============================================================

    task_vectors: dict: domain_name -> task_vector_state_dict, where: task_vector = finetuned_model - base_model
    base_model_state_dict: state_dict of the shared pretrained/base model
    dataloaders: dict: domain_name -> {"model": finetuned_model, "dataloader": dataloader}
        We need BOTH:
            - the finetuned model
            - the task/domain dataset
        because Fisher must be computed around the finetuned solution (not around the base model).

    ===============================================================
    PAPER CONNECTION
    ===============================================================

    The paper merges models via:
        θ* = (Σ F_i)^(-1) (Σ F_i θ_i)

    where:
        θ_i = finetuned model parameters
        F_i = Fisher information matrix

    Since task vectors are:
        τ_i = θ_i - θ_base

    we can rewrite:
        θ_i = θ_base + τ_i

    Substituting:
        θ* = θ_base + (Σ F_i τ_i) / (Σ F_i)

    Under the diagonal Fisher approximation, all operations become elementwise.

    ===============================================================
    RETURNS
    ===============================================================

    merged_state_dict
    """

    domains = list(task_vectors.keys())

    # ==========================================================
    # STEP 1:
    # Compute Fisher matrices for each domain
    #
    # Paper:
    # F_i ≈ E[g²]
    #
    # using the empirical Fisher approximation.
    # ==========================================================

    fisher_matrices = {}
    model.to(device)

    for i, domain in enumerate(domains):
        model.load_state_dict(state_dicts[i])
        print(f"Computing Fisher information matrix for domain {domain}...")
        fisher_matrices[domain] = compute_diagonal_fisher(model, dataloaders[domain]["train"], device, num_steps=num_steps)

    # ==========================================================
    # STEP 2:
    # Fisher merge the task vectors
    # τ* = (Σ F_i τ_i) / (Σ F_i)
    # ==========================================================

    merged_state_dict = fisher_merge_task_vectors(base_state_dict, task_vectors, fisher_matrices, device, fisher_floor=1e-12)

    return merged_state_dict


def consensus_merging(task_vectors, base_state_dict, consensus_threshold=2, lambda_sparsity=1.0):
    """
    Consensus Merging
    =================

    Paper:
        "Localizing Task Information for Improved Model Merging and Compression"
        https://arxiv.org/pdf/2405.07813

    -------------------------------------------------------------------------
    INPUT FORMAT
    -------------------------------------------------------------------------

    task_vectors: Dict[str, Dict[str, Tensor]]

        Dictionary mapping:
            domain_name -> task_vector

        Each task vector is:
            τ_t = θ_t - θ₀

        where:
            θ_t : fine-tuned model for task/domain t
            θ₀  : base/pretrained model

    base_state_dict:
        State dict of the base model θ₀.

    consensus_threshold: int
        Minimum number of tasks that must find a parameter important (k in paper).
        Default k=2 removes "selfish" (used by 1 task) and "catastrophic" (used by 0) weights.

    lambda_sparsity: float
        Sparsity parameter for mask computation. Higher values = sparser masks.

    -------------------------------------------------------------------------
    THEORY
    -------------------------------------------------------------------------

    The paper identifies three types of weights:
        1. "General" weights: important to multiple tasks (keep)
        2. "Selfish" weights: important to only one task (remove for k≥2)
        3. "Catastrophic" weights: important to no tasks (remove)

    -------------------------------------------------------------------------
    ALGORITHM
    -------------------------------------------------------------------------

    1. Compute merged task vector: τ_MTL = Σ_t τ_t

    2. For each task t, compute importance mask:
       m_t(i) = 1 if |τ_t(i)| ≥ λ_t |τ_MTL(i) - τ_t(i)|

    3. Count consensus: n_i = Σ_t m_t(i)

    4. Consensus mask: m_consensus(i) = 1 if n_i ≥ k

    5. Final merge: τ_consensus = m_consensus ⊙ τ_MTL
                   θ_final = θ₀ + τ_consensus

    -------------------------------------------------------------------------
    RETURNS
    -------------------------------------------------------------------------

    merged_state_dict:
        State dict of the merged model.
    """

    merged_state_dict = deepcopy(base_state_dict)
    domains = list(task_vectors.keys())
    T = len(domains)

    for key in base_state_dict.keys():
        base_param = base_state_dict[key]
        
        # =================================================================
        # STEP 1: Compute merged task vector (Task Arithmetic)
        # =================================================================
        #     τ_MTL = Σ_t τ_t
        # =================================================================
        task_sum = torch.zeros_like(base_param)
        task_deltas = []
        
        for domain in domains:
            delta = task_vectors[domain][key].to(base_param.device)
            task_deltas.append(delta)
            task_sum += delta

        # Stack for vectorized operations
        deltas = torch.stack(task_deltas, dim=0)  # Shape: [T, ...]

        # =================================================================
        # STEP 2: Compute task importance masks
        # =================================================================
        # For each task t:
        #     m_t(i) = 1 if |τ_t(i)| ≥ λ_t |τ_MTL(i) - τ_t(i)|
        # =================================================================
        importance_masks = []
        
        for t in range(T):
            tau_t = deltas[t]  # Current task vector
            tau_others = task_sum - tau_t  # τ_MTL - τ_t
            
            # Importance criterion: |τ_t(i)| ≥ λ |τ_others(i)|
            left_side = tau_t.abs()
            right_side = lambda_sparsity * tau_others.abs()
            
            mask_t = (left_side >= right_side).float()
            importance_masks.append(mask_t)

        # Stack masks: Shape [T, ...]
        masks = torch.stack(importance_masks, dim=0)

        # =================================================================
        # STEP 3: Count consensus strength
        # =================================================================
        #     n_i = Σ_t m_t(i)
        # =================================================================
        consensus_count = masks.sum(dim=0)  # Sum across tasks

        # =================================================================
        # STEP 4: Create consensus mask
        # =================================================================
        #     m_consensus(i) = 1 if n_i ≥ k
        # =================================================================
        consensus_mask = (consensus_count >= consensus_threshold).float()

        # =================================================================
        # STEP 5: Apply consensus mask to merged task vector
        # =================================================================
        #     τ_consensus = m_consensus ⊙ τ_MTL
        # =================================================================
        masked_task_sum = consensus_mask * task_sum

        # =================================================================
        # STEP 6: Construct final merged model
        # =================================================================
        #     θ_final = θ₀ + τ_consensus
        # =================================================================
        merged_state_dict[key] = base_param + masked_task_sum

    return merged_state_dict