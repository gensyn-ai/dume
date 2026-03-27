from tqdm import tqdm
from copy import deepcopy

import torch

from models.llama_experts import get_llama_expert
from lm_eval.tasks import get_task_dict


NUM_SHOTS = {
    'gsm8k': 8,
    'mathqa': 0,
    'hendrycks_math': 0,
    'm_mmlu': None,
    'humaneval': None,
    'ifeval': None,
    'm_arc': None
}


def align_and_stack(tensors):
    # find min shape for each dim
    min_shape = [min(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape))]
    
    aligned = []
    for t in tensors:
        slices = tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))
        aligned.append(t[slices])
    
    return torch.stack(aligned)


def get_task(task_name):

    # Define multilingual ARC task group
    if task_name == "m_arc":
        multilingual_arc_tasks = [
            "arc_ar", "arc_bn", "arc_ca", "arc_de", "arc_es", "arc_eu", 
            "arc_fr", "arc_gu", "arc_hi", "arc_hr", "arc_hu", "arc_hy",
            "arc_id", "arc_it", "arc_kn", "arc_ml", "arc_mr", "arc_ne",
            "arc_nl", "arc_pt", "arc_ro", "arc_ru", "arc_sk", "arc_sr",
            "arc_sv", "arc_ta", "arc_te", "arc_uk", "arc_vi", "arc_zh"
        ]
        task_dict = get_task_dict(multilingual_arc_tasks)  # type: ignore
    else:
        task_dict = get_task_dict([task_name])

    if task_name == 'hendrycks_math':
        limit = None
        for _, tasks in task_dict.items():  # this is a fake for, its length is only 1
            for name, t in tasks.items():
                t.fewshot = 0
                t.bootstrap_iters = 0
    elif task_name == "m_mmlu":
        limit = 1764
    elif task_name in ("humaneval", "ifeval", "m_arc"):
        limit = None
    else:
        task = task_dict[task_name]
        task.fewshot = NUM_SHOTS[task_name]
        task.bootstrap_iters = 0
        limit = None
    
    return task_dict, limit


def load_moerged_model(ckpt_path, num_parameters, k, temp, _lambda, context_dim, num_domains):

    print(f"Loading MoErged model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    print("Reconstructing model architecture...")
    model, _, _ = get_llama_expert(num_parameters, config=checkpoint['config'])

    moe_experts = []
    experts_names = []
    state_dicts = checkpoint['moe_experts_state_dicts']

    for _k in tqdm(checkpoint['model_state_dict'].keys()):
        if 'mlp' in _k:
            # Keep separate experts for each domain
            expert_name = f"{_k.split('mlp')[0]}mlp"
            layer_idx = int(expert_name.rsplit('.', 2)[1])
            if expert_name in experts_names:
                continue
            experts_names.append(expert_name)
            moe_experts.append([deepcopy(model.get_submodule(expert_name)) for _ in range(num_domains)])
            for domain_idx in range(num_domains):
                moe_experts[-1][domain_idx].load_state_dict(
                    state_dicts[layer_idx][domain_idx],
                    strict=True
                )
    
    model_custom, _, _ = get_llama_expert(config=checkpoint['config'], num_parameters=num_parameters, moe_experts=moe_experts,
                                            k=k, temp=temp, _lambda=_lambda, context_dim=context_dim)
    model_custom.load_state_dict(checkpoint['model_state_dict'], strict=False)

    config = checkpoint['config']

    print("Done.")

    return model_custom, moe_experts, config


def get_context(task_name, batch):

    if task_name == 'gsm8k':
        return [example['question'] for example in batch]
    elif task_name == 'm_arc':
        return [example['instruction'] for example in batch]
    elif task_name == 'humaneval':
        return [example['prompt'] for example in batch]
    elif task_name == 'ifeval':
        return [example['prompt'] for example in batch]
    
    elif task_name == "allenai/tulu-3-sft-personas-instruction-following":
        return [example['messages'][0]['content'] for example in batch]
    elif task_name == "hkust-nlp/dart-math-hard":
        return [example['query'] for example in batch]
    elif task_name == "CohereLabs/aya_dataset":
        return [example['inputs'] for example in batch]
    elif task_name == "ise-uiuc/Magicoder-OSS-Instruct-75K":
        return [example['problem'] for example in batch]
    
    else:
        raise NotImplementedError(f"Context extraction not implemented for task {task_name}.")
