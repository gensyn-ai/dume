import os
import argparse

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, set_seed

from clm_trainer import eval
from clm_train_experts import load_pretrained_model
from utils.datasets_utils import get_dataloaders
from utils.merging_utils import consensus_merging, fisher_merging, random_drop_and_rescale, ties_merging
from models.llama_experts import get_llama_expert
from clm2_train_experts import get_dataloaders as get_clm2_dataloaders


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the LLaMA merged model with weights averaging across all the five clm domains.')
    parser.add_argument('--setting', type=str, choices=['clm', 'clm2'], default='clm2', help='Whether to evaluate the Moerging techniques on the CLM or CLM2 experts')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--num-parameters', type=float, default=1e9, help='Number of model parameters')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument("--num-test-steps", type=int, default=4000, help="Number of test evaluation steps")
    parser.add_argument("--freeze_attn", action='store_true', help="Set to true to load a model that trains only the MLP layers")
    parser.add_argument('--merging-strategy', type=str, choices=['average', 'task-arithmetic', 'ties-merging', 'DARE', 'Fisher', 'consensus'], default='Fisher', help='Merging strategy to use.')
    parser.add_argument("--task-arithmetic-lambda", type=float, default=None, help="Lambda for task arithmetic merging strategy, controlling the weight of each task vector.")
    parser.add_argument("--ties-merging-k", type=float, default=None, help="Top-k% parameters to keep for TIES merging strategy.")
    parser.add_argument("--dare-p", type=float, default=None, help="Drop probability for DARE merging strategy.")
    parser.add_argument('--max-num-tokens', type=int, default=1024, help='Maximum number of tokens per sample for the clm2 setting')
    parser.add_argument('--max-num-samples', type=int, default=100000, help='Maximum number of samples to use in the dataset for the clm2 setting')
    parser.add_argument('--num-fisher-steps', type=int, default=10, help='Number of steps to use for estimating the Fisher information matrices for the Fisher merging strategy.')
    parser.add_argument('--consensus-threshold', type=int, default=1, help='Threshold for consensus merging strategy, controlling the minimum agreement required across task vectors to keep a parameter change.')
    parser.add_argument('--consensus-lambda-sparsity', type=float, default=0.2, help='Sparsity parameter for mask computation. Higher values = sparser masks.')
    args = parser.parse_args()
    return args
    

def main():

    args = get_args()
    if args.setting == 'clm':
        all_domains = ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']
    elif args.setting == 'clm2':
        all_domains = ['math', 'bio', 'legal', 'dialogue']
    else:
        raise ValueError(f"Unknown setting: {args.setting}")

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_dataloaders = {}
    state_dicts = []
    for domain in all_domains:
        if args.setting == 'clm':
            all_dataloaders[domain] = get_dataloaders(dataset_name=domain, batch_size=args.batch_size, num_workers=args.num_workers)
        elif args.setting == 'clm2':
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
            tokenizer.pad_token = tokenizer.eos_token
            all_dataloaders[domain] = get_clm2_dataloaders(domain, args.batch_size, args.num_workers, tokenizer, args.max_num_tokens, args.max_num_samples)
        # str_seed = "_seed=42" if args.num_parameters == 1.15e8 or args.freeze_attn else f"_seed={args.seed}"
        str_seed = f"_seed={args.seed}"
        path = f"checkpoints/llama_{int(args.num_parameters)}_{domain}{str_seed}{'_freeze_attn' if args.freeze_attn else ''}.pth"
        print(path)
        path = os.path.join('checkpoints', f"llama_{int(args.num_parameters)}_{domain}{str_seed}{'_freeze_attn' if args.freeze_attn else ''}.pth")
        if os.path.isfile(path):
            print(f"Loading checkpoint from {path}")
            state = torch.load(path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        state_dicts.append(state)

    if args.merging_strategy == 'average':
        merged_state_dict = {k: torch.mean(torch.stack([sd[k] for sd in state_dicts]), dim=0) for k in state_dicts[0].keys()}
    
    elif args.merging_strategy in ('task-arithmetic', 'DARE'):
        print("Loading base_model...")
        pretrained_path = f'checkpoints/llama_{int(args.num_parameters)}_pretrained.pth'
        base_model, _ = load_pretrained_model(args.num_parameters, pretrained_path, True)
        base_model_state_dict = base_model.state_dict()
        ta_lambda = args.task_arithmetic_lambda
        merged_state_dict = {}
        for _k in state_dicts[0].keys():
            task_vectors = []
            for domain_idx in range(len(all_domains)):
                task_vector = state_dicts[domain_idx][_k] - base_model_state_dict[_k]  # type: ignore
                if args.merging_strategy == 'DARE':
                    task_vector = random_drop_and_rescale(task_vector, p=args.dare_p)
                task_vectors.append(task_vector)
            task_vectors = torch.stack(task_vectors, dim=0)  # [num_domains, *param_shape]
            merged_state_dict[_k] = torch.sum(task_vectors * ta_lambda, dim=0) + base_model_state_dict[_k]  # type: ignore
    
    elif args.merging_strategy in ('ties-merging', 'Fisher', 'consensus'):
        
        print("Loading base_model...")
        pretrained_path = f'checkpoints/llama_{int(args.num_parameters)}_pretrained.pth'
        base_model, _ = load_pretrained_model(args.num_parameters, pretrained_path, True)
        base_model_state_dict = base_model.state_dict()
        
        print("Extracting task vectors...")
        task_vectors = {}
        for domain_idx in range(len(all_domains)):
            task_vector = {}
            for _k in state_dicts[domain_idx].keys():
                task_vector[_k] = state_dicts[domain_idx][_k] - base_model_state_dict[_k]  # type: ignore
            task_vectors[all_domains[domain_idx]] = task_vector
    
        if args.merging_strategy == 'ties-merging':
            assert args.ties_merging_k is not None, "TIES merging strategy requires --ties-merging-k argument to specify the top-k% parameters to keep."
            print(f"Performing TIES merging with {len(task_vectors)} task vectors...")
            merged_state_dict = ties_merging(task_vectors, base_model_state_dict, args.ties_merging_k)

        elif args.merging_strategy == 'Fisher':
            print("Performing Fisher merging...")
            merged_state_dict = fisher_merging(task_vectors, base_model_state_dict, base_model, state_dicts, all_dataloaders, device, num_steps=args.num_fisher_steps)
        
        elif args.merging_strategy == 'consensus':
            print("Performing Consensus merging...")
            merged_state_dict = consensus_merging(task_vectors, base_model_state_dict, consensus_threshold=args.consensus_threshold, lambda_sparsity=args.consensus_lambda_sparsity)
        
        else:
            raise ValueError(f"Unknown merging strategy: {args.merging_strategy}")

    else:
        raise ValueError(f"Unknown merging strategy: {args.merging_strategy}")

    if args.num_parameters == 1e9:
        model_name = 'meta-llama/Llama-3.2-1B'
        config = LlamaForCausalLM.from_pretrained(model_name).config
    else:
        config = None
    model, _, _ = get_llama_expert(args.num_parameters, config=config)
    model.load_state_dict(merged_state_dict, strict=False)
    model.to(device)
    
    perplexities = {}
    for domain in all_domains:
        print(f"Evaluating merged model on {domain}...")
        dataloaders = all_dataloaders[domain]
        perplexity = eval(dataloaders['test'], model, tot_steps=args.num_test_steps, device=device, ev_type='Test', dtype=torch.bfloat16)
        print(f"Perplexity on domain {domain} using merged model: {perplexity:.4f}")
        perplexities[domain] = perplexity
    
    print(",".join(f"{domain}" for domain in all_domains))
    print(','.join(str(round(perplexities[domain], 2)) for domain in all_domains))
            
    if args.merging_strategy == 'task-arithmetic':
        with open(f"logs/{args.setting}_{args.merging_strategy}_talambda={args.task_arithmetic_lambda}_seed={args.seed}.csv", "w") as f:
            f.write(",".join(f"{domain}" for domain in all_domains) + "\n")
            f.write(','.join(str(round(perplexities[domain], 2)) for domain in all_domains) + "\n")
    elif args.merging_strategy == 'ties-merging':
        with open(f"logs/{args.setting}_{args.merging_strategy}_k={args.ties_merging_k}_seed={args.seed}.csv", "w") as f:
            f.write(",".join(f"{domain}" for domain in all_domains) + "\n")
            f.write(','.join(str(round(perplexities[domain], 2)) for domain in all_domains) + "\n")
    elif args.merging_strategy == 'DARE':
        with open(f"logs/{args.setting}_{args.merging_strategy}_p={args.dare_p}_seed={args.seed}.csv", "w") as f:
            f.write(",".join(f"{domain}" for domain in all_domains) + "\n")
            f.write(','.join(str(round(perplexities[domain], 2)) for domain in all_domains) + "\n")
    elif args.merging_strategy == 'Fisher':
        with open(f"logs/{args.setting}_{args.merging_strategy}_steps={args.num_fisher_steps}_seed={args.seed}.csv", "w") as f:
            f.write(",".join(f"{domain}" for domain in all_domains) + "\n")
            f.write(','.join(str(round(perplexities[domain], 2)) for domain in all_domains) + "\n")
    elif args.merging_strategy == 'consensus':
        with open(f"logs/{args.setting}_{args.merging_strategy}_ct={args.consensus_threshold}_cls={args.consensus_lambda_sparsity}_seed={args.seed}.csv", "w") as f:
            f.write(",".join(f"{domain}" for domain in all_domains) + "\n")
            f.write(','.join(str(round(perplexities[domain], 2)) for domain in all_domains) + "\n")


if __name__ == "__main__":
    main()
