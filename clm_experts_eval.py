import os
import argparse

import torch
from transformers import set_seed

from utils.datasets_utils import get_dataloaders
from models.llama_experts import get_llama_expert
from clm_trainer import eval


def get_args():
    parser = argparse.ArgumentParser(description='Cross-evaluate LLaMA experts across domains')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-parameters', type=float, default=1.15e8, help='Number of model parameters')
    parser.add_argument("--num-test-steps", type=int, default=4000, help="Number of test evaluation steps")
    args = parser.parse_args()
    return args


def load_domain_statedict(domain, seed=42, num_parameters=1.15e8):
    path = os.path.join('checkpoints', f"llama_{int(num_parameters)}_{domain}{f'_seed={seed}' if seed != 42 else ''}.pth")
    if os.path.isfile(path):
        print(f"Loading checkpoint from {path}")
        state = torch.load(path, map_location='cpu')
    else:
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    return state


def main():
    
    args = get_args()
    all_domains = ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_dataloaders = {}
    for domain in all_domains:
        all_dataloaders[domain] = get_dataloaders(dataset_name=domain, batch_size=args.batch_size, num_workers=args.num_workers)

    perplexities = {}
    
    for domain_1 in all_domains:

        model, _, _ = get_llama_expert(args.num_parameters)

        state_dict = load_domain_statedict(domain_1, seed=args.seed, num_parameters=args.num_parameters)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        for domain_2 in all_domains:
            print(f"Evaluating domain {domain_2} using model trained on {domain_1}...")
            dataloaders = all_dataloaders[domain_2]
            perplexity = eval(dataloaders['test'], model, tot_steps=args.num_test_steps, device=device, ev_type='Test', dtype=torch.bfloat16)
            print(f"Perplexity on domain {domain_2} using model from {domain_1}: {perplexity:.4f}")
            if perplexities.get(domain_1) is None:
                perplexities[domain_1] = {}
            perplexities[domain_1][domain_2] = perplexity
    
    print("," + ",".join(f"{domain}" for domain in all_domains))
    for domain_1 in all_domains:
        print(f"{domain_1}", end=",")
        print(','.join(str(round(perplexities[domain_1][domain_2], 2)) for domain_2 in all_domains))


if __name__ == '__main__':
    main()
