import os
import argparse

import torch
from transformers import set_seed

from clm_trainer import eval
from utils.datasets_utils import get_dataloaders
from models.llama_experts import get_llama_expert


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the LLaMA merged model with weights averaging across all the five clm domains.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--num-parameters', type=float, default=1.15e8, help='Number of model parameters')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument("--num-test-steps", type=int, default=4000, help="Number of test evaluation steps")
    args = parser.parse_args()
    return args
    

def main():

    args = get_args()
    all_domains = ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_dataloaders = {}
    state_dicts = []
    for domain in all_domains:
        all_dataloaders[domain] = get_dataloaders(dataset_name=domain, batch_size=args.batch_size, num_workers=args.num_workers)
        path = os.path.join('checkpoints', f"llama_{int(args.num_parameters)}_{domain}{f'_seed={args.seed}' if args.seed != 42 else ''}.pth")
        if os.path.isfile(path):
            print(f"Loading checkpoint from {path}")
            state = torch.load(path, map_location='cpu')
        else:
            raise FileNotFoundError(f"Checkpoint path not found: {path}")
        state_dicts.append(state)

    merged_state_dict = {k: torch.mean(torch.stack([sd[k] for sd in state_dicts]), dim=0) for k in state_dicts[0].keys()}
    model, _, _ = get_llama_expert(args.num_parameters)
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
            

if __name__ == "__main__":
    main()
