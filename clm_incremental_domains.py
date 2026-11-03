import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from clm_moerging_strategies_eval import merge_datasets, moerge, collect_dume_stats
from utils.datasets_utils import get_dataloaders
from clm_trainer import eval, train
from clm2_train_experts import get_dataloaders as get_clm2_dataloaders
from utils.optimizer import get_training_strategy
from utils.distributed import distributed_init
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy


DEFAULT_ORDER = {
    'clm': ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking'],
    'clm2': ['math', 'bio', 'legal', 'dialogue']
}

def get_args():
    parser = argparse.ArgumentParser(description='Incremental DUME experiments for CLM')
    parser.add_argument('--method', type=str, choices=['DUME', 'BTX'], default='DUME', help='Whether to use DUME or BTX')
    parser.add_argument('--scenario', type=str, choices=['clm', 'clm2'], default='clm', help='Whether to use CLM or CLM2 experts and datasets')
    parser.add_argument('--setting', type=str, choices=['add', 'remove'], default='add', help='Setting')
    parser.add_argument('--seed', type=int, default=42, help='Random seed of the loaded experts')
    parser.add_argument('--num_parameters', type=float, default=1.15e8, help='Number of parameters for the base dense model experts')
    parser.add_argument('--num_dume_stats_steps', type=int, default=1000, help='Maximum number of steps for DUME statistics extraction')
    parser.add_argument('--num_test_steps', type=int, default=4000, help='Maximum number of steps for evaluation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--k', type=int, default=1, help='Number of top-k experts to route')
    parser.add_argument('--temp', type=float, default=1.0, help='Softmax temperature for routing')
    parser.add_argument('--_lambda', type=float, default=1e-2, help='Lambda for DUME routers')
    parser.add_argument('--num_dume_tokens', type=int, default=10000, help='Number of tokens to use for DUME statistics collection')
    parser.add_argument("--freeze_attn", action='store_true', default=False, help="Set to true to load experts that train only the MLP layers")
    parser.add_argument("--domains_order", nargs='+',
                        default=['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking'],
                        help="Ordered list of domains for DUME")
    parser.add_argument("--update_model", action='store_true', default=False, help="Set to true to update non-moe layers using a moving average when merging new experts")
    parser.add_argument("--average_experts", action='store_true', default=False, help="Set to true to average the parameters but the mlp layers of all experts from the beginning. Requires update_model to be False.")
    parser.add_argument('--max-num-tokens', type=int, default=1024, help='Maximum number of tokens per sample for the clm2 setting')
    parser.add_argument('--max-num-samples', type=int, default=100000, help='Maximum number of samples to use in the dataset for the clm2 setting')
    
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps for BTX training')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for BTX training')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of training steps for BTX training')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for BTX training. Set to None for other experiments.')
    parser.add_argument("--print_loss_interval", type=int, default=10, help="Interval for printing training loss during BTX training")
    
    args = parser.parse_args()
    assert not (args.update_model and args.average_experts), "Cannot use both update_model and average_experts at the same time."
    return args


def update_dume_stats(moerged_model, expert_id, all_domains, all_dataloaders, num_dume_stats_steps, device, mode='add'):
    assert mode in ['add', 'remove'], "mode must be either 'add' or 'remove'"
    print("Updating DUME statistics...")
    moerged_model.set_rr_gate_mode(True)  # type: ignore
    if mode == 'add':
        with torch.no_grad():
            domain = all_domains[expert_id]
            print(f"Collecting DUME statistics for domain {domain}...")
            dataloaders = all_dataloaders[domain]
            pbar = tqdm(total=num_dume_stats_steps, desc='DUME stats collection')
            for i, batch in enumerate(dataloaders['train']):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                moerged_model(input_ids=input_ids, expert_id=expert_id)  # type: ignore
                pbar.update(1)
                if i + 1 >= num_dume_stats_steps:
                    break
            pbar.close()
    moerged_model.set_rr_gate_mode(False)  # type: ignore


def update_model_params(moerged_model, state_dict, num_models, mode='add'):
    """
    Update the non-MoE parameters of the moerged model using a moving average
    with the parameters of the new expert being added.
    """
    with torch.no_grad():
        for name, param in moerged_model.named_parameters():
            if "mlp" not in name:
                if mode == 'add':
                    param.data.mul_(num_models - 1).add_(state_dict[name].to(param.device)).div_(num_models)
                elif mode == 'remove':
                    # param.data.mul_(num_models).sub_(state_dict[name].to(param.device)).div_(num_models - 1)
                    param.data.add_((param.data - state_dict[name].to(param.device)).div_(num_models - 1))  # Equivalent to the line above but numerically more stable
                else:
                    raise ValueError(f"Unknown mode {mode}.")


def evaluate(all_dataloaders, moerged_model, device, num_test_steps, perplexities, order, expert_id=None):
    perplexities[expert_id] = {}
    for _, domain in enumerate(order):
        print(f"Evaluating moerged model on {domain}...")
        dataloaders = all_dataloaders[domain]
        perplexity = eval(dataloaders['test'], moerged_model, tot_steps=num_test_steps, device=device, ev_type='Test', dtype=torch.bfloat16, expert_id=None)  # type: ignore
        print(f"Perplexity on domain {domain} using moerged model: {perplexity:.4f}")
        perplexities[expert_id][domain] = perplexity


def btx_training(moerged_model, moerged_loader, args):
    for name, param in moerged_model.named_parameters():
        param.requires_grad = ("router" in name)
    loss_fn, optimizer, scheduler = get_training_strategy(moerged_model, args.lr, warmup_steps=10, num_steps=args.num_steps)
    device, world_size, amp_dtype, is_main = distributed_init()
    use_fsdp = world_size > 1 and torch.cuda.is_available()
    if use_fsdp:
        mp_conf = MixedPrecision(param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype)
        moerged_model = FSDP(
            moerged_model,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_conf,
            use_orig_params=True,
        )
    print("\nTraining BTX model...")
    train(moerged_loader, None, moerged_model, loss_fn, optimizer, scheduler, device, amp_dtype, use_fsdp,
            num_steps=args.num_steps, num_val_steps=1, grad_accum_steps=args.grad_accumulation_steps, eval_interval=100000,
            is_main=is_main, alpha=args.alpha, print_loss_interval=args.print_loss_interval, setting=args.setting)

def add_setting(state_dicts, device, args, all_dataloaders, method):

    config = None
    if args.num_parameters == 1e9:
        base_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
        config = base_model.config
    moerged_model = moerge(state_dicts[:1], device, num_parameters=args.num_parameters, domains=args.domains_order[:1], k=args.k,
                           temp=args.temp, _lambda=args._lambda, context_dim=args.num_dume_tokens, alpha=None, config=config)
    
    if args.average_experts:
        if method == 'DUME':
            for i in range(1, len(args.domains_order)):
                print(f"\n\nAdding expert of domain {args.domains_order[i]}...")
                update_model_params(moerged_model, state_dicts[i], num_models=i+1, mode='add')
        elif method == 'BTX':
            raise NotImplementedError("Averaging experts is currently only implemented for DUME.")
        else:
            raise ValueError(f"Unknown method {method}.")

    if method == 'DUME':
        collect_dume_stats(args.domains_order[:1], all_dataloaders, moerged_model, device, args.num_dume_stats_steps)
    elif method == 'BTX':
        moerged_loader = merge_datasets({k: all_dataloaders[k] for k in args.domains_order[:1]}, args.domains_order[:1], args.seed)
        btx_training(moerged_model, moerged_loader, args)
    else:
        raise ValueError(f"Unknown method {method}.")
    
    expert_id = 0
    perplexities = {}
    evaluate(all_dataloaders, moerged_model, device, args.num_test_steps, perplexities, DEFAULT_ORDER[args.scenario], expert_id=expert_id)
    expert_id += 1

    while expert_id < len(args.domains_order):
        print(f"\n\nAdding expert of domain {args.domains_order[expert_id]}...")
        if args.update_model:
            update_model_params(moerged_model, state_dicts[expert_id], num_models=expert_id+1, mode='add')
        moerged_model.add_expert(state_dicts[expert_id])  # type: ignore
        if method == 'DUME':
            update_dume_stats(moerged_model, expert_id, args.domains_order, all_dataloaders, args.num_dume_stats_steps, device)
        elif method == 'BTX':
            moerged_loader = merge_datasets({k: all_dataloaders[k] for k in args.domains_order[:expert_id+1]}, args.domains_order[:expert_id+1], args.seed)
            btx_training(moerged_model, moerged_loader, args)
        else:
            raise ValueError(f"Unknown method {method}.")
        
        evaluate(all_dataloaders, moerged_model, device, args.num_test_steps, perplexities, DEFAULT_ORDER[args.scenario], expert_id=expert_id)
        expert_id += 1

    return perplexities


def remove_setting(state_dicts, device, args, all_dataloaders, method):

    config = None
    if args.num_parameters == 1e9:
        base_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
        config = base_model.config
    moerged_model = moerge(state_dicts, device, num_parameters=args.num_parameters, domains=args.domains_order, k=args.k,
                           temp=args.temp, _lambda=args._lambda, context_dim=args.num_dume_tokens, alpha=None, config=config)
    
    if method == 'DUME':
        collect_dume_stats(args.domains_order, all_dataloaders, moerged_model, device, args.num_dume_stats_steps)
    elif method == 'BTX':
        moerged_loader = merge_datasets({k: all_dataloaders[k] for k in args.domains_order}, args.domains_order, args.seed)
        btx_training(moerged_model, moerged_loader, args)
    else:
        raise ValueError(f"Unknown method {method}.")

    expert_id = len(args.domains_order)
    perplexities = {}
    evaluate(all_dataloaders, moerged_model, device, args.num_test_steps, perplexities, DEFAULT_ORDER[args.scenario], expert_id=expert_id-1)

    while expert_id > 1:
        expert_id -= 1
        print(f"\n\nRemoving expert of domain {args.domains_order[expert_id]}...")
        if args.update_model:
            if method == 'DUME':
                update_model_params(moerged_model, state_dicts[expert_id], num_models=expert_id+1, mode='remove')
            elif method == 'BTX':
                raise NotImplementedError("Updating model parameters when removing experts is currently only implemented for DUME.")
            else:
                raise ValueError(f"Unknown method {method}.")
        moerged_model.remove_last_expert()  # type: ignore
        if method == 'DUME':
            update_dume_stats(moerged_model, expert_id, args.domains_order, all_dataloaders, args.num_dume_stats_steps, device, mode='remove')
        evaluate(all_dataloaders, moerged_model, device, args.num_test_steps, perplexities, DEFAULT_ORDER[args.scenario], expert_id=expert_id-1)

    return perplexities


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args()

    # Loading all the dense experts and datasets
    state_dicts = []
    all_dataloaders = {}
    for domain in args.domains_order:
        state_dicts.append(torch.load(f"checkpoints/llama_{int(args.num_parameters)}_{domain}_seed={args.seed}{'_freeze_attn' if args.freeze_attn else ''}.pth", map_location='cpu'))
        if args.scenario == 'clm':
            all_dataloaders[domain] = get_dataloaders(dataset_name=domain, batch_size=args.batch_size, num_workers=args.num_workers)
        elif args.scenario == 'clm2':
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
            tokenizer.pad_token = tokenizer.eos_token
            all_dataloaders[domain] = get_clm2_dataloaders(domain, args.batch_size, args.num_workers, tokenizer, args.max_num_tokens, args.max_num_samples)
        else:
            raise ValueError(f"Unknown scenario {args.scenario}.")

    if args.setting == 'add':
        perplexities = add_setting(state_dicts, device, args, all_dataloaders, args.method)
    elif args.setting == 'remove':
        perplexities = remove_setting(state_dicts, device, args, all_dataloaders, args.method)
    else:
        raise ValueError(f"Unknown setting {args.setting}.")

    print(",".join(f"{domain}" for domain in DEFAULT_ORDER[args.scenario]))
    for expert_id in range(len(args.domains_order)) if args.setting == 'add' else range(len(args.domains_order)-1, -1, -1):
        print(','.join(str(round(perplexities[expert_id][domain], 2)) for domain in DEFAULT_ORDER[args.scenario]))


if __name__ == '__main__':
    main()
