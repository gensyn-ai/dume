import argparse
from copy import deepcopy
from tqdm import tqdm

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader

from clm_trainer import train, eval
from models.llama_experts import get_llama_expert
from utils.distributed import distributed_init
from utils.datasets_utils import BufferedShuffleConcat, get_dataloaders
from utils.utils import align_and_stack
from utils.optimizer import get_training_strategy


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of several MoErging techniques')
    parser.add_argument('--method', type=str, choices=['oracle', 'random_routing', 'DUME', 'BTX', 'DUMEplus'], required=True, help='Experiment type to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed of the loaded experts')
    parser.add_argument('--num_parameters', type=float, default=1.15e8, help='Number of parameters for the base dense model experts')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_dume_stats_steps', type=int, default=1000, help='Maximum number of steps for DUME statistics extraction')
    parser.add_argument('--num_test_steps', type=int, default=4000, help='Maximum number of steps for evaluation')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--k', type=int, default=1, help='Number of top-k experts to route')
    parser.add_argument('--temp', type=float, default=1.0, help='Softmax temperature for routing')
    parser.add_argument('--_lambda', type=float, default=1e-2, help='Lambda for DUME routers')
    parser.add_argument('--num_dume_tokens', type=int, default=10000, help='Number of tokens to use for DUME statistics collection')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for BTX or DUME+ training')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps for BTX or DUME+ training')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha value for BTX or DUME+ training. Set to None for other experiments.')
    args = parser.parse_args()
    return args


def moerge(state_dicts, device, num_parameters=1.15e8,
           domains=('cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking'),
           k=1, temp=1.0, _lambda=1e-2, context_dim=1024, config=None, alpha=None, return_moe_experts=False):

    print("MoErging the dense experts...")
    moe_experts = []
    model, _, _ = get_llama_expert(num_parameters, config=config)
    experts_names = []
    moerged_state_dict = {}
    for _k in tqdm(state_dicts[0].keys()):
        if 'mlp' in _k:
            # Keep separate experts for each domain
            expert_name = _k.rsplit('.', 2)[0]
            if expert_name in experts_names:
                continue
            experts_names.append(expert_name)
            moe_experts.append([deepcopy(model.get_submodule(expert_name)) for _ in range(len(domains))])
            for domain_idx in range(len(domains)):
                moe_experts[-1][domain_idx].load_state_dict(
                    {k2[len(expert_name)+1:] : state_dicts[domain_idx][k2] for k2 in state_dicts[domain_idx].keys() if k2.startswith(expert_name)},
                    strict=True
                )
        else:
            # Average the parameters for non-MLP layers
            tensors = [sd[_k] for sd in state_dicts]
            if len(set(t.shape for t in tensors)) != 1:
                print(f"Warning: Aligning tensors for key: {_k}, shapes: {[t.shape for t in tensors]}")
                moerged_state_dict[_k] = torch.mean(align_and_stack(tensors), dim=0)
            else:
                moerged_state_dict[_k] = torch.mean(torch.stack(tensors), dim=0)
        
    moerged_model, _, _ = get_llama_expert(config=config, num_parameters=num_parameters, moe_experts=moe_experts, k=k, temp=temp, _lambda=_lambda, context_dim=context_dim, alpha=alpha)
    moerged_model.load_state_dict(moerged_state_dict, strict=False)  # routing keys still missing at this point
    
    moerged_model.to(device)
    for expert_group in moe_experts:
        for expert in expert_group:
            expert.to(device)
            expert.eval()
    moerged_model.eval()
    
    print("OK! MoErged model created.")

    # vocab_size = moerged_model._model.config.vocab_size
    # input_ids = torch.randint(0, vocab_size, (2, 8), dtype=torch.long).to(device)
    # with torch.no_grad():
    #     logits = moerged_model(input_ids)
    #     print(logits.shape)

    if return_moe_experts:
        return moerged_model, moe_experts
    
    return moerged_model


def collect_dume_stats(all_domains, all_dataloaders, moerged_model, device, num_steps):
    moerged_model.set_rr_gate_mode(True)
    print("Collecting DUME statistics...")
    with torch.no_grad():
        for expert_id, domain in enumerate(all_domains):
            print(f"Collecting DUME statistics for domain {domain}...")
            dataloaders = all_dataloaders[domain]
            pbar = tqdm(total=num_steps, desc='DUME stats collection')
            for i, batch in enumerate(dataloaders['train']):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                moerged_model(input_ids=input_ids, expert_id=expert_id)
                pbar.update(1)
                if i + 1 >= num_steps:
                    break
            pbar.close()
    moerged_model.set_rr_gate_mode(False)


def merge_datasets(all_dataloaders, all_domains, seed):
    print("Merging datasets...")
    train_datasets = [all_dataloaders[domain]['train'].dataset for domain in all_domains]
    moerged_dataset = BufferedShuffleConcat(train_datasets, buffer_size=2048, seed=seed)
    moerged_loader = DataLoader(
        moerged_dataset,
        batch_size=min([all_dataloaders[domain]['train'].batch_size for domain in all_domains]),
        shuffle=False,  # ignored for IterableDataset
        num_workers=min([all_dataloaders[domain]['train'].num_workers for domain in all_domains]),
        collate_fn=all_dataloaders[all_domains[0]]['train'].collate_fn
    )
    return moerged_loader


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_domains = ['cs_l1', 'math_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking']
    args = get_args()

    # Loading all the dense experts and datasets
    state_dicts = []
    all_dataloaders = {}
    for domain in all_domains:
        state_dicts.append(torch.load(f"checkpoints/llama_{int(args.num_parameters)}_{domain}{f'_seed={args.seed}' if args.seed != 42 else ''}.pth", map_location='cpu'))
        all_dataloaders[domain] = get_dataloaders(dataset_name=domain, batch_size=args.batch_size, num_workers=args.num_workers)

    # MoErging all the dense experts. Randomly initializing the routers parameters for now.
    moerged_model = moerge(state_dicts, device, num_parameters=args.num_parameters, domains=all_domains, k=args.k, temp=args.temp, _lambda=args._lambda, context_dim=args.num_dume_tokens, alpha=args.alpha)
    assert not isinstance(moerged_model, tuple), "Expected moerged_model to be a single model instance."
    
    if args.method == 'oracle':
        """
        Use the average parameters for all the layers except the MLP layers. Substitute the MLP layers with MoE blocks.
        For routing, always deterministically select the expert corresponding to the input domain. Do no training, only evaluation.
        """
        assert args.alpha is None, "Alpha must be None for oracle."

    elif args.method == 'random_routing':
        """
        Use the average parameters for all the layers except the MLP layers. Substitute the MLP layers with MoE blocks. For the routing,
        use the random routing parameters. Do no training, only evaluation.
        """
        assert args.alpha is None, "Alpha must be None for random routing."
        
    elif args.method == 'DUME':
        """
        Use the average parameters for all the layers except the MLP layers. Substitute the MLP layers with MoE blocks.
        For routing, first forward all samples from each domain to the model to collect A and b DUME statistics before
        each MoE block. Then, compute optimal DUME routing weights for each domain and MoE block, and use them for evaluation.
        Do no training, only evaluation.
        """
        assert args.alpha is None, "Alpha must be None for DUME."
        collect_dume_stats(all_domains, all_dataloaders, moerged_model, device, args.num_dume_stats_steps)
        print("DUME statistics collected. Evaluating DUME model...")  # Now, evaluate using the computed weights of the DUME routers
        
    elif args.method == 'BTX':
        """
        Train and evaluate the BTX model.
        """

        moerged_loader = merge_datasets(all_dataloaders, all_domains, args.seed)

        # Freeze all layers except those with "gate" in their name
        for name, param in moerged_model.named_parameters():
            param.requires_grad = ("gate" in name)

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
              num_steps=args.num_steps, num_val_steps=1, grad_accum_steps=1, eval_interval=10, is_main=is_main, alpha=args.alpha)

        print("\nEvaluating BTX model...")

    elif args.method == 'DUMEplus':
        """Like DUME, but with an additional routers training phase after initializing the DUME routers weight."""
        
        collect_dume_stats(all_domains, all_dataloaders, moerged_model, device, args.num_dume_stats_steps)
        moerged_loader = merge_datasets(all_dataloaders, all_domains, args.seed)

        # Freeze all layers except those with "gate" in their name
        for name, param in moerged_model.named_parameters():
            param.requires_grad = ("gate" in name)

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

        print("\nTraining DUME+ model...")
        train(moerged_loader, None, moerged_model, loss_fn, optimizer, scheduler, device, amp_dtype, use_fsdp,
              num_steps=args.num_steps, num_val_steps=1, grad_accum_steps=1, eval_interval=10, is_main=is_main, alpha=args.alpha)

        print("\nEvaluating DUME+ model...")
        
    else:
        raise NotImplementedError(f"Unknown method: {args.method}")
    
    # The evaluation code is the same for all the strategies
    perplexities = {}
    for expert_id, domain in enumerate(all_domains):
        print(f"Evaluating moerged model on {domain}...")
        dataloaders = all_dataloaders[domain]
        perplexity = eval(dataloaders['test'], moerged_model, tot_steps=args.num_test_steps, device=device, ev_type='Test', dtype=torch.bfloat16,
                          expert_id=expert_id if args.method == 'oracle' else None)  # Only use expert_id for oracle routing
        print(f"Perplexity on domain {domain} using moerged model: {perplexity:.4f}")
        perplexities[domain] = perplexity
    print(",".join(f"{domain}" for domain in all_domains))
    print(','.join(str(round(perplexities[domain], 2)) for domain in all_domains))


if __name__ == '__main__':
    main()