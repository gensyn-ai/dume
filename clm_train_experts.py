import os
import argparse

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy
from transformers import set_seed

from clm_trainer import train, eval
from utils.distributed import cleanup_distributed, distributed_init
from utils.datasets_utils import get_dataloaders
from utils.optimizer import get_training_strategy
from models.llama_experts import get_llama_expert


def get_args():
    parser = argparse.ArgumentParser(description='Train LLaMA experts with configurable domain')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--domain', type=str, default='math_l1', choices=['math_l1', 'cs_l1', 'physics_l1', 'History_and_events', 'Philosophy_and_thinking'], help='Dataset domain to use')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--pretrained-path', type=str, default='checkpoints/llama_115000000_pretrained.pth', help='Path to pre-trained checkpoint')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-parameters', type=float, default=1.15e8, help='Number of model parameters')
    parser.add_argument("--num-steps", type=int, default=1000, help="Total number of training steps")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--num-val-steps", type=int, default=2000, help="Number of validation steps")
    parser.add_argument("--num-test-steps", type=int, default=4000, help="Number of test evaluation steps")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Evaluation interval in steps")
    args = parser.parse_args()
    return args


def main():

    device, world_size, amp_dtype, is_main = distributed_init()
    args = get_args()

    set_seed(args.seed)
    model, _, max_lr = get_llama_expert(args.num_parameters)
    max_lr *= 0.1
    
    if os.path.isfile(args.pretrained_path):
        if is_main:
            print(f"Loading checkpoint from {args.pretrained_path}")
        state = torch.load(args.pretrained_path, map_location='cpu')
        load_res = model.load_state_dict(state, strict=False)
        if is_main:
            if getattr(load_res, 'missing_keys', None) or getattr(load_res, 'unexpected_keys', None):
                print(f"Loaded with missing keys: {getattr(load_res, 'missing_keys', [])}")
                print(f"Loaded with unexpected keys: {getattr(load_res, 'unexpected_keys', [])}")
    else:
        if is_main:
            print(f"Checkpoint path not found: {args.pretrained_path}")

    model.to(device)

    model.train()

    use_fsdp = world_size > 1 and torch.cuda.is_available()
    if use_fsdp:
        mp_conf = MixedPrecision(param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype)
        model = FSDP(
            model,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_conf,
            use_orig_params=True,
        )

    dataloaders = get_dataloaders(dataset_name=args.domain, batch_size=args.batch_size, num_workers=args.num_workers)
    loss_fn, optimizer, scheduler = get_training_strategy(model, max_lr, args.warmup_steps, args.num_steps)

    train(dataloaders['train'], dataloaders['validation'], model,
          loss_fn, optimizer, scheduler,
          device, amp_dtype, use_fsdp,
          args.num_steps, args.num_val_steps, args.grad_accum_steps, args.eval_interval, is_main)

    # Final Test Eval
    eval(dataloaders['test'], model, tot_steps=args.num_test_steps, device=device, ev_type='Test', dtype=amp_dtype)

    # Save checkpoint on rank 0
    if is_main:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
    if use_fsdp:
        # Gather a full state dict on rank 0 only
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_state_dict = model.state_dict()
        if is_main:
            torch.save(full_state_dict, f'checkpoints/llama_{int(args.num_parameters)}_{args.domain}_seed={args.seed}.pth')
    else:
        if is_main:
            torch.save(model.state_dict(), f'checkpoints/llama_{int(args.num_parameters)}_{args.domain}_seed={args.seed}.pth')

    cleanup_distributed()
    

if __name__ == "__main__":
    main()
