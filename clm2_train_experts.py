import os
import argparse

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy
from transformers import set_seed, AutoTokenizer, LlamaForCausalLM

from clm_trainer import save_checkpoint, train, eval
from models.llama_experts import get_llama_expert
from utils.distributed import cleanup_distributed, distributed_init
from utils.datasets_utils import get_clm2_math_dl, get_clm2_bio_legal_dl, get_clm2_dialogue_dl
from utils.optimizer import get_training_strategy


def get_args():
    parser = argparse.ArgumentParser(description='Train LLaMA experts with configurable domain')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument('--domain', type=str, default='math', choices=['math', 'bio', 'legal', 'dialogue'], help='Dataset domain to use')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--pretrained-path', type=str, default='checkpoints/llama_1000000000_pretrained.pth', help='Path to pre-trained checkpoint')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--max-num-tokens', type=int, default=1024, help='Maximum number of tokens per sample')
    parser.add_argument('--max-num-samples', type=int, default=100000, help='Maximum number of samples to use in the dataset')
    parser.add_argument('--num-parameters', type=float, default=1e9, help='Number of model parameters')
    parser.add_argument("--num-steps", type=int, default=2000, help="Total number of training steps")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--num-val-steps", type=int, default=1000, help="Number of validation steps")
    parser.add_argument("--num-test-steps", type=int, default=1000, help="Number of test evaluation steps")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument("--freeze_attn", action='store_true', help="Set to true to train only the MLP layers")
    parser.add_argument("--print-loss-interval", type=int, default=10, help="Loss printing interval in steps")
    parser.add_argument("--use-flash-attn", action='store_true', help="Whether to use flash attention implementation")
    parser.add_argument("--store-best-val", action='store_true', help="Whether to store the best validation checkpoint during training")
    args = parser.parse_args()
    return args


def load_pretrained_model(num_parameters, pretrained_path, is_main):
    model_base = LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
    model, _, max_lr = get_llama_expert(num_parameters, config=model_base.config)
    max_lr *= 0.1
    if os.path.isfile(pretrained_path):
        if is_main:
            print(f"Loading checkpoint from {pretrained_path}")
        state = torch.load(pretrained_path, map_location='cpu')
        load_res = model.load_state_dict(state, strict=False)
        if is_main:
            if getattr(load_res, 'missing_keys', None) or getattr(load_res, 'unexpected_keys', None):
                print(f"Loaded with missing keys: {getattr(load_res, 'missing_keys', [])}")
                print(f"Loaded with unexpected keys: {getattr(load_res, 'unexpected_keys', [])}")
    else:
        if is_main:
            print(f"Checkpoint path not found: {pretrained_path}")    
    return model, max_lr


def get_dataloaders(domain, batch_size, num_workers, tokenizer, max_num_tokens, max_num_samples):
    if domain == 'math':
        return get_clm2_math_dl(batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer, max_length=max_num_tokens, max_num_samples=max_num_samples)
    if domain == 'bio' or domain == 'legal':
        return get_clm2_bio_legal_dl(batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer, max_length=max_num_tokens, max_num_samples=max_num_samples, _type=domain)
    if domain == 'dialogue':
        return get_clm2_dialogue_dl(batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer, max_length=max_num_tokens, max_num_samples=max_num_samples)
    raise ValueError(f"Unsupported domain: {domain}")
        

def main():

    device, world_size, amp_dtype, is_main = distributed_init()
    args = get_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    model, max_lr = load_pretrained_model(args.num_parameters, args.pretrained_path, is_main)

    model.to(device)
    
    if args.freeze_attn:
        for name, param in model.named_parameters():
            param.requires_grad = False if 'train_linear' not in name and 'mlp' not in name else True

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

    dataloaders = get_dataloaders(domain=args.domain, batch_size=args.batch_size, num_workers=args.num_workers, tokenizer=tokenizer, max_num_tokens=args.max_num_tokens, max_num_samples=args.max_num_samples)

    loss_fn, optimizer, scheduler = get_training_strategy(model, max_lr, args.warmup_steps, args.num_steps)

    train(
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['validation'],
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        amp_dtype=amp_dtype,
        use_fsdp=use_fsdp,
        num_steps=args.num_steps,
        num_val_steps=args.num_val_steps,
        grad_accum_steps=args.grad_accum_steps,
        eval_interval=args.eval_interval,
        is_main=is_main,
        print_loss_interval=args.print_loss_interval,
        store_best_val=args.store_best_val,
        num_parameters=args.num_parameters,
        domain=args.domain,
        seed=args.seed,
        freeze_attn=args.freeze_attn,
    )

    # Final Test Eval
    eval(dataloaders['test'], model, tot_steps=args.num_test_steps, device=device, ev_type='Test', dtype=amp_dtype)
    
    if not args.store_best_val:
        save_checkpoint(is_main, use_fsdp, model, args.num_parameters, args.domain, args.seed, args.freeze_attn, args.comment)
    
    cleanup_distributed()    


if __name__ == '__main__':
    main()
