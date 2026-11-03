import math
import os
from typing import List
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from utils.distributed import is_distributed, get_rank
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullyShardedDataParallel as FSDP


def save_checkpoint(is_main, use_fsdp, model, num_parameters, domain="", seed=42, freeze_attn=False, comment: str = ""):
    
    if domain != "":
        path = f'checkpoints/llama_{int(num_parameters)}_{domain}_seed={seed}{"_freeze_attn" if freeze_attn else ""}{f"_{comment}" if comment else ""}.pth'
    else:
        path = f'checkpoints/llama_{int(num_parameters)}_pretrained{f"_{comment}" if comment else ""}.pth'
    
    print("Saving checkpoint to ", path, "...")
    if is_main:
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
    if use_fsdp:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full_state_dict = model.state_dict()
        if is_main:
            torch.save(full_state_dict, path)
    else:
        if is_main:
            torch.save(model.state_dict(), path)


def train(train_dataloader: torch.utils.data.DataLoader | List, val_dataloader: torch.utils.data.DataLoader | None, model: torch.nn.Module,
          loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
          device: torch.device, amp_dtype: torch.dtype, use_fsdp: bool,
          num_steps: int, num_val_steps: int, grad_accum_steps: int, eval_interval: int, is_main: bool, alpha: float | None = None,
          print_loss_interval: int = 100, store_best_val: bool = False, num_parameters: int | None = None, domain: str = "", seed: int = 42,
          freeze_attn: bool = False, comment: str = "", model_type: str = "llama", setting: str = "clm"):

    assert not store_best_val or (store_best_val and num_parameters is not None), "num_parameters must be provided if store_best_val is True"
    
    step = 1
    acc_step = 0
    best_val_perplexity = float('inf')

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=num_steps, desc="Training") if is_main else None
    while step <= num_steps:

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            loss_load_balancing = torch.tensor(0.0, device=device)
            with torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None else torch.cuda.amp.autocast(enabled=False):
                outputs = model(input_ids=input_ids)
                if isinstance(outputs, tuple):
                    all_p = outputs[1]
                    all_u = outputs[2]
                    outputs = outputs[0]
                    num_experts = all_p[0].shape[2]
                    for i in range(len(all_p)):
                        loss_load_balancing = loss_load_balancing + torch.sum(all_p[i].mean(dim=(0,1)) * all_u[i].mean(dim=(0,1))) * num_experts * alpha
                
                if model_type == "gemma":
                    outputs = outputs.logits

                loss = loss_fn(
                    outputs.view(-1, outputs.size(-1)),  # [64*1024, vocab_size]
                    labels.view(-1)  # [64*1024]
                )
            
                if loss_load_balancing != 0:
                    loss = loss + loss_load_balancing

                if model_type == "gemma":
                    loss = 4 / grad_accum_steps * loss # preserve same gradients magnitude for num accumulation steps != 4
                elif setting == "clm2":
                    loss = loss / grad_accum_steps

            loss.backward()
            acc_step += 1

            if acc_step == grad_accum_steps:
                acc_step = 0
                if use_fsdp:
                    _ = model.clip_grad_norm_(1.0)  # type: ignore
                else:
                    clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if is_main:
                    pbar.update(1)  # type: ignore

                if is_main and step % print_loss_interval == 0:
                        print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                if step % eval_interval == 0 and step <= num_steps and val_dataloader is not None:
                        val_perplexity = eval(val_dataloader, model.eval(), num_val_steps, device, ev_type='Validation', dtype=amp_dtype)
                        if store_best_val and val_perplexity < best_val_perplexity:
                            print(f"New best validation perplexity: {val_perplexity:.4f} at step {step}, saving checkpoint...")
                            best_val_perplexity = val_perplexity
                            save_checkpoint(is_main, use_fsdp, model, num_parameters, domain=domain, seed=seed, freeze_attn=freeze_attn, comment=comment)
                
                if model_type == "gemma":
                    torch.cuda.empty_cache()

            if step > num_steps:
                break

    if is_main and pbar is not None:
        pbar.close()


def eval(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, tot_steps: int,
         device: torch.device, ev_type: str = 'Validation', dtype: torch.dtype | None = None, expert_id: int | None = None):

    model.eval()
    ppl_loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    rank = get_rank()
    is_main = rank == 0

    with torch.no_grad():
        avg_loss = 0.0
        tot_ppl_loss = 0.0
        tot_num_tokens = 0
        steps = 0
        pbar = tqdm(
            # total=tot_steps,
            desc=ev_type) if is_main else None

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=dtype) if dtype is not None else torch.cuda.amp.autocast(enabled=False):
                if expert_id is not None:
                    outputs = model(input_ids=input_ids, expert_id=expert_id)
                else:
                    outputs = model(input_ids=input_ids)
                num_tokens = labels.shape[0] * labels.shape[1]
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                ppl_loss = ppl_loss_fn(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )
            avg_loss += ppl_loss.item() / num_tokens
            tot_ppl_loss += ppl_loss.item()
            tot_num_tokens += num_tokens

            steps += 1
            if is_main:
                pbar.update(1)  # type: ignore

            if steps >= tot_steps:
                break

        if is_distributed():
            t = torch.tensor([avg_loss, tot_ppl_loss, tot_num_tokens, steps], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_loss, tot_ppl_loss, tot_num_tokens, steps = t.tolist()

        avg_loss /= steps
        perplexity = math.exp(tot_ppl_loss / tot_num_tokens)
        if is_main and pbar is not None:
            pbar.close()

    if is_main:
        print(f"{ev_type} loss: {avg_loss:.4f}, {ev_type} perplexity: {perplexity:.4f}")

    model.train()

    return perplexity
