import math
from typing import List
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from utils.distributed import is_distributed, get_rank


def train(train_dataloader: torch.utils.data.DataLoader | List, val_dataloader: torch.utils.data.DataLoader | None, model: torch.nn.Module,
          loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
          device: torch.device, amp_dtype: torch.dtype, use_fsdp: bool,
          num_steps: int, num_val_steps: int, grad_accum_steps: int, eval_interval: int, is_main: bool, alpha: float | None = None):

    step = 1
    acc_step = 0

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
                        loss_load_balancing += torch.sum(all_p[i].mean(dim=(0,1)) * all_u[i].mean(dim=(0,1))) * num_experts * alpha
                    
                loss = loss_fn(
                    outputs.view(-1, outputs.size(-1)),  # [64*1024, vocab_size]
                    labels.view(-1)  # [64*1024]
                )
            
                if loss_load_balancing != 0:
                    loss += loss_load_balancing
    
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

                if step % eval_interval == 0 and step <= num_steps:
                    if is_main:
                        print(f"Step {step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                    # Validation eval across ranks
                    if val_dataloader is not None:
                        eval(val_dataloader, model.eval(), num_val_steps, device, ev_type='Validation', dtype=amp_dtype)

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
