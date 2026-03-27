import os
from datetime import timedelta

import torch
import torch.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def setup_distributed():
    if is_distributed():
        return
    # Initialize process group if launched with torchrun/torch.distributed
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl"
        dist.init_process_group(backend=backend, timeout=timedelta(hours=2))


def cleanup_distributed():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def distributed_init():

    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    rank = get_rank()
    world_size = get_world_size()
    is_main = rank == 0

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float32  # automatic mixed precision dtype

    if is_main:
        print(f"Running with world_size={world_size}, rank={rank}, device={device}, amp_dtype={amp_dtype}")

    return device, world_size, amp_dtype, is_main
