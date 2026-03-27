import os
import glob
import random

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info, DataLoader
from datasets.io.parquet import ParquetDatasetReader


class ShardedDataset(torch.utils.data.IterableDataset):

    def __init__(self, path: str):
        self.path = path
        # Defer reader creation to __iter__ to avoid forking issues with workers
        self._reader = None

    def _ensure_reader(self):
        if self._reader is None:
            self._reader = ParquetDatasetReader(self.path, streaming=True).read()
        return self._reader

    def __iter__(self):
        reader = self._ensure_reader()

        # Discover distributed rank/world size if initialized
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Account for DataLoader workers per process
        wi = get_worker_info()
        if wi is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = wi.id
            num_workers = wi.num_workers

        # Global sharding across all ranks and workers
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        for idx, sample in enumerate(reader):
            if (idx % num_shards) != shard_id:
                continue
            yield {
                key: torch.tensor(value, dtype=torch.int64, device="cpu")
                for key, value in sample.items()
            }

    # def __len__(self):
    #     return len(self.data_reader)


def get_dataloaders(dataset_name: str, batch_size: int, num_workers: int):

    if dataset_name == 'openwebtext':
        path = 'dataset/openwebtext/raw'

    elif dataset_name == 'math_l1':
        path = 'dataset/math_l1'
    elif dataset_name == 'cs_l1':
        path = 'dataset/cs_l1'
    elif dataset_name == 'physics_l1':
        path = 'dataset/physics_l1'
    elif dataset_name == 'History_and_events':
        path = 'dataset/History_and_events'
    elif dataset_name == 'Philosophy_and_thinking':
        path = 'dataset/Philosophy_and_thinking'
                
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported.")

    # Resolve dataset directory relative to workspace root (parent of DetGatingMoE-exp)
    workspace_root = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(workspace_root, path)

    # Pre-check: ensure train split exists to avoid opaque HF error
    train_glob = os.path.join(dataset_dir, 'train', '*.parquet')
    if not glob.glob(train_glob):
        raise ValueError(
            f"No Parquet files found for train split at: {train_glob}. "
            f"Verify your dataset path and splits."
        )

    datasets = {
        'train': ShardedDataset(train_glob),
        'validation': ShardedDataset(os.path.join(dataset_dir, 'validation', '*.parquet')),
        'test': ShardedDataset(os.path.join(dataset_dir, 'test', '*.parquet'))
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, drop_last=True),
        'validation': DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    }

    return dataloaders


class BufferedShuffleConcat(torch.utils.data.IterableDataset):
    
    def __init__(self, datasets, buffer_size=2048, seed=0):
        super().__init__()
        self.datasets = datasets
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        worker_seed = self.seed if info is None else (self.seed + info.id)
        rng = random.Random(worker_seed)

        iters = []
        for ds in self.datasets:
            iters.append(iter(ds))

        buffer = []
        # Warm up buffer
        while len(buffer) < self.buffer_size and iters:
            for i in range(len(iters) - 1, -1, -1):
                try:
                    buffer.append(next(iters[i]))
                    if len(buffer) >= self.buffer_size:
                        break
                except StopIteration:
                    iters.pop(i)

        # Yield with random replacement from available iterators
        while buffer:
            idx = rng.randrange(len(buffer))
            item = buffer[idx]
            yield item
            if iters:
                j = rng.randrange(len(iters))
                try:
                    buffer[idx] = next(iters[j])
                except StopIteration:
                    iters.pop(j)
                    buffer.pop(idx)
            else:
                buffer.pop(idx)