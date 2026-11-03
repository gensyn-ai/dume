from collections import defaultdict
import os
import glob
import random

import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info, DataLoader, Dataset, Subset
from datasets.io.parquet import ParquetDatasetReader
from tqdm import tqdm
from datasets import load_dataset


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


class PileDataset(Dataset):

    def __init__(self, prompts, tokenizer, max_length=1024):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()[2:]
        input_ids = input_ids[1:-1]
        return {"input_ids": input_ids, "label": labels}

def get_pile_statistics(dataset):
    counts = defaultdict(int)
    for example in dataset:
        counts[example['meta']['pile_set_name']] += 1
    counts = list(dict(counts).items())
    counts.sort(key=lambda x: x[1], reverse=True)
    for domain, count in counts:
        print(f"{domain}: {count}")
    print("----------------------------------")


def get_pile_domain_datasets(dataset, domains, tokenizer):
    all_train_datasets = {domain: [] for domain in domains}
    for example in tqdm(dataset):
        if example['meta']['pile_set_name'] not in domains:
            continue
        all_train_datasets[example['meta']['pile_set_name']].append(example['text'])
    datasets = {domain: PileDataset(prompts=texts, tokenizer=tokenizer) for domain, texts in tqdm(all_train_datasets.items())}
    return datasets


def get_pile_domain_dataloaders(tokenizer, batch_size, num_workers, seed):

    random.seed(seed)
    domains = ['Pile-CC', 'OpenWebText2', 'StackExchange', 'PubMed Abstracts', 'Github', 'Wikipedia (en)', 'USPTO Backgrounds']
    dataset = load_dataset("EleutherAI/pile_val_test")
    validation_dataset = dataset['validation']
    
    train_datasets = get_pile_domain_datasets(dataset['validation'], domains, tokenizer)
    test_datasets = get_pile_domain_datasets(dataset['test'], domains, tokenizer)
    
    dataloaders = {domain: {} for domain in domains}

    for domain in domains:
        
        validation_dataset = train_datasets[domain]
        total_size = len(validation_dataset)
        train_size = int(0.85 * total_size)
        indices = list(range(total_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_subset = Subset(validation_dataset, train_indices)
        val_subset = Subset(validation_dataset, val_indices)

        dataloaders[domain]['train'] = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
        dataloaders[domain]['validation'] = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
        dataloaders[domain]['test'] = DataLoader(test_datasets[domain], batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    return dataloaders


def preprocess_samples(tokenizer, key, max_length):
    def preprocess_function(examples):
        out = tokenizer(examples[key], max_length=max_length+1, padding='max_length', truncation=True, return_tensors="pt")
        return {
            "input_ids": out['input_ids'][:, :-1],
            "label": out['input_ids'].clone()[:, 1:],
            "attention_mask": out['attention_mask'][:, :-1]}
    return preprocess_function


def get_clm2_math_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples):
    
    dataset = load_dataset("open-web-math/open-web-math")
    total_size = len(dataset['train'])
    
    assert max_num_samples <= len(dataset['train']), "max_num_samples cannot be greater than the total size of the dataset"
    all_indices = list(range(total_size))
    random.shuffle(all_indices)
    indices = all_indices[:max_num_samples]
    dataset = dataset['train'].select(indices)
    total_size = max_num_samples
    
    preprocess_function = preprocess_samples(tokenizer, key="text", max_length=max_length)
    dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
    
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),  # type: ignore
        'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),  # type: ignore
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)  # type: ignore
    }

    return dataloaders
    

def get_clm2_bio_legal_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples, _type='bio'):
    
    if _type == 'bio':
        dataset = load_dataset("allenai/peS2o")
    elif _type == 'legal':
        dataset = load_dataset("pile-of-law/pile-of-law", "all")
    else:
        raise ValueError(f"Unknown dataset type: {_type}")

    total_size = len(dataset['train']) + len(dataset['validation'])
    
    assert max_num_samples <= total_size, "max_num_samples cannot be greater than the total size of the dataset"

    all_train_indices = list(range(len(dataset['train'])))
    random.shuffle(all_train_indices)
    
    train_indices = all_train_indices[:int(0.7 * max_num_samples)]
    test_indices = all_train_indices[int(0.7 * max_num_samples):int(0.9 * max_num_samples)]
    train_dataset = dataset['train'].select(train_indices)
    test_dataset = dataset['train'].select(test_indices)
    
    all_val_indices = list(range(len(dataset['validation'])))
    random.shuffle(all_val_indices)
    
    val_indices = all_val_indices[:int(0.1 * max_num_samples)]
    val_dataset = dataset['validation'].select(val_indices)

    preprocess_function = preprocess_samples(tokenizer, key="text", max_length=max_length)
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
    val_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),  # type: ignore
        'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),  # type: ignore
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)  # type: ignore
    }
    return dataloaders


def get_clm2_dialogue_dl(batch_size, num_workers, tokenizer, max_length, max_num_samples):
    
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    dataset = dataset.map(lambda x: {
        "text": " ".join([m["content"] for m in x["messages"]])
    })
    
    total_size = len(dataset['train_sft']) + len(dataset['test_sft'])
    
    assert max_num_samples <= total_size, "max_num_samples cannot be greater than the total size of the dataset"

    all_train_indices = list(range(len(dataset['train_sft'])))
    random.shuffle(all_train_indices)
    
    train_indices = all_train_indices[:int(0.7 * max_num_samples)]
    val_indices = all_train_indices[int(0.7 * max_num_samples):int(0.8 * max_num_samples)]
    train_dataset = dataset['train_sft'].select(train_indices)
    val_dataset = dataset['train_sft'].select(val_indices)
    
    all_test_indices = list(range(len(dataset['test_sft'])))
    random.shuffle(all_test_indices)
    
    test_indices = all_test_indices[:int(0.2 * max_num_samples)]
    test_dataset = dataset['test_sft'].select(test_indices)

    preprocess_function = preprocess_samples(tokenizer, key="prompt", max_length=max_length)
    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train_sft"].column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train_sft"].column_names)
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=dataset["test_sft"].column_names)
    train_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
    val_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "label", "attention_mask"])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True),  # type: ignore
        'validation': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers),  # type: ignore
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)  # type: ignore
    }
    return dataloaders