"""
Data pipeline with memory-mapped datasets and distributed sharding.
"""

import os
import random
import numpy as np
from typing import Iterator, Tuple, Optional, List
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import tiktoken


class TokenizerWrapper:
    _instance = None
    def __new__(cls, encoding='gpt2'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.tokenizer = tiktoken.get_encoding(encoding)
        return cls._instance
    def encode(self, text): return self.tokenizer.encode(text)
    def decode(self, tokens): return self.tokenizer.decode(tokens)
    @property
    def vocab_size(self): return self.tokenizer.n_vocab


class MMapTokenizedDataset(IterableDataset):
    def __init__(self, data_dir, block_size, split='train', seed=42, shuffle=True):
        self.data_dir = data_dir
        self.block_size = block_size
        self.split = split
        self.seed = seed
        self.shuffle = shuffle
        self._data = None
    @property
    def data(self):
        if self._data is None:
            path = os.path.join(self.data_dir, f'{self.split}.bin')
            self._data = np.memmap(path, dtype=np.uint16, mode='r')
        return self._data
    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_tokens = len(self.data)
        max_start = total_tokens - self.block_size
        tokens_per_rank = max_start // world_size
        rank_start = rank * tokens_per_rank
        rank_end = (rank + 1) * tokens_per_rank if rank < world_size - 1 else max_start
        tokens_per_worker = (rank_end - rank_start) // num_workers
        worker_start = rank_start + worker_id * tokens_per_worker
        worker_end = worker_start + tokens_per_worker
        indices = list(range(worker_start, worker_end, self.block_size))
        if self.shuffle:
            rng = random.Random(self.seed + rank + worker_id)
            rng.shuffle(indices)
        for start in indices:
            chunk = self.data[start:start+self.block_size+1]
            if len(chunk) == self.block_size+1:
                x = torch.from_numpy(chunk[:-1].astype(np.int64))
                y = torch.from_numpy(chunk[1:].astype(np.int64))
                yield x, y


def create_dataloaders(model_config, training_config, distributed=False):
    train_ds = MMapTokenizedDataset(training_config.data_dir, model_config.block_size, 'train', training_config.seed)
    val_ds = MMapTokenizedDataset(training_config.data_dir, model_config.block_size, 'val', training_config.seed, shuffle=False)
    # Для распределённого обучения отключаем workers
    if distributed:
        num_workers = 0
        persistent = False
    else:
        num_workers = training_config.num_workers
        persistent = getattr(training_config, 'persistent_workers', False)
    train_loader = DataLoader(train_ds, batch_size=training_config.micro_batch_size, num_workers=num_workers,
                              pin_memory=training_config.pin_memory, drop_last=True, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=training_config.micro_batch_size, num_workers=num_workers,
                            pin_memory=training_config.pin_memory, drop_last=False, persistent_workers=False)
    return train_loader, val_loader