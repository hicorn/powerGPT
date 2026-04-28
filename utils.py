import os
import sys
import time
import random
import math
import logging
from contextlib import contextmanager
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
def is_distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()
def get_rank() -> int:
    if is_distributed_available():
        return dist.get_rank()
    return 0
def get_world_size() -> int:
    if is_distributed_available():
        return dist.get_world_size()
    return 1
def is_main_process() -> bool:
    return get_rank() == 0
def barrier():
    if is_distributed_available():
        dist.barrier()
def broadcast(tensor, src=0):
    if is_distributed_available():
        dist.broadcast(tensor, src=src)
    return tensor
def all_reduce(tensor, op='avg'):
    if is_distributed_available():
        if op == 'avg':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= get_world_size()
        elif op == 'sum':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == 'max':
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor
def cleanup_distributed():
    if is_distributed_available():
        dist.destroy_process_group()
def save_checkpoint(model, optimizer, iter_num, val_loss, output_dir, is_best=False, rank=0, additional_data=None):
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
    elif hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    checkpoint = {
        'iter_num': iter_num,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    if additional_data:
        checkpoint.update(additional_data)
    ckpt_path = os.path.join(output_dir, f'checkpoint_{iter_num:07d}.pt')
    torch.save(checkpoint, ckpt_path)
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
def load_checkpoint(model, optimizer, checkpoint_path, rank=0, load_optimizer=True):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint['model_state_dict'])
    elif hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iter_num = checkpoint.get('iter_num', 0)
    return iter_num
def get_gpu_memory_map():
    if not torch.cuda.is_available():
        return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0, 'total_gb': 0}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'free_gb': free,
        'total_gb': total,
    }
@contextmanager
def measure_gpu_memory(operation="operation"):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e9
        yield
        mem_after = torch.cuda.memory_allocated() / 1e9
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[MEMORY] {operation}: before={mem_before:.2f}GB, after={mem_after:.2f}GB, peak={mem_peak:.2f}GB")
    else:
        yield
def compute_perplexity(loss):
    return math.exp(loss)
def compute_accuracy(logits, targets):
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0
def log_metrics(writer, wandb_log, step, loss, lr, tokens_per_sec):
    if writer is None:
        return
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/lr', lr, step)
    writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
    if wandb_log:
        try:
            import wandb
            wandb.log({'train_loss': loss, 'lr': lr, 'tokens_per_sec': tokens_per_sec}, step=step)
        except:
            pass
@torch.no_grad()
def evaluate_loss(model, dataloader, ctx, device, eval_iters=200, distributed=False):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dataloader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    if not losses:
        return float('inf')
    mean_loss = sum(losses) / len(losses)
    if distributed:
        loss_tensor = torch.tensor(mean_loss, device=device)
        all_reduce(loss_tensor, op='avg')
        mean_loss = loss_tensor.item()
    model.train()
    return mean_loss
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}
def print_model_summary(model, rank=0):
    if rank != 0:
        return
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable']/1e6:.2f}M)")
def is_colab():
    return 'google.colab' in sys.modules
def get_device_info():
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info
