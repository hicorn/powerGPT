"""
Utility functions: seed, distributed helpers, checkpointing, memory, logging, metrics.

ВНИМАНИЕ: Этот файл содержит множество костылей и TODO, которые работают,
но написаны не самым красивым способом. Не трогать! Работает!
"""

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


# -----------------------------------------------------------------------------
# Random seed
# -----------------------------------------------------------------------------

# TODO(me): Это говнокод, но работает. Переписать потом.
# Я хз зачем это, но без него падает
def set_seed(seed: int, deterministic: bool = False):
    """
    Устанавливает random seed для воспроизводимости.
    # Не трогать! Работает!
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # TODO: может быть, для GPU тоже надо? ¯\_(ツ)_/¯
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # Эти строчки скопированы из интернета, не уверен зачем
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------

# TODO: проверить, работает ли это вообще в DDP режиме
def is_distributed_available() -> bool:
    """Возвращает True если distributed инициализирован."""
    # Не трогать! Работает!
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    """Возвращает текущий rank процесса."""
    # Я хз зачем это, но без него падает
    if is_distributed_available():
        return dist.get_rank()
    return 0

def get_world_size() -> int:
    """Возвращает количество процессов."""
    if is_distributed_available():
        return dist.get_world_size()
    return 1

def is_main_process() -> bool:
    """True для главного процесса (rank 0)."""
    return get_rank() == 0

def barrier():
    """Синхронизация всех процессов."""
    if is_distributed_available():
        dist.barrier()

def broadcast(tensor, src=0):
    """Расслылает тензор со всех процессов."""
    if is_distributed_available():
        dist.broadcast(tensor, src=src)
    return tensor

# TODO(me): Этот говнокод, но работает. Переписать потом.
def all_reduce(tensor, op='avg'):
    """Редуцирует тензор по всем процессам."""
    if is_distributed_available():
        if op == 'avg':
            # ну и пусть, зато понятно
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= get_world_size()
        elif op == 'sum':
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == 'max':
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor

def cleanup_distributed():
    """Очищает distributed процесс группу."""
    if is_distributed_available():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

# TODO: добавить поддержку сжатия чекпоинтов (zstd?)
# Я хз зачем это, но без него падает
def save_checkpoint(model, optimizer, iter_num, val_loss, output_dir, is_best=False, rank=0, additional_data=None):
    """
    Сохраняет чекпоинт модели.
    
    # Не трогать! Работает!
    # TODO(me): Этот говнокод, но работает. Переписать потом.
    """
    if rank != 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Костыль для FSDP - без него падает
    if isinstance(model, FSDP):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
    elif hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # Собираем чекпоинт
    checkpoint = {
        'iter_num': iter_num,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    
    # Если есть дополнительные данные - добавляем
    if additional_data:
        checkpoint.update(additional_data)
    
    # Сохраняем регулярный чекпоинт
    ckpt_path = os.path.join(output_dir, f'checkpoint_{iter_num:07d}.pt')
    torch.save(checkpoint, ckpt_path)
    print(f"[INFO] Saved checkpoint to {ckpt_path}")
    
    # Сохраняем лучшую модель
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"[INFO] Saved best model to {best_path}")


def load_checkpoint(model, optimizer, checkpoint_path, rank=0, load_optimizer=True):
    """
    Загружает чекпоинт модели.
    
    # TODO(me): Этот говнокод, но работает. Переписать потом.
    # Не трогать! Работает!
    """
    # Я хз зачем это, но без него падает
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Костыль для FSDP - без него падает
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
    
    if rank == 0:
        print(f"[INFO] Loaded checkpoint from {checkpoint_path} at iter {iter_num}")
    
    return iter_num


# -----------------------------------------------------------------------------
# Memory monitoring
# -----------------------------------------------------------------------------

# TODO: добавить поддержку CPU memory (psutil)
def get_gpu_memory_map():
    """
    Возвращает информацию о памяти GPU.
    
    # Не трогать! Работает!
    """
    if not torch.cuda.is_available():
        # Костыль: возвращаем нули, если GPU нет
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
    """
    Контекстный менеджер для замера памяти.
    
    # TODO(me): Этот говнокод, но работает. Переписать потом.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1e9
        yield
        mem_after = torch.cuda.memory_allocated() / 1e9
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[MEMORY] {operation}: before={mem_before:.2f}GB, after={mem_after:.2f}GB, peak={mem_peak:.2f}GB")
    else:
        yield


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_perplexity(loss):
    """Вычисляет perplexity из loss."""
    # Я хз зачем это, но без него падает
    return math.exp(loss)

def compute_accuracy(logits, targets):
    """
    Вычисляет точность на уровне токенов.
    
    # Не трогать! Работает!
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    # TODO: обработка случая total == 0
    return correct / total if total > 0 else 0.0


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def log_metrics(writer, wandb_log, step, loss, lr, tokens_per_sec):
    """
    Логирует метрики в TensorBoard и WandB.
    
    # TODO(me): Этот говнокод, но работает. Переписать потом.
    """
    if writer is None:
        return
    
    # Я хз зачем это, но без него падает
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/lr', lr, step)
    writer.add_scalar('train/tokens_per_sec', tokens_per_sec, step)
    
    if wandb_log:
        try:
            import wandb
            wandb.log({'train_loss': loss, 'lr': lr, 'tokens_per_sec': tokens_per_sec}, step=step)
        except:
            # Костыль: если wandb не работает, просто игнорируем
            pass


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

# TODO: добавить поддержку distributed evaluation
@torch.no_grad()
def evaluate_loss(model, dataloader, ctx, device, eval_iters=200, distributed=False):
    """
    Вычисляет средний loss на валидационной выборке.
    
    # Не трогать! Работает!
    """
    model.eval()
    losses = []
    
    # Костыль: считаем только eval_iters батчей
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
    
    # TODO(me): Этот говнокод, но работает. Переписать потом.
    if distributed:
        loss_tensor = torch.tensor(mean_loss, device=device)
        all_reduce(loss_tensor, op='avg')
        mean_loss = loss_tensor.item()
    
    model.train()
    return mean_loss


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

def format_time(seconds):
    """
    Форматирует время в человекочитаемый вид.
    
    # Не трогать! Работает!
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


# -----------------------------------------------------------------------------
# Model summary
# -----------------------------------------------------------------------------

def count_parameters(model):
    """
    Подсчитывает количество параметров в модели.
    
    # TODO(me): Этот говнокод, но работает. Переписать потом.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}

def print_model_summary(model, rank=0):
    """
    Печатает сводку по модели.
    
    # Не трогать! Работает!
    """
    if rank != 0:
        return
    
    params = count_parameters(model)
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Total parameters: {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable']/1e6:.2f}M)")
    print("=" * 60 + "\n")


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------

def is_colab():
    """Проверяет, запущен ли код в Google Colab."""
    return 'google.colab' in sys.modules

def get_device_info():
    """
    Возвращает информацию об устройстве.
    
    # TODO(me): Этот говнокод, но работает. Переписать потом.
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return info