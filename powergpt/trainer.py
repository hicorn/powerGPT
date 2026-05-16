import os
import sys
import time
import math
import argparse
import signal
import traceback
from contextlib import nullcontext
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import asdict
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import wandb
from .config import ConfigManager, ModelArchConfig, TrainingConfig
from .model import GPT, TransformerBlock
from .data_pipeline import create_dataloaders
from .optimizer import create_optimizer, get_lr_schedule
from .utils import (
    set_seed,
    save_checkpoint,
    load_checkpoint,
    evaluate_loss,
    log_metrics,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    barrier,
    format_time,
    get_gpu_memory_map,
)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
class Trainer:
    def __init__(self, config: ConfigManager, local_rank: int = -1):
        self.config = config
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.device = torch.device(f'cuda:{local_rank}' if self.is_distributed else config.training.device)
        self._setup_distributed()
        self._setup_mixed_precision()
        self._setup_model()
        self._apply_lora_if_needed()
        self._setup_optimizer()
        self._setup_data()
        self._setup_logging()
        self.iter_num = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        self.total_tokens_processed = 0
        self.scaler = GradScaler(enabled=(config.training.dtype == 'fp16'))
        self.accum_step = 0
        self.memory_snapshot_interval = 1000
        self.last_memory_log = 0
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if is_main_process():
            self._print_config()
    def _setup_distributed(self):
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
    def _setup_mixed_precision(self):
        dtype_str = self.config.training.dtype
        if dtype_str == 'fp16':
            self.dtype = torch.float16
            self.ctx = torch.amp.autocast(device_type='cuda', dtype=self.dtype)
        elif dtype_str == 'bf16' and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.ctx = torch.amp.autocast(device_type='cuda', dtype=self.dtype)
        else:
            self.dtype = torch.float32
            self.ctx = nullcontext()
    def _setup_model(self):
        model = GPT(self.config.model)
        model.to(self.device)
        if self.config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        use_fsdp = self.config.training.use_fsdp and self.is_distributed and self.world_size > 1
        use_ddp = self.config.training.use_ddp and self.is_distributed and not use_fsdp
        self.raw_model = model
        if use_fsdp:
            auto_wrap_policy = lambda m: isinstance(m, TransformerBlock)
            mixed_precision = MixedPrecision(
                param_dtype=self.dtype,
                reduce_dtype=self.dtype,
                buffer_dtype=self.dtype,
            )
            self.model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                auto_wrap_policy=auto_wrap_policy,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                cpu_offload=CPUOffload(offload_params=False),
                use_orig_params=True,
                device_id=self.device,
                limit_all_gathers=True,
            )
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, CheckpointImpl.NO_REENTRANT),
                auto_wrap_policy=auto_wrap_policy,
            )
        elif use_ddp:
            self.model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
        else:
            self.model = model
    def _apply_lora_if_needed(self):
        use_lora = getattr(self.config.training, 'use_lora', False)
        if not use_lora:
            return
        if not PEFT_AVAILABLE:
            return
        raw_model = self.raw_model if hasattr(self, 'raw_model') else self.model
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=getattr(self.config.training, 'lora_r', 8),
            lora_alpha=getattr(self.config.training, 'lora_alpha', 32),
            target_modules=getattr(self.config.training, 'lora_target_modules', ["q_proj", "v_proj"]),
            lora_dropout=getattr(self.config.training, 'lora_dropout', 0.1),
            bias="none",
        )
        peft_model = get_peft_model(raw_model, lora_config)
        self.raw_model = peft_model
        if isinstance(self.model, (FSDP, DDP)):
            pass
        self.model = peft_model
    def _setup_optimizer(self):
        raw_model = getattr(self, 'raw_model', self.model)
        if isinstance(raw_model, (FSDP, DDP)):
            raw_model = raw_model.module
        self.optimizer = create_optimizer(raw_model, self.config.training)
    def _setup_data(self):
        self.train_loader, self.val_loader = create_dataloaders(
            self.config.model,
            self.config.training,
            distributed=self.is_distributed
        )
        self.train_iter = iter(self.train_loader)
    def _setup_logging(self):
        if is_main_process():
            log_dir = os.path.join(self.config.training.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)
            wandb_api_key = os.environ.get('WANDB_API_KEY')
            if wandb_api_key:
                try:
                    wandb.init(
                        project=self.config.training.wandb_project,
                        entity=self.config.training.wandb_entity,
                        config=self.config.to_dict(),
                        name=f"run_{int(time.time())}",
                        resume="allow",
                    )
                    self.wandb_log = True
                except Exception:
                    self.wandb_log = False
            else:
                self.wandb_log = False
        else:
            self.writer = None
            self.wandb_log = False
    def _print_config(self):
        pass
    def _signal_handler(self, sig, frame):
        if is_main_process():
            self.save_checkpoint(is_best=False)
        cleanup_distributed()
        sys.exit(0)
    def train_step(self, micro_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        x, y = micro_batch
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        with self.ctx:
            _, loss, _ = self.model(x, y)
            loss = loss / self.config.training.gradient_accumulation_steps
        self.scaler.scale(loss).backward()
        if is_main_process() and torch.isnan(loss).any():
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0
        return loss.item() * self.config.training.gradient_accumulation_steps
    def optimizer_step(self, lr: float):
        if (self.iter_num + 1) % self.config.training.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.total_tokens_processed += (
                self.config.training.micro_batch_size *
                self.config.training.gradient_accumulation_steps *
                self.config.model.block_size
            )
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        eval_iters = self.config.training.eval_iters
        for i, (x, y) in enumerate(self.val_loader):
            if i >= eval_iters:
                break
            x, y = x.to(self.device), y.to(self.device)
            with self.ctx:
                _, loss, _ = self.model(x, y)
            losses.append(loss.item())
        if not losses:
            return float('inf')
        mean_loss = sum(losses) / len(losses)
        if self.is_distributed:
            loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = loss_tensor.item() / self.world_size
        self.model.train()
        return mean_loss
    def save_checkpoint(self, is_best: bool = False):
        if not is_main_process():
            return
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        if isinstance(self.model, FSDP):
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state = self.model.state_dict()
        elif isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        checkpoint = {
            'iter_num': self.iter_num,
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'total_tokens_processed': self.total_tokens_processed,
        }
        ckpt_path = os.path.join(self.config.training.output_dir, f'checkpoint_{self.iter_num:07d}.pt')
        torch.save(checkpoint, ckpt_path)
        if is_best:
            best_path = os.path.join(self.config.training.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                self.model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.iter_num = checkpoint.get('iter_num', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.total_tokens_processed = checkpoint.get('total_tokens_processed', 0)
    def log_metrics(self, loss: float, lr: float, tokens_per_sec: float):
        if not is_main_process():
            return
        self.writer.add_scalar('train/loss', loss, self.iter_num)
        self.writer.add_scalar('train/lr', lr, self.iter_num)
        self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, self.iter_num)
        self.writer.add_scalar('train/total_tokens', self.total_tokens_processed, self.iter_num)
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            self.writer.add_scalar('memory/allocated_gb', mem_allocated, self.iter_num)
            self.writer.add_scalar('memory/reserved_gb', mem_reserved, self.iter_num)
        if self.wandb_log:
            wandb.log({
                'train_loss': loss,
                'lr': lr,
                'tokens_per_sec': tokens_per_sec,
                'total_tokens': self.total_tokens_processed,
                'iter': self.iter_num,
            }, step=self.iter_num)
    def log_eval(self, val_loss: float):
        if not is_main_process():
            return
        self.writer.add_scalar('val/loss', val_loss, self.iter_num)
        if self.wandb_log:
            wandb.log({'val_loss': val_loss}, step=self.iter_num)
    def log_memory(self):
        if not is_main_process():
            return
        if torch.cuda.is_available() and (self.iter_num - self.last_memory_log) >= self.memory_snapshot_interval:
            mem_map = get_gpu_memory_map()
            self.writer.add_scalar('memory/used_gb', mem_map.get('allocated_gb', 0), self.iter_num)
            self.last_memory_log = self.iter_num
    def train(self, resume_from: Optional[str] = None):
        set_seed(self.config.training.seed)
        if resume_from:
            self.load_checkpoint(resume_from)
        while self.iter_num < self.config.training.max_iters:
            try:
                micro_batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                self.epoch += 1
                micro_batch = next(self.train_iter)
            lr = get_lr_schedule(self.iter_num, self.config.training)
            loss_scalar = self.train_step(micro_batch)
            if (self.iter_num + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer_step(lr)
            if is_main_process() and self.iter_num % self.config.training.log_interval == 0 and self.iter_num > 0:
                dt = time.time() - self.start_time
                tokens_per_sec = self.total_tokens_processed / dt if dt > 0 else 0
                self.log_metrics(loss_scalar, lr, tokens_per_sec)
                self.start_time = time.time()
            if is_main_process() and self.iter_num % self.config.training.eval_interval == 0 and self.iter_num > 0:
                val_loss = self.evaluate()
                self.log_eval(val_loss)
                if val_loss < self.best_val_loss and self.config.training.save_best:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
            if is_main_process() and self.iter_num % self.config.training.save_interval == 0 and self.iter_num > 0:
                self.save_checkpoint(is_best=False)
            self.log_memory()
            self.iter_num += 1
        if is_main_process():
            self.save_checkpoint(is_best=True)
            self.writer.close()
            if self.wandb_log:
                wandb.finish()
        cleanup_distributed()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/t4_124m.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    config = ConfigManager().load_yaml(args.config).validate()
    trainer = Trainer(config, local_rank=args.local_rank)
    trainer.train(resume_from=args.resume)
if __name__ == '__main__':
    main()
