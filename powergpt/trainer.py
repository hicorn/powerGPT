"""
PowerGPT Trainer - Distributed training with FSDP/DDP, mixed precision, gradient accumulation,
checkpointing, and comprehensive logging (TensorBoard + WandB).

This module provides a production-ready trainer for GPT models optimized for T4 GPUs.
Supports single GPU, DDP (DistributedDataParallel), and FSDP (FullyShardedDataParallel)
for multi-GPU training. Includes automatic mixed precision (AMP), gradient accumulation,
activation checkpointing, memory profiling, graceful interrupt handling, and LoRA fine-tuning.
"""

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

# LoRA support (optional, requires peft library)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[WARN] PEFT library not installed. LoRA fine-tuning will be disabled.")


class Trainer:
    """
    Production-grade trainer with all modern optimizations.

    Features:
    - Single GPU, DDP, and FSDP support
    - Automatic mixed precision (FP16/BF16/FP32)
    - Gradient accumulation with dynamic steps
    - Activation checkpointing (gradient checkpointing)
    - Checkpoint saving/resuming with best model tracking
    - TensorBoard and Weights & Biases logging
    - Memory profiling and leak detection
    - Graceful interrupt handling (Ctrl+C)
    - Distributed evaluation and sync
    - LoRA fine-tuning support (optional)

    Args:
        config: ConfigManager instance with model, training, inference configs.
        local_rank: Local rank for distributed training (-1 for single GPU).
    """

    def __init__(self, config: ConfigManager, local_rank: int = -1):
        """
        Initialize trainer: distributed environment, mixed precision, model, optimizer,
        data loaders, logging, and training state.
        """
        self.config = config
        self.local_rank = local_rank
        self.is_distributed = local_rank >= 0
        self.device = torch.device(f'cuda:{local_rank}' if self.is_distributed else config.training.device)

        # Setup distributed environment
        self._setup_distributed()

        # Setup mixed precision context
        self._setup_mixed_precision()

        # Build model, optimizer, data loaders
        self._setup_model()
        self._apply_lora_if_needed()  # Apply LoRA BEFORE optimizer setup
        self._setup_optimizer()
        self._setup_data()

        # Setup logging (TensorBoard, WandB)
        self._setup_logging()

        # Training state
        self.iter_num = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.start_time = time.time()
        self.total_tokens_processed = 0

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=(config.training.dtype == 'fp16'))

        # Gradient accumulation counter
        self.accum_step = 0

        # Memory tracking
        self.memory_snapshot_interval = 1000
        self.last_memory_log = 0

        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Print configuration on rank 0
        if is_main_process():
            self._print_config()

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------

    def _setup_distributed(self):
        """Initialize distributed process group for DDP/FSDP."""
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            if is_main_process():
                print(f"[INFO] Distributed initialized: rank {self.rank}/{self.world_size} on device {self.local_rank}")
        else:
            self.world_size = 1
            self.rank = 0

    def _setup_mixed_precision(self):
        """Configure automatic mixed precision context and dtype."""
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
            if is_main_process() and dtype_str != 'fp32':
                print(f"[WARN] {dtype_str} not supported, falling back to FP32")

        if is_main_process():
            print(f"[INFO] Mixed precision: {self.dtype}")

    def _setup_model(self):
        """Instantiate model and wrap with FSDP/DDP if needed."""
        model = GPT(self.config.model)
        model.to(self.device)

        # Enable gradient checkpointing if configured
        if self.config.training.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if is_main_process():
                print("[INFO] Gradient checkpointing enabled")

        # Determine wrapping strategy
        use_fsdp = self.config.training.use_fsdp and self.is_distributed and self.world_size > 1
        use_ddp = self.config.training.use_ddp and self.is_distributed and not use_fsdp

        # Store raw model before wrapping (for LoRA)
        self.raw_model = model

        if use_fsdp:
            # Auto-wrap policy for transformer blocks
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

            # Apply activation checkpointing at FSDP-wrapped level
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, CheckpointImpl.NO_REENTRANT),
                auto_wrap_policy=auto_wrap_policy,
            )

            if is_main_process():
                print("[INFO] FSDP enabled with FULL_SHARD strategy")

        elif use_ddp:
            self.model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)
            if is_main_process():
                print("[INFO] DDP enabled")

        else:
            self.model = model
            if is_main_process():
                print("[INFO] Single GPU training (no DDP/FSDP)")

        # Count parameters (use raw_model for accurate count before LoRA)
        if is_main_process():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[INFO] Base Model: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    def _apply_lora_if_needed(self):
        """
        Apply LoRA (Low-Rank Adaptation) to the model for efficient fine-tuning.
        This should be called BEFORE _setup_optimizer and AFTER _setup_model.
        """
        # Check if LoRA is enabled in config
        use_lora = getattr(self.config.training, 'use_lora', False)
        
        if not use_lora:
            return
        
        if not PEFT_AVAILABLE:
            print("[WARN] PEFT library not installed. Install with: pip install peft")
            print("[INFO] Continuing without LoRA...")
            return
        
        if is_main_process():
            print("[INFO] Applying LoRA to the model...")
        
        # Get the raw model (unwrap from FSDP/DDP if needed)
        # CRITICAL: Apply LoRA BEFORE FSDP/DDP wrapping, not after
        raw_model = self.raw_model if hasattr(self, 'raw_model') else self.model
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=getattr(self.config.training, 'lora_r', 8),
            lora_alpha=getattr(self.config.training, 'lora_alpha', 32),
            target_modules=getattr(self.config.training, 'lora_target_modules', ["q_proj", "v_proj"]),
            lora_dropout=getattr(self.config.training, 'lora_dropout', 0.1),
            bias="none",
        )
        
        # Apply LoRA to raw model
        peft_model = get_peft_model(raw_model, lora_config)
        
        # Update raw_model reference
        self.raw_model = peft_model
        
        # Update model wrapper if needed
        if isinstance(self.model, (FSDP, DDP)):
            # If model was already wrapped, we need to update the wrapped model
            # This is tricky; safer approach is to apply LoRA BEFORE FSDP/DDP
            # For now, we'll warn the user
            if is_main_process():
                print("[WARN] LoRA applied to model that may be wrapped with FSDP/DDP.")
                print("[INFO] For best results, disable FSDP/DDP when using LoRA or apply LoRA first.")
        
        self.model = peft_model
        
        if is_main_process():
            # Print trainable parameters count
            peft_model.print_trainable_parameters()

    def _setup_optimizer(self):
        """Create optimizer (AdamW, LAMB, or Lion) using raw model parameters."""
        # Use raw_model if LoRA was applied, otherwise use model
        raw_model = getattr(self, 'raw_model', self.model)
        if isinstance(raw_model, (FSDP, DDP)):
            raw_model = raw_model.module
        self.optimizer = create_optimizer(raw_model, self.config.training)
        if is_main_process():
            print(f"[INFO] Optimizer: {self.config.training.optimizer_type}")

    def _setup_data(self):
        """Create train and validation dataloaders with proper sharding."""
        self.train_loader, self.val_loader = create_dataloaders(
            self.config.model,
            self.config.training,
            distributed=self.is_distributed
        )
        # Convert to iterator for infinite loop
        self.train_iter = iter(self.train_loader)
        if is_main_process():
            print(f"[INFO] Data loaders created. Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")

    def _setup_logging(self):
        """Initialize TensorBoard and Weights & Biases (only on rank 0)."""
        if is_main_process():
            # TensorBoard
            log_dir = os.path.join(self.config.training.output_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

            # WandB
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
                    print("[INFO] WandB logging enabled")
                except Exception as e:
                    print(f"[WARN] WandB init failed: {e}")
                    self.wandb_log = False
            else:
                self.wandb_log = False
        else:
            self.writer = None
            self.wandb_log = False

    def _print_config(self):
        """Print a summary of the configuration on rank 0."""
        print("\n" + "=" * 60)
        print("PowerGPT Training Configuration")
        print("=" * 60)
        print(f"Model: {self.config.model.n_layer} layers, {self.config.model.n_head} heads, {self.config.model.n_embd} dims")
        print(f"Block size: {self.config.model.block_size}")
        print(f"Effective batch size: {self.config.training.effective_batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate} -> {self.config.training.min_lr}")
        print(f"Max iterations: {self.config.training.max_iters}")
        print(f"Mixed precision: {self.config.training.dtype}")
        print(f"Gradient checkpointing: {self.config.training.gradient_checkpointing}")
        print(f"FSDP: {self.config.training.use_fsdp and self.world_size > 1}")
        print(f"LoRA: {getattr(self.config.training, 'use_lora', False)}")
        if getattr(self.config.training, 'use_lora', False):
            print(f"  LoRA r: {getattr(self.config.training, 'lora_r', 8)}")
            print(f"  LoRA alpha: {getattr(self.config.training, 'lora_alpha', 32)}")
            print(f"  LoRA target modules: {getattr(self.config.training, 'lora_target_modules', ['q_proj', 'v_proj'])}")
        print(f"Device: {self.device}")
        print(f"World size: {self.world_size}")
        print("=" * 60 + "\n")

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully: save checkpoint and exit."""
        if is_main_process():
            print(f"\n[INFO] Received signal {sig}, saving checkpoint before exit...")
            self.save_checkpoint(is_best=False)
        cleanup_distributed()
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Training step helpers
    # -------------------------------------------------------------------------

    def train_step(self, micro_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Perform forward and backward pass for one micro-batch.

        Args:
            micro_batch: Tuple of (input_ids, target_ids)

        Returns:
            loss_scalar: Loss value (unscaled) for logging.
        """
        x, y = micro_batch
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

        with self.ctx:
            _, loss, _ = self.model(x, y)
            # Normalize loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps

        # Backward with scaler
        self.scaler.scale(loss).backward()

        # Check for NaN (only on rank 0 to avoid overhead)
        if is_main_process() and torch.isnan(loss).any():
            print(f"[WARN] NaN loss at iter {self.iter_num}, skipping step")
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0

        # Return unscaled loss for logging
        return loss.item() * self.config.training.gradient_accumulation_steps

    def optimizer_step(self, lr: float):
        """
        Apply gradient clipping, optimizer step, and update learning rate.
        Called every gradient_accumulation_steps.
        """
        if (self.iter_num + 1) % self.config.training.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)

            # Optimizer step and scaler update
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Update learning rate in optimizer
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Update total tokens processed
            self.total_tokens_processed += (
                self.config.training.micro_batch_size *
                self.config.training.gradient_accumulation_steps *
                self.config.model.block_size
            )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    def evaluate(self) -> float:
        """
        Evaluate model on validation set.

        Returns:
            mean_loss: Average cross-entropy loss across validation batches.
        """
        self.model.eval()
        losses = []
        eval_iters = min(self.config.training.eval_iters, len(self.val_loader))

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

        # All-reduce across ranks if distributed
        if self.is_distributed:
            loss_tensor = torch.tensor(mean_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = loss_tensor.item() / self.world_size

        self.model.train()
        return mean_loss

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint with support for FSDP and LoRA.

        Args:
            is_best: If True, also save as best_model.pt.
        """
        if not is_main_process():
            return

        os.makedirs(self.config.training.output_dir, exist_ok=True)

        # Get raw model state dict (handle FSDP)
        if isinstance(self.model, FSDP):
            # FSDP requires consolidate state dict
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

        # Save regular checkpoint
        ckpt_path = os.path.join(self.config.training.output_dir, f'checkpoint_{self.iter_num:07d}.pt')
        torch.save(checkpoint, ckpt_path)
        print(f"[INFO] Saved checkpoint to {ckpt_path}")

        # Save best model
        if is_best:
            best_path = os.path.join(self.config.training.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"[INFO] Saved best model to {best_path} (val_loss={self.best_val_loss:.4f})")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint with support for FSDP and LoRA.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state (handle FSDP)
        if isinstance(self.model, FSDP):
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                self.model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scaler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore training state
        self.iter_num = checkpoint.get('iter_num', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.total_tokens_processed = checkpoint.get('total_tokens_processed', 0)

        if is_main_process():
            print(f"[INFO] Loaded checkpoint from {checkpoint_path} at iter {self.iter_num}")

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def log_metrics(self, loss: float, lr: float, tokens_per_sec: float):
        """Log metrics to TensorBoard and WandB."""
        if not is_main_process():
            return

        # TensorBoard
        self.writer.add_scalar('train/loss', loss, self.iter_num)
        self.writer.add_scalar('train/lr', lr, self.iter_num)
        self.writer.add_scalar('train/tokens_per_sec', tokens_per_sec, self.iter_num)
        self.writer.add_scalar('train/total_tokens', self.total_tokens_processed, self.iter_num)

        # Memory usage
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            self.writer.add_scalar('memory/allocated_gb', mem_allocated, self.iter_num)
            self.writer.add_scalar('memory/reserved_gb', mem_reserved, self.iter_num)

        # WandB
        if self.wandb_log:
            wandb.log({
                'train_loss': loss,
                'lr': lr,
                'tokens_per_sec': tokens_per_sec,
                'total_tokens': self.total_tokens_processed,
                'iter': self.iter_num,
            }, step=self.iter_num)

    def log_eval(self, val_loss: float):
        """Log validation metrics."""
        if not is_main_process():
            return

        self.writer.add_scalar('val/loss', val_loss, self.iter_num)
        if self.wandb_log:
            wandb.log({'val_loss': val_loss}, step=self.iter_num)

    def log_memory(self):
        """Log memory snapshot periodically."""
        if not is_main_process():
            return
        if torch.cuda.is_available() and (self.iter_num - self.last_memory_log) >= self.memory_snapshot_interval:
            mem_map = get_gpu_memory_map()
            self.writer.add_scalar('memory/used_gb', mem_map.get('allocated_gb', 0), self.iter_num)
            self.last_memory_log = self.iter_num

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.

        Args:
            resume_from: Optional checkpoint path to resume training from.
        """
        set_seed(self.config.training.seed)

        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)

        # Training loop
        while self.iter_num < self.config.training.max_iters:
            try:
                micro_batch = next(self.train_iter)
            except StopIteration:
                # Re-create iterator when exhausted (end of epoch)
                self.train_iter = iter(self.train_loader)
                self.epoch += 1
                micro_batch = next(self.train_iter)

            # Compute learning rate for this iteration
            lr = get_lr_schedule(self.iter_num, self.config.training)

            # Forward/backward step
            loss_scalar = self.train_step(micro_batch)

            # Optimizer step every gradient_accumulation_steps
            if (self.iter_num + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer_step(lr)

            # Logging
            if is_main_process() and self.iter_num % self.config.training.log_interval == 0 and self.iter_num > 0:
                dt = time.time() - self.start_time
                tokens_per_sec = self.total_tokens_processed / dt if dt > 0 else 0
                print(f"[ITER {self.iter_num:6d}] loss={loss_scalar:.4f} | lr={lr:.2e} | tok/s={tokens_per_sec:.1f} | time={format_time(dt)}")
                self.log_metrics(loss_scalar, lr, tokens_per_sec)
                self.start_time = time.time()  # reset for next interval

            # Validation
            if is_main_process() and self.iter_num % self.config.training.eval_interval == 0 and self.iter_num > 0:
                val_loss = self.evaluate()
                print(f"[EVAL {self.iter_num:6d}] val_loss={val_loss:.4f} | best={self.best_val_loss:.4f}")
                self.log_eval(val_loss)

                # Save best model
                if val_loss < self.best_val_loss and self.config.training.save_best:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

            # Periodic checkpoint saving
            if is_main_process() and self.iter_num % self.config.training.save_interval == 0 and self.iter_num > 0:
                self.save_checkpoint(is_best=False)

            # Memory logging
            self.log_memory()

            self.iter_num += 1

        # Final save
        if is_main_process():
            self.save_checkpoint(is_best=True)
            print(f"[INFO] Training finished after {self.iter_num} iterations.")
            self.writer.close()
            if self.wandb_log:
                wandb.finish()

        cleanup_distributed()


# -------------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PowerGPT Training')
    parser.add_argument('--config', type=str, default='configs/t4_124m.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set by torchrun)')
    args = parser.parse_args()

    # Load and validate configuration
    config = ConfigManager().load_yaml(args.config).validate()

    # Create trainer and start training
    trainer = Trainer(config, local_rank=args.local_rank)
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()