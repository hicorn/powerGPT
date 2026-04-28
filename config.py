from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Tuple
import torch
import yaml
import os
import argparse
@dataclass
class ModelArchConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    flash_attention: bool = True
    use_rope: bool = True
    rope_theta: float = 10000.0
    use_rms_norm: bool = True
    activation: Literal['gelu', 'swiglu'] = 'swiglu'
    use_moe: bool = False
    num_experts: int = 8
    top_k_experts: int = 2
    gradient_checkpointing: bool = True
    use_compile: bool = False
    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        self.head_dim = self.n_embd // self.n_head
@dataclass
class TrainingConfig:
    data_dir: str = "data"
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    max_iters: int = 600000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    optimizer_type: Literal['adamw', 'lamb', 'lion'] = 'adamw'
    dtype: Literal['fp16', 'bf16', 'fp32'] = 'fp16'
    label_smoothing: float = 0.0
    drop_path_rate: float = 0.0
    use_fsdp: bool = False
    use_ddp: bool = False
    fsdp_sharding_strategy: str = "full_shard"
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 200
    save_interval: int = 5000
    output_dir: str = "checkpoints"
    wandb_project: str = "powergpt"
    wandb_entity: Optional[str] = None
    save_best: bool = True
    num_workers: int = 4
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    persistent_workers: bool = False
    profile: bool = False
    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps
@dataclass
class InferenceConfig:
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = False
    use_kv_cache: bool = True
    max_kv_cache_tokens: int = 2048
    batch_size: int = 1
@dataclass
class BenchmarkConfig:
    seq_lengths: Tuple[int, ...] = (128, 256, 512, 1024)
    batch_sizes: Tuple[int, ...] = (1, 2, 4, 8)
    num_warmup: int = 10
    num_runs: int = 50
    profile_memory: bool = True
    output_json: Optional[str] = None
class ConfigManager:
    def __init__(self):
        self.model = ModelArchConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.benchmark = BenchmarkConfig()
    def load_yaml(self, path: str) -> 'ConfigManager':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        if 'model' in data:
            for k, v in data['model'].items():
                if hasattr(self.model, k):
                    setattr(self.model, k, v)
        if 'training' in data:
            for k, v in data['training'].items():
                if hasattr(self.training, k):
                    setattr(self.training, k, v)
        if 'inference' in data:
            for k, v in data['inference'].items():
                if hasattr(self.inference, k):
                    setattr(self.inference, k, v)
        if 'benchmark' in data:
            for k, v in data['benchmark'].items():
                if hasattr(self.benchmark, k):
                    setattr(self.benchmark, k, v)
        return self
    def validate(self) -> 'ConfigManager':
        if not os.path.exists(self.training.data_dir):
            print(f"[WARN] Data directory {self.training.data_dir} does not exist")
        return self
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
        }
def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--max_iters', type=int, default=None, help='Override max iterations')
    parser.add_argument('--batch_size', type=int, default=None, help='Override micro batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    return parser
def load_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    if args.output_dir:
        config_data.setdefault('training', {})['output_dir'] = args.output_dir
    if args.max_iters:
        config_data.setdefault('training', {})['max_iters'] = args.max_iters
    if args.batch_size:
        config_data.setdefault('training', {})['micro_batch_size'] = args.batch_size
    if args.learning_rate:
        config_data.setdefault('training', {})['learning_rate'] = args.learning_rate
    return config_data
