"""
Command-line interface for PowerGPT.

Provides:
- Interactive chat mode
- Single generation mode
- Training launch
- Benchmarking
- Model export
- Configuration management
"""

import os
import sys
import argparse
from typing import Optional

import torch
from tiktoken import get_encoding

# Исправленные импорты — теперь из пакета powergpt
from powergpt.config import (
    ConfigManager, ModelArchConfig, TrainingConfig,
    InferenceConfig, BenchmarkConfig, add_config_args, load_config_from_args
)
from powergpt.model import GPT
from powergpt.trainer import Trainer
from powergpt.export import Exporter
from powergpt.benchmark import Benchmarker
from powergpt.utils import set_seed, get_device_info, is_main_process, print_model_summary


# -----------------------------------------------------------------------------
# Color codes for terminal output
# -----------------------------------------------------------------------------

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'


def color_text(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


# -----------------------------------------------------------------------------
# CLI Commands
# -----------------------------------------------------------------------------

class PowerGPTCLI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = None
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_config(self) -> 'PowerGPTCLI':
        self.config = ConfigManager()
        if hasattr(self.args, 'config') and self.args.config:
            self.config.load_yaml(self.args.config)
        if hasattr(self.args, 'output_dir') and self.args.output_dir:
            self.config.training.output_dir = self.args.output_dir
        if hasattr(self.args, 'max_iters') and self.args.max_iters:
            self.config.training.max_iters = self.args.max_iters
        self.config.validate()
        return self
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> 'PowerGPTCLI':
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'config' in checkpoint:
                self.config = ConfigManager().from_dict(checkpoint['config'])
            self.model = GPT(self.config.model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if is_main_process():
                print(f"[INFO] Loaded model from {checkpoint_path}")
        else:
            self.model = GPT(self.config.model)
            if is_main_process():
                print(f"[INFO] Created new model with {self.config.model.n_layer} layers")
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = get_encoding('gpt2')
        return self
    
    def train(self) -> None:
        if not is_main_process():
            return
        print(color_text("\n" + "=" * 60, Colors.CYAN))
        print(color_text("PowerGPT Training", Colors.BOLD))
        print(color_text("=" * 60 + "\n", Colors.CYAN))
        local_rank = self.args.local_rank if hasattr(self.args, 'local_rank') else -1
        trainer = Trainer(self.config, local_rank=local_rank)
        resume_from = self.args.resume if hasattr(self.args, 'resume') else None
        trainer.train(resume_from=resume_from)
    
    def generate(self, prompt: str, interactive: bool = False) -> None:
        if self.model is None:
            self.load_model(self.args.checkpoint if hasattr(self.args, 'checkpoint') else None)
        if interactive:
            self._interactive_mode()
        else:
            self._single_generation(prompt)
    
    def _single_generation(self, prompt: str) -> None:
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=self.config.inference.max_new_tokens,
            temperature=self.config.inference.temperature,
            top_k=self.config.inference.top_k,
            top_p=self.config.inference.top_p,
            repetition_penalty=self.config.inference.repetition_penalty,
            use_kv_cache=self.config.inference.use_kv_cache,
            max_kv_cache_tokens=self.config.inference.max_kv_cache_tokens,
        )
        full_text = self.tokenizer.decode(output_ids[0].tolist())
        print(color_text("\n" + "=" * 60, Colors.GREEN))
        print(color_text(f"Prompt: {prompt}", Colors.BOLD))
        print(color_text("=" * 60, Colors.GREEN))
        print(full_text)
        print(color_text("=" * 60 + "\n", Colors.GREEN))
    
    def _interactive_mode(self) -> None:
        print(color_text("\n" + "=" * 60, Colors.CYAN))
        print(color_text("PowerGPT Interactive Mode", Colors.BOLD))
        print(color_text("Type 'exit' to quit, 'clear' to clear history, 'reset' to reset model", Colors.DIM))
        print(color_text("=" * 60 + "\n", Colors.CYAN))
        history = []
        while True:
            try:
                user_input = input(color_text("You: ", Colors.BLUE)).strip()
                if user_input.lower() == 'exit':
                    print(color_text("Goodbye!", Colors.GREEN))
                    break
                elif user_input.lower() == 'clear':
                    history = []
                    print(color_text("[INFO] History cleared", Colors.YELLOW))
                    continue
                elif user_input.lower() == 'reset':
                    self.model.reset_kv_cache()
                    print(color_text("[INFO] KV cache reset", Colors.YELLOW))
                    continue
                elif not user_input:
                    continue
                if history:
                    context = "\n".join(history[-10:]) + f"\nUser: {user_input}\nAssistant: "
                else:
                    context = f"User: {user_input}\nAssistant: "
                input_ids = self.tokenizer.encode(context)
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                output_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=self.config.inference.max_new_tokens,
                    temperature=self.config.inference.temperature,
                    top_k=self.config.inference.top_k,
                    top_p=self.config.inference.top_p,
                    repetition_penalty=self.config.inference.repetition_penalty,
                    use_kv_cache=self.config.inference.use_kv_cache,
                )
                response = self.tokenizer.decode(output_ids[0].tolist())
                if response.startswith(context):
                    response = response[len(context):]
                print(color_text(f"Assistant: {response}", Colors.GREEN))
                history.append(f"User: {user_input}")
                history.append(f"Assistant: {response}")
            except KeyboardInterrupt:
                print(color_text("\n[INFO] Interrupted. Type 'exit' to quit.", Colors.YELLOW))
            except Exception as e:
                print(color_text(f"\n[ERROR] {e}", Colors.RED))
    
    def benchmark(self) -> None:
        if self.model is None:
            self.load_model(self.args.checkpoint if hasattr(self.args, 'checkpoint') else None)
        bench_config = BenchmarkConfig(
            seq_lengths=tuple(self.args.seq_lengths) if hasattr(self.args, 'seq_lengths') else (128, 256, 512, 1024),
            batch_sizes=tuple(self.args.batch_sizes) if hasattr(self.args, 'batch_sizes') else (1, 2, 4, 8),
            num_warmup=self.args.num_warmup if hasattr(self.args, 'num_warmup') else 10,
            num_runs=self.args.num_runs if hasattr(self.args, 'num_runs') else 50,
            profile_memory=self.args.profile_memory if hasattr(self.args, 'profile_memory') else True,
        )
        benchmarker = Benchmarker(self.model, bench_config, self.device)
        results = benchmarker.run_all()
        benchmarker.print_summary()
        if hasattr(self.args, 'output') and self.args.output:
            benchmarker.save_results(self.args.output)
    
    def export(self) -> None:
        if self.model is None:
            self.load_model(self.args.checkpoint if hasattr(self.args, 'checkpoint') else None)
        exporter = Exporter(self.model, self.config.model)
        example_input = torch.randint(0, self.config.model.vocab_size, (1, 128), device=self.device)
        export_format = self.args.export_format if hasattr(self.args, 'export_format') else 'torchscript'
        export_path = self.args.export_path if hasattr(self.args, 'export_path') else 'exported_model'
        if export_format == 'torchscript':
            exporter.to_torchscript(example_input, f"{export_path}.pt")
        elif export_format == 'onnx':
            exporter.to_onnx(example_input, f"{export_path}.onnx", use_fp16=self.args.export_fp16 if hasattr(self.args, 'export_fp16') else False)
        elif export_format == 'tensorrt':
            exporter.to_tensorrt(example_input, f"{export_path}_trt.pt")
        elif export_format == 'all':
            exporter.export_all(example_input, export_path)
        else:
            print(color_text(f"[ERROR] Unknown format: {export_format}", Colors.RED))
    
    def info(self) -> None:
        print(color_text("\n" + "=" * 60, Colors.CYAN))
        print(color_text("PowerGPT Information", Colors.BOLD))
        print(color_text("=" * 60 + "\n", Colors.CYAN))
        print(color_text("System Information:", Colors.YELLOW))
        device_info = get_device_info()
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        if self.model:
            print(color_text("\nModel Information:", Colors.YELLOW))
            print_model_summary(self.model)
            print(color_text("Configuration:", Colors.YELLOW))
            print(f"  n_layer: {self.config.model.n_layer}")
            print(f"  n_head: {self.config.model.n_head}")
            print(f"  n_embd: {self.config.model.n_embd}")
            print(f"  block_size: {self.config.model.block_size}")
            print(f"  flash_attention: {self.config.model.flash_attention}")
            print(f"  use_rope: {self.config.model.use_rope}")
            print(f"  use_moe: {self.config.model.use_moe}")
        print()


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='PowerGPT - Optimized GPT Training and Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, default='configs/t4_124m.yaml', help='Config file')
    train_parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    train_parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    add_config_args(train_parser)
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text from prompt')
    gen_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    gen_parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    gen_parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens to generate')
    gen_parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    gen_parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
    gen_parser.add_argument('--top_p', type=float, default=0.9, help='Top-p sampling')
    add_config_args(gen_parser)
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    chat_parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens per response')
    chat_parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    add_config_args(chat_parser)
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    bench_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    bench_parser.add_argument('--batch_sizes', type=str, default='1,2,4,8', help='Comma-separated batch sizes')
    bench_parser.add_argument('--seq_lengths', type=str, default='128,256,512,1024', help='Comma-separated seq lengths')
    bench_parser.add_argument('--num_warmup', type=int, default=10, help='Warmup runs')
    bench_parser.add_argument('--num_runs', type=int, default=50, help='Measured runs')
    bench_parser.add_argument('--profile_memory', action='store_true', help='Profile memory usage')
    bench_parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file')
    add_config_args(bench_parser)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    export_parser.add_argument('--export_format', type=str, default='torchscript', choices=['torchscript', 'onnx', 'tensorrt', 'all'], help='Export format')
    export_parser.add_argument('--export_path', type=str, default='exported_model', help='Output path')
    export_parser.add_argument('--export_fp16', action='store_true', help='Export with FP16')
    add_config_args(export_parser)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model and system info')
    info_parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint (optional)')
    add_config_args(info_parser)
    
    return parser


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main():
    parser = create_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    set_seed(1337)
    cli = PowerGPTCLI(args)
    if args.command == 'train':
        cli.load_config()
        cli.train()
    elif args.command == 'generate':
        cli.load_config()
        cli.load_model(args.checkpoint)
        cli.generate(args.prompt)
    elif args.command == 'chat':
        cli.load_config()
        cli.load_model(args.checkpoint)
        cli.config.inference.max_new_tokens = args.max_tokens
        cli.config.inference.temperature = args.temperature
        cli.generate(None, interactive=True)
    elif args.command == 'benchmark':
        cli.load_config()
        cli.load_model(args.checkpoint)
        cli.benchmark()
    elif args.command == 'export':
        cli.load_config()
        cli.load_model(args.checkpoint)
        cli.export()
    elif args.command == 'info':
        cli.load_config()
        if hasattr(args, 'checkpoint') and args.checkpoint:
            cli.load_model(args.checkpoint)
        cli.info()
    else:
        print(color_text(f"[ERROR] Unknown command: {args.command}", Colors.RED))
        sys.exit(1)


if __name__ == '__main__':
    main()