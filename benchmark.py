import time
import json
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from .config import ModelArchConfig, BenchmarkConfig
from .model import GPT
from .utils import get_gpu_memory_map, format_time, is_main_process
@torch.no_grad()
def benchmark_throughput(
    model: nn.Module,
    batch_sizes: List[int],
    seq_lengths: List[int],
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 50,
    use_kv_cache: bool = False
) -> Dict[str, Dict[int, Dict[int, float]]]:
    model.eval()
    results = {}
    for batch_size in batch_sizes:
        results[batch_size] = {}
        for seq_len in seq_lengths:
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            for _ in range(num_warmup):
                _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            avg_time = sum(times) / len(times)
            tokens_per_sec = (batch_size * seq_len) / avg_time
            results[batch_size][seq_len] = tokens_per_sec
            if is_main_process():
                print(f"[BENCH] Throughput: batch={batch_size}, seq={seq_len}, tok/s={tokens_per_sec:.0f}")
    return results
@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    batch_sizes: List[int],
    seq_lengths: List[int],
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, Dict[int, Dict[int, float]]]:
    model.eval()
    results = {}
    for batch_size in batch_sizes:
        results[batch_size] = {}
        for seq_len in seq_lengths:
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            for _ in range(num_warmup):
                _ = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  
            avg_time = sum(times) / len(times)
            ms_per_token = avg_time / (batch_size * seq_len)
            results[batch_size][seq_len] = ms_per_token
            if is_main_process():
                print(f"[BENCH] Latency: batch={batch_size}, seq={seq_len}, ms/tok={ms_per_token:.3f}")
    return results
@torch.no_grad()
def benchmark_generation(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    use_kv_cache: bool = True,
    device: torch.device = None,
    num_runs: int = 5
) -> Dict[str, Any]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_tokens_generated = 0
    total_time = 0.0
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_kv_cache=use_kv_cache,
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
        generated = output_ids.shape[1] - input_tensor.shape[1]
        total_tokens_generated += generated
        total_time += elapsed
    avg_tokens_per_sec = total_tokens_generated / total_time if total_time > 0 else 0
    avg_ms_per_token = (total_time * 1000) / total_tokens_generated if total_tokens_generated > 0 else 0
    return {
        'total_tokens': total_tokens_generated,
        'total_time_ms': total_time * 1000,
        'tokens_per_second': avg_tokens_per_sec,
        'ms_per_token': avg_ms_per_token,
        'num_prompts': len(prompts),
        'avg_tokens_per_prompt': total_tokens_generated / len(prompts),
    }
def benchmark_memory(
    model: nn.Module,
    batch_sizes: List[int],
    seq_lengths: List[int],
    vocab_size: int,
    device: torch.device
) -> Dict[str, Dict[int, Dict[int, float]]]:
    if not torch.cuda.is_available():
        return {}
    model.eval()
    results = {}
    for batch_size in batch_sizes:
        results[batch_size] = {}
        for seq_len in seq_lengths:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            _ = model(x)
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            results[batch_size][seq_len] = peak_memory
            if is_main_process():
                print(f"[MEM] batch={batch_size}, seq={seq_len}, peak={peak_memory:.2f}GB")
    return results
def benchmark_memory_peak(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device
) -> float:
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9
def estimate_flops(
    config: ModelArchConfig,
    batch_size: int = 1,
    seq_len: int = 1024
) -> Dict[str, float]:
    n_layer = config.n_layer
    n_head = config.n_head
    n_embd = config.n_embd
    vocab_size = config.vocab_size
    block_size = seq_len
    attn_flops = 2 * batch_size * block_size * block_size * n_embd * n_layer
    mlp_flops = 2 * batch_size * block_size * (4 * n_embd * n_embd) * n_layer
    embed_flops = batch_size * block_size * n_embd
    head_flops = batch_size * block_size * n_embd * vocab_size
    total_flops = attn_flops + mlp_flops + embed_flops + head_flops
    total_gflops = total_flops / 1e9
    return {
        'total_gflops': total_gflops,
        'attention_gflops': attn_flops / 1e9,
        'mlp_gflops': mlp_flops / 1e9,
        'embedding_gflops': embed_flops / 1e9,
        'head_gflops': head_flops / 1e9,
    }
class Benchmarker:
    def __init__(
        self,
        model: nn.Module,
        config: BenchmarkConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        self.vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 50257
        self.results = {}
    def run_all(self) -> Dict[str, Any]:
        if is_main_process():
            print("\n" + "=" * 60)
            print("Running PowerGPT Benchmarks")
            print("=" * 60)
        self.results['throughput'] = benchmark_throughput(
            self.model,
            self.config.batch_sizes,
            self.config.seq_lengths,
            self.vocab_size,
            self.device,
            self.config.num_warmup,
            self.config.num_runs
        )
        self.results['latency'] = benchmark_latency(
            self.model,
            self.config.batch_sizes,
            self.config.seq_lengths,
            self.vocab_size,
            self.device,
            self.config.num_warmup,
            self.config.num_runs
        )
        if self.config.profile_memory and torch.cuda.is_available():
            self.results['memory'] = benchmark_memory(
                self.model,
                self.config.batch_sizes,
                self.config.seq_lengths,
                self.vocab_size,
                self.device
            )
        self.results['flops'] = estimate_flops(
            self.model.config,
            batch_size=self.config.batch_sizes[0],
            seq_len=self.config.seq_lengths[0]
        )
        return self.results
    def save_results(self, output_path: str) -> None:
        if not is_main_process():
            return
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        serializable = convert(self.results)
        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[INFO] Benchmark results saved to {output_path}")
    def print_summary(self) -> None:
        if not is_main_process():
            return
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)
        if 'throughput' in self.results:
            print("\n--- Throughput (tokens/second) ---")
            for bs, seq_results in self.results['throughput'].items():
                for seq_len, tps in seq_results.items():
                    print(f"  batch={bs}, seq={seq_len}: {tps:.0f} tok/s")
        if 'latency' in self.results:
            print("\n--- Latency (ms/token) ---")
            for bs, seq_results in self.results['latency'].items():
                for seq_len, ms in seq_results.items():
                    print(f"  batch={bs}, seq={seq_len}: {ms:.3f} ms/tok")
        if 'memory' in self.results:
            print("\n--- Peak Memory (GB) ---")
            for bs, seq_results in self.results['memory'].items():
                for seq_len, mem in seq_results.items():
                    print(f"  batch={bs}, seq={seq_len}: {mem:.2f} GB")
        if 'flops' in self.results:
            print("\n--- FLOPs Estimation ---")
            print(f"  Total: {self.results['flops']['total_gflops']:.1f} GFLOPs")
            print(f"  Attention: {self.results['flops']['attention_gflops']:.1f} GFLOPs")
            print(f"  MLP: {self.results['flops']['mlp_gflops']:.1f} GFLOPs")
        print("=" * 60 + "\n")
def main():
    import argparse
    parser = argparse.ArgumentParser(description='PowerGPT Benchmark')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch_sizes', type=str, default='1,2,4,8',
                        help='Comma-separated batch sizes')
    parser.add_argument('--seq_lengths', type=str, default='128,256,512,1024',
                        help='Comma-separated sequence lengths')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output path for results')
    args = parser.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'config' in checkpoint and checkpoint['config']:
        from .config import ConfigManager
        config = ConfigManager().from_dict(checkpoint['config']).model
    else:
        config = ModelArchConfig()
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    bench_config = BenchmarkConfig(
        batch_sizes=tuple(batch_sizes),
        seq_lengths=tuple(seq_lengths),
    )
    benchmarker = Benchmarker(model, bench_config, device)
    results = benchmarker.run_all()
    benchmarker.print_summary()
    benchmarker.save_results(args.output)
if __name__ == '__main__':
    main()
