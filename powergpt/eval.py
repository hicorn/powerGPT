import math
import time
from typing import Optional, List, Dict, Any, Tuple, Callable
import torch
import torch.nn as nn
import torch.distributed as dist
from .utils import (
    get_rank, get_world_size, is_main_process, all_reduce,
    compute_perplexity, compute_accuracy, log_metrics, Timer
)
@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dataloader,
    ctx,
    device: torch.device,
    eval_iters: int = 200,
    distributed: bool = False
) -> float:
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
@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader,
    ctx,
    device: torch.device,
    eval_iters: int = 200,
    distributed: bool = False
) -> float:
    loss = evaluate_loss(model, dataloader, ctx, device, eval_iters, distributed)
    return compute_perplexity(loss)
@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader,
    ctx,
    device: torch.device,
    eval_iters: int = 200,
    distributed: bool = False
) -> float:
    model.eval()
    correct = 0
    total = 0
    for i, (x, y) in enumerate(dataloader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            logits, _, _ = model(x, y)
        predictions = logits.argmax(dim=-1)
        correct += (predictions == y).sum().item()
        total += y.numel()
    if total == 0:
        return 0.0
    accuracy = correct / total
    if distributed:
        acc_tensor = torch.tensor(accuracy, device=device)
        all_reduce(acc_tensor, op='avg')
        accuracy = acc_tensor.item()
    model.train()
    return accuracy
@torch.no_grad()
def evaluate_all_metrics(
    model: nn.Module,
    dataloader,
    ctx,
    device: torch.device,
    eval_iters: int = 200,
    distributed: bool = False
) -> Dict[str, float]:
    loss = evaluate_loss(model, dataloader, ctx, device, eval_iters, distributed)
    perplexity = compute_perplexity(loss)
    accuracy = evaluate_accuracy(model, dataloader, ctx, device, eval_iters, distributed)
    return {
        'loss': loss,
        'perplexity': perplexity,
        'accuracy': accuracy,
    }
@torch.no_grad()
def generate_samples(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    use_kv_cache: bool = True,
    device: torch.device = None
) -> List[str]:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    generated_texts = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_kv_cache=use_kv_cache,
        )
        full_text = tokenizer.decode(output_ids[0].tolist())
        if full_text.startswith(prompt):
            completion = full_text[len(prompt):]
        else:
            completion = full_text
        generated_texts.append(completion)
    model.train()
    return generated_texts
def evaluate_generation_quality(
    model: nn.Module,
    tokenizer,
    test_prompts: List[Tuple[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: torch.device = None
) -> Dict[str, float]:
    if device is None:
        device = next(model.parameters()).device
    exact_matches = 0
    partial_matches = 0
    total_length = 0
    for prompt, expected in test_prompts:
        generated = generate_samples(
            model, tokenizer, [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device
        )[0]
        total_length += len(generated)
        if generated.strip() == expected.strip():
            exact_matches += 1
        elif expected.strip() in generated.strip():
            partial_matches += 1
    n = len(test_prompts)
    return {
        'exact_match_rate': exact_matches / n if n > 0 else 0,
        'partial_match_rate': partial_matches / n if n > 0 else 0,
        'avg_generation_length': total_length / n if n > 0 else 0,
    }
@torch.no_grad()
def benchmark_forward_pass(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, float]:
    model.eval()
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
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    throughput = (batch_size * seq_len) / (avg_time / 1000)  
    model.train()
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'throughput_tok_per_sec': throughput,
    }
@torch.no_grad()
def benchmark_forward_backward(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 10,
    num_runs: int = 50
) -> Dict[str, float]:
    model.train()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(num_warmup):
        logits, loss, _ = model(x, y)
        loss.backward()
        model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        logits, loss, _ = model(x, y)
        loss.backward()
        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    avg_time = sum(times) / len(times)
    throughput = (batch_size * seq_len) / (avg_time / 1000)
    model.eval()
    return {
        'avg_time_ms': avg_time,
        'throughput_tok_per_sec': throughput,
    }
class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        ctx,
        distributed: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.ctx = ctx
        self.distributed = distributed
        self.rank = get_rank()
        self.timer = Timer()
    def evaluate_loss(
        self,
        dataloader,
        eval_iters: int = 200
    ) -> float:
        return evaluate_loss(
            self.model, dataloader, self.ctx, self.device,
            eval_iters, self.distributed
        )
    def evaluate_perplexity(
        self,
        dataloader,
        eval_iters: int = 200
    ) -> float:
        return evaluate_perplexity(
            self.model, dataloader, self.ctx, self.device,
            eval_iters, self.distributed
        )
    def evaluate_accuracy(
        self,
        dataloader,
        eval_iters: int = 200
    ) -> float:
        return evaluate_accuracy(
            self.model, dataloader, self.ctx, self.device,
            eval_iters, self.distributed
        )
    def evaluate_all(
        self,
        dataloader,
        eval_iters: int = 200
    ) -> Dict[str, float]:
        return evaluate_all_metrics(
            self.model, dataloader, self.ctx, self.device,
            eval_iters, self.distributed
        )
    def generate_samples(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        use_kv_cache: bool = True
    ) -> List[str]:
        return generate_samples(
            self.model, self.tokenizer, prompts,
            max_new_tokens, temperature, top_k, top_p,
            repetition_penalty, use_kv_cache, self.device
        )
    def benchmark_forward(
        self,
        batch_size: int,
        seq_len: int,
        num_warmup: int = 10,
        num_runs: int = 50
    ) -> Dict[str, float]:
        return benchmark_forward_pass(
            self.model, batch_size, seq_len,
            self.model.config.vocab_size, self.device,
            num_warmup, num_runs
        )
    def benchmark_forward_backward(
        self,
        batch_size: int,
        seq_len: int,
        num_warmup: int = 10,
        num_runs: int = 50
    ) -> Dict[str, float]:
        return benchmark_forward_backward(
            self.model, batch_size, seq_len,
            self.model.config.vocab_size, self.device,
            num_warmup, num_runs
        )
    def run_full_evaluation(
        self,
        dataloader,
        test_prompts: Optional[List[Tuple[str, str]]] = None,
        eval_iters: int = 200,
        benchmark_batch_sizes: List[int] = [1, 2, 4, 8],
        seq_len: int = 512
    ) -> Dict[str, Any]:
        results = {}
        self.timer.reset()
        metrics = self.evaluate_all(dataloader, eval_iters)
        results['metrics'] = metrics
        results['eval_time_s'] = self.timer.elapsed()
        if is_main_process():
            print(f"\n[EVAL] Loss: {metrics['loss']:.4f}")
            print(f"[EVAL] Perplexity: {metrics['perplexity']:.2f}")
            print(f"[EVAL] Accuracy: {metrics['accuracy']:.4f}")
        if test_prompts and is_main_process():
            self.timer.reset()
            gen_metrics = evaluate_generation_quality(
                self.model, self.tokenizer, test_prompts,
                device=self.device
            )
            results['generation'] = gen_metrics
            results['generation_time_s'] = self.timer.elapsed()
            print(f"[EVAL] Exact match: {gen_metrics['exact_match_rate']:.2%}")
            print(f"[EVAL] Partial match: {gen_metrics['partial_match_rate']:.2%}")
        if is_main_process():
            results['benchmark'] = {}
            for bs in benchmark_batch_sizes:
                self.timer.reset()
                fwd = self.benchmark_forward(bs, seq_len)
                results['benchmark'][f'forward_bs{bs}'] = fwd
                print(f"[BENCH] Forward BS={bs}: {fwd['throughput_tok_per_sec']:.0f} tok/s")
                self.timer.reset()
                fwd_bwd = self.benchmark_forward_backward(bs, seq_len)
                results['benchmark'][f'forward_backward_bs{bs}'] = fwd_bwd
                print(f"[BENCH] Forward+Backward BS={bs}: {fwd_bwd['throughput_tok_per_sec']:.0f} tok/s")
        return results
