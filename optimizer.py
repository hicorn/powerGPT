import math
from typing import Optional, List, Dict, Any, Tuple, Callable
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, fused=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, fused=fused)
        super().__init__(params, defaults)
        if fused and not torch.cuda.is_available():
            raise RuntimeError("Fused AdamW requires CUDA")
        if fused:
            self._step_supports_amp_scaling = True
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            if group['fused']:
                for p in group['params']:
                    if p.grad is not None:
                        params_with_grad.append(p)
                        grads.append(p.grad)
                        state = self.state[p]
                        if len(state) == 0:
                            state['step'] = 0
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            if group['amsgrad']:
                                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        exp_avgs.append(state['exp_avg'])
                        exp_avg_sqs.append(state['exp_avg_sq'])
                        if group['amsgrad']:
                            max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                        state_steps.append(state['step'])
                torch._foreach_add_(
                    exp_avgs,
                    torch._foreach_mul(
                        torch._foreach_sub(grads, exp_avgs),
                        torch._foreach_mul(torch._foreach_pow(beta1, state_steps), 1 - beta1)
                    ),
                    1
                )
                for p, grad, exp_avg, exp_avg_sq, step in zip(params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps):
                    step += 1
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                    state['step'] = step
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError('AdamW does not support sparse gradients')
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    if group['amsgrad']:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                    state['step'] += 1
                    step = state['step']
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    if group['amsgrad']:
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    else:
                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss
class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
class LAMB(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = (exp_avg / bias_correction1) / denom
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)
                p_norm = torch.norm(p)
                update_norm = torch.norm(update)
                if p_norm > 0 and update_norm > 0:
                    trust_ratio = p_norm / update_norm
                else:
                    trust_ratio = 1.0
                p.add_(update, alpha=-lr * trust_ratio)
        return loss
class CosineWithWarmupScheduler:
    def __init__(self, optimizer: Optimizer, lr_max: float, lr_min: float,
                 warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    def step(self) -> float:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.lr_max * self.current_step / self.warmup_steps
        elif self.current_step >= self.total_steps:
            lr = self.lr_min
        else:
            decay_ratio = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.lr_min + coeff * (self.lr_max - self.lr_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    def get_last_lr(self) -> List[float]:
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
class CosineWithRestartsScheduler:
    def __init__(self, optimizer: Optimizer, lr_max: float, lr_min: float,
                 warmup_steps: int, restart_interval: int, total_steps: int):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.restart_interval = restart_interval
        self.total_steps = total_steps
        self.current_step = 0
    def step(self) -> float:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.lr_max * self.current_step / self.warmup_steps
        else:
            cycle_step = (self.current_step - self.warmup_steps) % self.restart_interval
            cycle_length = self.restart_interval
            if cycle_step == 0:
                cycle_step = cycle_length
            decay_ratio = cycle_step / cycle_length
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.lr_min + coeff * (self.lr_max - self.lr_min)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
class LinearWithWarmupScheduler:
    def __init__(self, optimizer: Optimizer, lr_max: float, lr_min: float,
                 warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
    def step(self) -> float:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.lr_max * self.current_step / self.warmup_steps
        elif self.current_step >= self.total_steps:
            lr = self.lr_min
        else:
            decay_ratio = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.lr_max - (self.lr_max - self.lr_min) * decay_ratio
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
class ConstantScheduler:
    def __init__(self, optimizer: Optimizer, lr: float):
        self.optimizer = optimizer
        self.lr = lr
    def step(self) -> float:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr
def create_scheduler(optimizer: Optimizer, scheduler_type: str,
                     lr_max: float, lr_min: float,
                     warmup_steps: int, total_steps: int,
                     restart_interval: int = 50000) -> Callable:
    if scheduler_type == 'cosine':
        return CosineWithWarmupScheduler(optimizer, lr_max, lr_min, warmup_steps, total_steps)
    elif scheduler_type == 'cosine_restarts':
        return CosineWithRestartsScheduler(optimizer, lr_max, lr_min, warmup_steps, restart_interval, total_steps)
    elif scheduler_type == 'linear':
        return LinearWithWarmupScheduler(optimizer, lr_max, lr_min, warmup_steps, total_steps)
    elif scheduler_type == 'constant':
        return ConstantScheduler(optimizer, lr_max)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
def get_lr_schedule(iter_num: int, training_config) -> float:
    warmup_iters = training_config.warmup_iters
    lr_decay_iters = training_config.lr_decay_iters
    learning_rate = training_config.learning_rate
    min_lr = training_config.min_lr
    if iter_num < warmup_iters:
        return learning_rate * (iter_num + 1) / (warmup_iters + 1)
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
def create_optimizer(model: torch.nn.Module, training_config) -> Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) >= 2 and 'weight' in name and 'norm' not in name.lower():
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': training_config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    opt_type = training_config.optimizer_type
    lr = training_config.learning_rate
    betas = (training_config.beta1, training_config.beta2)
    if opt_type == 'adamw':
        try:
            if torch.cuda.is_available():
                optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas,
                                              weight_decay=training_config.weight_decay,
                                              fused=True)
                print("[INFO] Using fused AdamW (CUDA)")
            else:
                raise RuntimeError("CUDA not available")
        except (RuntimeError, AttributeError):
            optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas,
                                          weight_decay=training_config.weight_decay)
            print("[INFO] Using standard AdamW")
    elif opt_type == 'lamb':
        try:
            from torch_optimizer import Lamb
            optimizer = Lamb(optim_groups, lr=lr, betas=betas,
                             weight_decay=training_config.weight_decay)
            print("[INFO] Using LAMB optimizer")
        except ImportError:
            print("[WARN] torch-optimizer not installed, falling back to AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas,
                                          weight_decay=training_config.weight_decay)
    elif opt_type == 'lion':
        optimizer = Lion(optim_groups, lr=lr, betas=betas,
                         weight_decay=training_config.weight_decay)
        print("[INFO] Using Lion optimizer")
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")
    return optimizer
def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
def clip_grad_value_(parameters, clip_value: float) -> None:
    torch.nn.utils.clip_grad_value_(parameters, clip_value)
def get_param_groups(model: torch.nn.Module, weight_decay: float) -> List[Dict]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) >= 2 and 'weight' in name and 'norm' not in name.lower():
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
def zero_grad(optimizer: Optimizer, set_to_none: bool = True) -> None:
    optimizer.zero_grad(set_to_none=set_to_none)
def get_current_lr(optimizer: Optimizer) -> float:
    return optimizer.param_groups[0]['lr']
def set_lr(optimizer: Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
