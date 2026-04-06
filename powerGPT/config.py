"""
GPT model from scratch with FlashAttention-2, Rotary Positional Embeddings,
Mixture of Experts, KV cache with memory limits, and gradient checkpointing.
Optimized for T4 (16GB VRAM) and production use.

Features:
- FlashAttention-2 (with fallback to manual attention)
- Rotary Positional Embeddings (RoPE) with FP32 stability
- RMSNorm or LayerNorm (configurable)
- SwiGLU or GELU activation
- Mixture of Experts (MoE) with load balancing loss
- KV cache with automatic length limiting (prevents OOM)
- Gradient checkpointing for memory efficiency
- torch.compile support (optional)
- Weight tying (embedding + lm_head)
- Proper initialization for deep models
"""

import math
from typing import Optional, Tuple, List, Any, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# FlashAttention import with graceful fallback
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("[INFO] FlashAttention not installed. Using manual attention (slower).")

from .config import ModelArchConfig


# -----------------------------------------------------------------------------
# Core building blocks: RMSNorm, LayerNorm, RoPE, Activations
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class LayerNorm(nn.Module):
    """Standard LayerNorm with bias option."""
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key.
    Supports both (B, T, nh, hd) and (B, nh, T, hd) layouts.
    """
    if q.dim() == 4 and q.shape[2] == cos.shape[2]:  # (B, nh, T, hd) case
        cos = cos.transpose(1, 2)  # (1, T, 1, hd) -> (1, 1, T, hd)
        sin = sin.transpose(1, 2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings with FP32 computation for numerical stability.
    Precomputes cos/sin tables for all possible lengths up to max_seq_len.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Compute inverse frequencies in FP32 for accuracy
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Precompute cos/sin tables for given sequence length (in FP32 then cast)."""
        if seq_len <= self._seq_len_cached:
            return
        # Compute in FP32 for numerical stability
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (seq_len, dim)
        cos = emb.cos()  # (seq_len, dim)
        sin = emb.sin()
        # Reshape to (1, seq_len, 1, dim) for broadcasting
        self._cos_cached = cos[None, :, None, :].to(dtype)
        self._sin_cached = sin[None, :, None, :].to(dtype)
        self._seq_len_cached = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: (B, T, nh, hd) or (B, nh, T, hd)
            seq_len: actual sequence length
        Returns:
            cos, sin: (1, T, 1, hd) or (1, 1, T, hd) matching input layout
        """
        self._update_cache(seq_len, q.device, q.dtype)
        # If q is (B, nh, T, hd), transpose cos/sin accordingly
        if q.dim() == 4 and q.shape[2] == seq_len:  # (B, nh, T, hd)
            cos = self._cos_cached.transpose(1, 2)  # (1, 1, T, hd)
            sin = self._sin_cached.transpose(1, 2)
        else:  # (B, T, nh, hd)
            cos = self._cos_cached
            sin = self._sin_cached
        return cos, sin


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(xW1) * (xW2) then projection."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GELU(nn.Module):
    """GELU activation (legacy, slower than SwiGLU)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


# -----------------------------------------------------------------------------
# Mixture of Experts (MoE) layer with load balancing
# -----------------------------------------------------------------------------

class MoERouter(nn.Module):
    """Router for Mixture of Experts."""
    def __init__(self, dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.router_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B*T, dim)
        Returns:
            weights: (B*T, top_k) softmaxed
            indices: (B*T, top_k)
        """
        logits = self.gate(x) * self.router_scale
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return weights, indices


class MoELayer(nn.Module):
    """Mixture of Experts layer with auxiliary load balancing loss."""
    def __init__(self, dim: int, num_experts: int, top_k: int,
                 hidden_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = MoERouter(dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            SwiGLU(dim, hidden_dim, bias) for _ in range(num_experts)
        ])
        # Statistics for load balancing
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.register_buffer("expert_importance", torch.zeros(num_experts))
        self.load_balance_loss = torch.tensor(0.0, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            y: (B, T, C)
        """
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        router_weights, expert_indices = self.router(x_flat)  # (BT, top_k), (BT, top_k)

        # Update statistics for auxiliary loss
        self.expert_counts.zero_()
        self.expert_counts.scatter_add_(
            0, expert_indices.view(-1),
            torch.ones_like(expert_indices.view(-1), dtype=torch.float)
        )

        # Dispatch and compute expert outputs
        y_flat = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            expert_output = self.experts[expert_idx](expert_input)
            # Multiply by router weight
            weight_mask = (expert_indices == expert_idx).float()
            weight = (router_weights * weight_mask).sum(dim=-1, keepdim=True)[mask]
            y_flat[mask] += expert_output * weight

        # Auxiliary load balancing loss (encourages equal usage)
        fraction_per_expert = self.expert_counts / (B * T * self.top_k)
        router_probs = torch.softmax(self.router.gate.weight, dim=-1).mean(0)
        self.load_balance_loss = torch.sum(fraction_per_expert * router_probs) * self.num_experts

        return y_flat.view(B, T, C)


# -----------------------------------------------------------------------------
# FlashAttention with optional KV cache and memory limit
# -----------------------------------------------------------------------------

class FlashAttention(nn.Module):
    """Multi-head self-attention using FlashAttention-2 or manual implementation."""
    def __init__(self, config: ModelArchConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.flash = config.flash_attention and FLASH_ATTENTION_AVAILABLE
        self.use_rope = config.use_rope

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.block_size,
                                        base=config.rope_theta)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, T, C)
            past_kv: (k, v) from previous steps
            use_cache: whether to return updated kv cache
        Returns:
            out: (B, T, C)
            new_kv: (k, v) if use_cache else None
        """
        B, T, C = x.shape

        # Project and reshape to (B, T, nh, hd)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)

        # Apply RoPE
        if self.rope:
            cos, sin = self.rope(q, k, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_kv = (k, v) if use_cache else None

        # FlashAttention expects (B, T, nh, hd)
        if self.flash:
            # FlashAttention-2 (causal, dropout only during training)
            attn_out = flash_attn_func(q, k, v,
                                       dropout_p=self.dropout if self.training else 0.0,
                                       causal=True)
        else:
            # Manual attention (fallback)
            # Reshape to (B, nh, T, hd) for manual attention
            q = q.transpose(1, 2)  # (B, nh, T, hd)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            # Causal mask
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout if self.training else 0.0)
            attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)

        # Output projection
        out = self.out_proj(attn_out)
        return out, new_kv


# -----------------------------------------------------------------------------
# Transformer Block (supports both regular MLP and MoE)
# -----------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization."""
    def __init__(self, config: ModelArchConfig, layer_idx: int):
        super().__init__()
        # Normalization
        if config.use_rms_norm:
            self.norm1 = RMSNorm(config.n_embd)
            self.norm2 = RMSNorm(config.n_embd)
        else:
            self.norm1 = LayerNorm(config.n_embd, bias=config.bias)
            self.norm2 = LayerNorm(config.n_embd, bias=config.bias)

        # Attention
        self.attn = FlashAttention(config)

        # MLP or MoE
        if config.use_moe:
            self.mlp = MoELayer(config.n_embd, config.num_experts,
                                config.top_k_experts, bias=config.bias)
        else:
            if config.activation == 'swiglu':
                self.mlp = SwiGLU(config.n_embd, bias=config.bias)
            else:
                self.mlp = nn.Sequential(
                    nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                    GELU(),
                    nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_idx = layer_idx
        self.use_moe = config.use_moe

    def forward(self, x: torch.Tensor,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention with residual
        attn_out, new_kv = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + self.dropout(attn_out)

        # MLP with residual
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)
        return x, new_kv


# -----------------------------------------------------------------------------
# Full GPT Model
# -----------------------------------------------------------------------------

class GPT(nn.Module):
    """Complete GPT language model with all modern features."""
    def __init__(self, config: ModelArchConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings (only if not using RoPE)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layer)
        ])

        # Final normalization
        if config.use_rms_norm:
            self.norm = RMSNorm(config.n_embd)
        else:
            self.norm = LayerNorm(config.n_embd, bias=config.bias)

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (share embedding and lm_head weights)
        self.token_embedding.weight = self.lm_head.weight

        # Gradient checkpointing flag
        self._gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)

        # Special scaled init for residual projections (helps deep models)
        for name, param in self.named_parameters():
            if 'c_proj.weight' in name or 'w3.weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        """Initialize weights with GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            idx: (B, T) input token indices
            targets: (B, T) target token indices for loss (optional)
            past_kv: list of (k, v) caches from previous steps (one per layer)
            use_cache: return updated kv cache
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
            new_past_kv: list of (k, v) if use_cache else None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Token embeddings
        x = self.token_embedding(idx)  # (B, T, C)

        # Position embeddings (if not using RoPE)
        if not self.config.use_rope:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
            x = x + self.position_embedding(pos)

        x = self.drop(x)

        # Forward through blocks
        new_past_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            kv_i = past_kv[i] if past_kv is not None else None
            if self._gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                x, kv_i = torch.utils.checkpoint.checkpoint(
                    block, x, kv_i, use_cache, use_reentrant=False
                )
            else:
                x, kv_i = block(x, past_kv=kv_i, use_cache=use_cache)
            if use_cache:
                new_past_kv.append(kv_i)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1,
                                   label_smoothing=0.0)
            # Add MoE auxiliary loss if any
            if self.config.use_moe:
                moe_loss = sum(block.mlp.load_balance_loss for block in self.blocks if hasattr(block.mlp, 'load_balance_loss'))
                loss = loss + 0.01 * moe_loss

        return logits, loss, new_past_kv if use_cache else None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                 repetition_penalty: float = 1.0, use_kv_cache: bool = True,
                 max_kv_cache_tokens: int = 2048) -> torch.Tensor:
        """
        Autoregressive generation with KV cache and memory limit.
        Args:
            idx: (B, T) initial token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (>0)
            top_k: keep only top_k tokens
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeated tokens
            use_kv_cache: enable KV cache for speed
            max_kv_cache_tokens: maximum length of KV cache (prevents OOM)
        Returns:
            idx: (B, T+max_new_tokens)
        """
        self.eval()
        past_kv = [] if use_kv_cache else None
        cache_length = 0

        for _ in range(max_new_tokens):
            # Crop input to block size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass
            logits, _, past_kv_out = self.forward(idx_cond, past_kv=past_kv, use_cache=use_kv_cache)

            # Update KV cache with length limiting
            if use_kv_cache:
                past_kv = past_kv_out
                cache_length = idx_cond.size(1)
                if cache_length > max_kv_cache_tokens:
                    # Trim cache to half to free memory
                    past_kv = [(k[:, -max_kv_cache_tokens//2:], v[:, -max_kv_cache_tokens//2:]) for k, v in past_kv]
                    cache_length = max_kv_cache_tokens // 2

            # Get logits of last token
            logits = logits[:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(idx.shape[0]):
                    for token in set(idx[i].tolist()):
                        logits[i, token] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory during training."""
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False

    def reset_kv_cache(self):
        """Reset any internal KV cache (useful for long-running inference)."""
        pass  # No persistent cache inside model; handled in generate()

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: Tuple[float, float], optimizer_type: str = 'adamw'):
        """
        Configure optimizer with separate weight decay for bias/LayerNorm.
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) >= 2 and 'weight' in name and 'norm' not in name.lower():
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        if optimizer_type == 'adamw':
            if hasattr(torch.optim, 'AdamW') and hasattr(torch.optim.AdamW, 'fused'):
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                                              betas=betas, fused=True)
            else:
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        elif optimizer_type == 'lamb':
            try:
                from torch_optimizer import Lamb
                optimizer = Lamb(optim_groups, lr=learning_rate, betas=betas,
                                 weight_decay=weight_decay)
            except ImportError:
                print("[WARN] torch-optimizer not installed, falling back to AdamW")
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        elif optimizer_type == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(optim_groups, lr=learning_rate, betas=betas,
                                 weight_decay=weight_decay)
            except ImportError:
                print("[WARN] lion-pytorch not installed, falling back to AdamW")
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return optimizer