⚠️ CRITICAL BUG FIXED (Apr 28, 2026)

If you cloned the repository before April 28, 2026 — `git pull` immediately.

 What was broken:
`trainer.py` referenced `training.gradient_checkpointing`, but this field lives in `model.gradient_checkpointing`. Training would crash with `AttributeError` on the first iteration.

 What is fixed:
- Single-line fix in `trainer.py` (line ~95): `self.config.training.gradient_checkpointing` → `self.config.model.gradient_checkpointing`
- All comments, docstrings, and dead code stripped across the entire codebase
- LoRA now applies **before** FSDP/DDP wrapping (correct order)
- Clean `git history` without junk

 How to update:
```bash
cd powerGPT
git pull origin main
pip install -e . --force-reinstall

&#x20;PowerGPT



Modern GPT implementation from scratch. Not a fork. Not a toy.



Runs faster than nanoGPT and does more.





🚀 Features



| Feature              | What it gives you 

|----------------------------------------------------------------|

| FlashAttention-2     | Doesn't choke on long contexts          |

| RoPE                 | Instead of absolute position embeddings |

| RMSNorm + SwiGLU     | Because they're just better             |

| MoE                  | More parameters, more problems          |

| LoRA                 | Fine-tune on cheap hardware             |

| KV cache             | Generation doesn't suck                 |

| ONNX / TensorRT export| Run anywhere                           |

| FSDP + checkpointing | Big models without OOM                  |

| CLI                  | train, generate, chat, benchmark        |







&#x20;⚡ Quick Start



```bash

git clone https://github.com/hicorn/powerGPT

cd powergpt

pip install -e .

```



Train on Shakespeare (sanity check)



```bash

python data/shakespeare\_char/prepare.py

python powergpt train --config configs/tiny\_shakespeare.yaml

```



Generate text



```bash

python powergpt generate --checkpoint checkpoints/best.pt --prompt "To be or not to be"

```



Interactive chat



```bash

python powergpt chat --checkpoint checkpoints/best.pt

```



Benchmark



```bash

python powergpt benchmark --checkpoint checkpoints/best.pt --batch\_sizes 1,2,4,8

```



\*\*Export to ONNX / TensorRT\*\*



```bash

python powergpt export --checkpoint checkpoints/best.pt --export\_format onnx

```



\---



&#x20;📦 Dependencies



```bash

pip install torch numpy tiktoken wandb tensorboard pyyaml

```



\*\*Optional:\*\*

\- `flash-attn` — FlashAttention-2 (strongly recommended)

\- `onnx` / `onnxruntime` — ONNX export

\- `torch2trt` — TensorRT export

\- `peft` — LoRA fine-tuning



\---



&#x20;🔥 Training on OpenWebText (GPT-2 scale)



```bash

python data/openwebtext/prepare.py

python powergpt train --config configs/gpt2\_124m.yaml

```



With DDP on 8 GPUs:\*\*



```bash

torchrun --standalone --nproc\_per\_node=8 powergpt train --config configs/gpt2\_124m.yaml

```



On 8×A100 — \~4 days to reach loss \~2.85.





&#x20;🎯 Fine-tuning



```bash

python data/shakespeare/prepare.py

python powergpt train --config configs/finetune\_shakespeare.yaml --init\_from gpt2

```







&#x20;⚙️ Performance



\- FlashAttention-2 + torch.compile — \~2x faster than nanoGPT on A100

\- KV cache — \~10x faster generation for long sequences





❗ Troubleshooting



| Issue | Solution |

|-------|-------------------------------------------------------------------------------------|

| CUDA OOM               |Reduce `batch\_size` or `block\_size`, enable gradient checkpointing  |

| FlashAttention-2 fails | Falls back to manual attention (slower but works)                  |







&#x20;🙏 Acknowledgements



\- \[nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy

\- \[FlashAttention](https://github.com/Dao-AILab/flash-attention) by Tri Dao

\- PyTorch team





