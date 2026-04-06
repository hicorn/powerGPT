#!/usr/bin/env python3
"""
Universal dataset preparer for PowerGPT.
Usage: python prepare.py --dataset openwebtext --max_tokens 5000000000
"""

import os
import argparse
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Hugging Face dataset name')
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=5_000_000_000, help='~5GB')
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.dataset}...")
    dataset = load_dataset(args.dataset, args.subset, split='train', streaming=True)

    enc = tiktoken.get_encoding('gpt2')
    all_tokens = []
    total_tokens = 0

    for example in tqdm(dataset):
        text = example.get('text', example.get('content', ''))
        if not text:
            continue
        tokens = enc.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(enc.eot_token)
        total_tokens += len(tokens) + 1
        if total_tokens >= args.max_tokens:
            all_tokens = all_tokens[:args.max_tokens]
            break

    # split train/val (99/1)
    split_idx = int(len(all_tokens) * 0.99)
    train_tokens = np.array(all_tokens[:split_idx], dtype=np.uint16)
    val_tokens = np.array(all_tokens[split_idx:], dtype=np.uint16)

    train_tokens.tofile(os.path.join(args.output_dir, 'train.bin'))
    val_tokens.tofile(os.path.join(args.output_dir, 'val.bin'))

    print(f"Saved {len(train_tokens):,} train tokens, {len(val_tokens):,} val tokens to {args.output_dir}")

if __name__ == '__main__':
    main()