#!/usr/bin/env python3
"""
Universal dataset preparer for PowerGPT.
Downloads any dataset from Hugging Face and tokenizes it.
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
    parser.add_argument('--subset', type=str, default=None, help='Dataset subset')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--max_tokens', type=int, default=None, help='Max tokens to extract')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {args.dataset}...")
    dataset = load_dataset(args.dataset, args.subset, split='train')
    
    enc = tiktoken.get_encoding('gpt2')
    all_tokens = []
    
    for example in tqdm(dataset):
        text = example.get('text', example.get('content', ''))
        if not text:
            continue
        tokens = enc.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(enc.eot_token)
        
        if args.max_tokens and len(all_tokens) >= args.max_tokens:
            all_tokens = all_tokens[:args.max_tokens]
            break
    
    # Split train/val (99%/1%)
    split_idx = int(len(all_tokens) * 0.99)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    
    train_tokens = np.array(train_tokens, dtype=np.uint16)
    val_tokens = np.array(val_tokens, dtype=np.uint16)
    
    train_tokens.tofile(os.path.join(args.output_dir, 'train.bin'))
    val_tokens.tofile(os.path.join(args.output_dir, 'val.bin'))
    
    print(f"Saved {len(train_tokens):,} train tokens")
    print(f"Saved {len(val_tokens):,} val tokens")

if __name__ == '__main__':
    main()