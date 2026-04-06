#!/usr/bin/env python3
"""
Prepare text data for PowerGPT training.

Tokenizes raw text files into memory-mappable .bin files.
Supports:
- Single file or directory of files
- Streaming large files without loading into RAM
- Train/val split
- Multiple datasets concatenation
- Progress bar with tqdm (optional)

Usage:
    python scripts/prepare_data.py --input /path/to/texts --output data/my_dataset/
"""

import os
import sys
import argparse
import glob
from typing import List, Optional, Iterator

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from powergpt.data_pipeline import TokenizerWrapper


def read_text_chunks(file_path: str, chunk_size_mb: int = 100) -> Iterator[str]:
    """
    Read text file in chunks to avoid loading entire file into memory.

    Args:
        file_path: Path to text file
        chunk_size_mb: Chunk size in megabytes

    Yields:
        Text chunks
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def tokenize_file(
    file_path: str,
    tokenizer: TokenizerWrapper,
    max_tokens: Optional[int] = None,
    chunk_size_mb: int = 100,
    verbose: bool = True
) -> np.ndarray:
    """
    Tokenize a single text file.

    Args:
        file_path: Path to text file
        tokenizer: Tokenizer instance
        max_tokens: Maximum number of tokens to extract (None = all)
        chunk_size_mb: Chunk size for reading
        verbose: Print progress

    Returns:
        numpy array of token IDs (uint16)
    """
    tokens = []
    total_chars = 0

    if verbose:
        print(f"Tokenizing: {file_path}")

    for chunk in read_text_chunks(file_path, chunk_size_mb):
        chunk_tokens = tokenizer.encode(chunk)
        tokens.extend(chunk_tokens)
        total_chars += len(chunk)

        if max_tokens and len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            break

        if verbose and len(tokens) % 1000000 < chunk_size_mb:
            print(f"  Tokens so far: {len(tokens):,}")

    token_array = np.array(tokens, dtype=np.uint16)

    if verbose:
        print(f"  Total tokens: {len(token_array):,}")
        print(f"  Total chars: {total_chars:,}")

    return token_array


def tokenize_directory(
    dir_path: str,
    tokenizer: TokenizerWrapper,
    extensions: List[str] = ['.txt', '.text', '.md'],
    max_tokens: Optional[int] = None,
    chunk_size_mb: int = 100,
    verbose: bool = True
) -> np.ndarray:
    """
    Tokenize all text files in a directory.

    Args:
        dir_path: Directory path
        tokenizer: Tokenizer instance
        extensions: File extensions to include
        max_tokens: Maximum total tokens
        chunk_size_mb: Chunk size for reading
        verbose: Print progress

    Returns:
        numpy array of concatenated token IDs
    """
    all_tokens = []

    # Find all matching files
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
        files.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))

    files = sorted(set(files))

    if verbose:
        print(f"Found {len(files)} files in {dir_path}")

    for file_path in files:
        if max_tokens and len(all_tokens) >= max_tokens:
            break

        remaining = max_tokens - len(all_tokens) if max_tokens else None
        file_tokens = tokenize_file(file_path, tokenizer, remaining, chunk_size_mb, verbose)
        all_tokens.extend(file_tokens)

    return np.array(all_tokens, dtype=np.uint16)


def save_tokens(tokens: np.ndarray, output_path: str, split: str = 'train'):
    """
    Save tokens to binary file.

    Args:
        tokens: numpy array of tokens
        output_path: Output directory or file path
        split: 'train' or 'val' (used for naming)
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if os.path.isdir(output_path) or output_path.endswith('/'):
        output_file = os.path.join(output_path, f'{split}.bin')
    else:
        output_file = output_path

    tokens.tofile(output_file)
    print(f"Saved {len(tokens):,} tokens to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def split_train_val(
    tokens: np.ndarray,
    train_ratio: float,
    output_dir: str,
    val_output_dir: Optional[str] = None
) -> tuple:
    """
    Split tokens into train and validation sets.

    Args:
        tokens: Full token array
        train_ratio: Ratio for training (0.0-1.0)
        output_dir: Output directory for train.bin
        val_output_dir: Output directory for val.bin (same as output_dir if None)

    Returns:
        (train_tokens, val_tokens)
    """
    split_idx = int(len(tokens) * train_ratio)

    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    save_tokens(train_tokens, output_dir, 'train')
    save_tokens(val_tokens, val_output_dir or output_dir, 'val')

    print(f"Train tokens: {len(train_tokens):,} ({len(train_tokens)/len(tokens)*100:.1f}%)")
    print(f"Val tokens: {len(val_tokens):,} ({len(val_tokens)/len(tokens)*100:.1f}%)")

    return train_tokens, val_tokens


def main():
    parser = argparse.ArgumentParser(description='Prepare data for PowerGPT training')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file or directory with text files')
    parser.add_argument('--output', type=str, default='data/',
                        help='Output directory for .bin files')
    parser.add_argument('--train_ratio', type=float, default=0.99,
                        help='Train/validation split ratio (default: 0.99)')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Maximum number of tokens to extract (for debugging)')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                        help='Tokenizer to use (gpt2, r50k_base, p50k_base, cl100k_base)')
    parser.add_argument('--chunk_size_mb', type=int, default=100,
                        help='Chunk size for reading files in MB')
    parser.add_argument('--no_split', action='store_true',
                        help='Do not split into train/val (output single file)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print progress')

    args = parser.parse_args()

    # Initialize tokenizer
    tokenizer = TokenizerWrapper(args.tokenizer)
    print(f"Using tokenizer: {args.tokenizer}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Process input
    if os.path.isfile(args.input):
        print(f"Processing single file: {args.input}")
        tokens = tokenize_file(
            args.input, tokenizer,
            max_tokens=args.max_tokens,
            chunk_size_mb=args.chunk_size_mb,
            verbose=args.verbose
        )

        if args.no_split:
            save_tokens(tokens, args.output, 'data')
        else:
            split_train_val(tokens, args.train_ratio, args.output)

    elif os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        tokens = tokenize_directory(
            args.input, tokenizer,
            max_tokens=args.max_tokens,
            chunk_size_mb=args.chunk_size_mb,
            verbose=args.verbose
        )

        if args.no_split:
            save_tokens(tokens, args.output, 'data')
        else:
            split_train_val(tokens, args.train_ratio, args.output)

    else:
        print(f"Error: {args.input} is not a valid file or directory")
        sys.exit(1)

    print("\nDone!")


if __name__ == '__main__':
    main()