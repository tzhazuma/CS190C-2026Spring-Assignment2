from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TinyStories assets for offline use")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config-name", type=str, default=None)
    parser.add_argument("--tokenizer-name-or-path", type=str, default="roneneldan/TinyStories-33M")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    tokenizer_dir = output_dir / "tokenizer" / "TinyStories-33M"
    dataset_dir = output_dir / "datasets" / "TinyStories"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer to {tokenizer_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.hf_cache_dir,
    )
    tokenizer.save_pretrained(tokenizer_dir)

    print(f"Downloading dataset to {dataset_dir}...", flush=True)
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.hf_cache_dir,
    )
    dataset.save_to_disk(dataset_dir)

    print("Offline assets are ready.", flush=True)
    print(f"Tokenizer path: {tokenizer_dir}", flush=True)
    print(f"Dataset path: {dataset_dir}", flush=True)


if __name__ == "__main__":
    main()