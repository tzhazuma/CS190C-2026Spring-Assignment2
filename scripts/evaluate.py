from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, default_data_collator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hw2.common import ensure_dir, format_metrics, load_json, load_yaml
from hw2.data import build_language_modeling_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS190C HW2 evaluation script")
    parser.add_argument("--experiment-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    return parser.parse_args()


def create_accelerator() -> "Accelerator":
    """
    Create the evaluation accelerator.
    """
    return Accelerator()


def build_eval_dataloader(exp_config: dict, tokenizer) -> DataLoader:
    """
    Build the validation dataloader used for offline evaluation.
    """
    datasets = build_language_modeling_splits(
        dataset_name=exp_config["dataset_name"],
        dataset_config_name=exp_config["dataset_config_name"],
        tokenizer=tokenizer,
        block_size=exp_config["block_size"],
        num_preprocessing_workers=exp_config["num_preprocessing_workers"],
        cache_dir=exp_config.get("hf_cache_dir"),
    )
    val_dataset = cast(Any, datasets["validation"])
    return DataLoader(
        val_dataset,
        batch_size=exp_config["per_device_eval_batch_size"],
        shuffle=False,
        collate_fn=default_data_collator,
    )


@torch.no_grad()
def evaluate(accelerator, model, dataloader) -> dict[str, float]:
    """
    Evaluate the model on the validation split and return loss and perplexity.
    """
    model.eval()
    losses = []

    for batch in dataloader:
        outputs = model(**batch)
        gathered_loss = accelerator.gather_for_metrics(outputs.loss.detach().repeat(batch["input_ids"].size(0)))
        losses.append(gathered_loss)

    loss_tensor = torch.cat(losses)
    val_loss = loss_tensor.mean().item()
    return {
        "val_loss": val_loss,
        "val_perplexity": math.exp(val_loss),
    }


def main() -> None:
    args = parse_args()
    exp_config = load_yaml(args.experiment_config)
    model_config_dict = load_json(args.model_config)

    ensure_dir(Path(exp_config["output_dir"]) / "eval")

    accelerator = create_accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
        exp_config["tokenizer_name_or_path"],
        cache_dir=exp_config.get("hf_cache_dir"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config_dict["vocab_size"] = len(tokenizer)
    if tokenizer.bos_token_id is not None:
        model_config_dict["bos_token_id"] = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        model_config_dict["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        model_config_dict["pad_token_id"] = tokenizer.pad_token_id

    model_config = LlamaConfig(**model_config_dict)
    model = LlamaForCausalLM(model_config)

    eval_dataloader = build_eval_dataloader(exp_config, tokenizer)

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    checkpoint_path = Path(args.checkpoint_path)
    state_dict_path = checkpoint_path if checkpoint_path.is_file() else checkpoint_path / "model_state.pt"
    if not state_dict_path.is_file():
        raise FileNotFoundError(f"Expected model weights at {state_dict_path}.")

    state_dict = torch.load(state_dict_path, map_location="cpu")
    accelerator.unwrap_model(model).load_state_dict(state_dict)

    metrics = evaluate(accelerator, model, eval_dataloader)

    if accelerator.is_main_process:
        print(format_metrics(metrics))


if __name__ == "__main__":
    main()
