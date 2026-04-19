from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hw2.common import count_trainable_parameters, ensure_dir, format_metrics, load_json, load_yaml, set_seed
from hw2.data import build_language_modeling_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CS190C HW2 training script")
    parser.add_argument("--experiment-config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    return parser.parse_args()


def create_accelerator(exp_config: dict) -> "Accelerator":
    """
    Create the training accelerator with gradient accumulation and TensorBoard logging.
    """
    return Accelerator(
        gradient_accumulation_steps=exp_config["gradient_accumulation_steps"],
        log_with="tensorboard",
        project_dir=exp_config["output_dir"],
    )


def build_dataloaders(exp_config: dict, tokenizer) -> tuple[DataLoader, DataLoader]:
    """
    Build shuffled training and deterministic validation dataloaders.
    """
    datasets = build_language_modeling_splits(
        dataset_name=exp_config["dataset_name"],
        dataset_config_name=exp_config["dataset_config_name"],
        tokenizer=tokenizer,
        block_size=exp_config["block_size"],
        num_preprocessing_workers=exp_config["num_preprocessing_workers"],
        cache_dir=exp_config.get("hf_cache_dir"),
    )
    train_dataset = cast(Any, datasets["train"])
    val_dataset = cast(Any, datasets["validation"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=exp_config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=default_data_collator,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=exp_config["per_device_eval_batch_size"],
        shuffle=False,
        collate_fn=default_data_collator,
    )
    return train_dataloader, val_dataloader


def prepare_training_components(accelerator, model, optimizer, train_dataloader, val_dataloader, lr_scheduler):
    """
    Wrap the model, optimizer, dataloaders, and scheduler for distributed training.
    """
    return accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )


@torch.no_grad()
def run_validation(accelerator, model, dataloader) -> dict[str, float]:
    """
    Evaluate the model on the validation set and return loss and perplexity.
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

    ensure_dir(exp_config["output_dir"])
    set_seed(exp_config["seed"])

    accelerator = create_accelerator(exp_config)

    tokenizer = AutoTokenizer.from_pretrained(
        exp_config["tokenizer_name_or_path"],
        cache_dir=exp_config.get("hf_cache_dir"),
        local_files_only=exp_config.get("hf_local_files_only", False),
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

    train_dataloader, val_dataloader = build_dataloaders(exp_config, tokenizer)

    model_config = LlamaConfig(**model_config_dict)
    model = LlamaForCausalLM(model_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_config["learning_rate"],
        weight_decay=exp_config["weight_decay"],
    )

    total_warmup_steps = int(exp_config["warmup_ratio"] * exp_config["max_train_steps"])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=total_warmup_steps,
        num_training_steps=exp_config["max_train_steps"],
    )

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = prepare_training_components(
        accelerator,
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="cs190c-hw2",
            config={
                **exp_config,
                "model_parameters": count_trainable_parameters(model),
            },
        )

    progress_bar = tqdm(
        range(exp_config["max_train_steps"]),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    completed_steps = 0
    optimizer.zero_grad()
    model.train()

    while completed_steps < exp_config["max_train_steps"]:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), exp_config["max_grad_norm"])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            completed_steps += 1
            progress_bar.update(1)
            gathered_train_loss = cast(torch.Tensor, accelerator.reduce(loss.detach(), reduction="mean"))
            train_loss = gathered_train_loss.item()

            if completed_steps % exp_config["logging_every_steps"] == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "learning_rate": current_lr,
                    },
                    step=completed_steps,
                )

            if completed_steps % exp_config["eval_every_steps"] == 0:
                metrics = run_validation(accelerator, model, val_dataloader)
                accelerator.log(metrics, step=completed_steps)
                if accelerator.is_main_process:
                    print(f"[step {completed_steps}] {format_metrics(metrics)}")
                model.train()

            if completed_steps % exp_config["save_every_steps"] == 0:
                save_dir = Path(exp_config["output_dir"]) / f"checkpoint-{completed_steps}"
                accelerator.save_state(str(save_dir))
                accelerator.save(
                    accelerator.get_state_dict(model),
                    save_dir / "model_state.pt",
                )

            if completed_steps >= exp_config["max_train_steps"]:
                break

    final_metrics = run_validation(accelerator, model, val_dataloader)
    accelerator.log(final_metrics, step=completed_steps)
    if accelerator.is_main_process:
        print(f"[final] {format_metrics(final_metrics)}")

    final_save_dir = Path(exp_config["output_dir"]) / "checkpoint-final"
    accelerator.save_state(str(final_save_dir))
    accelerator.save(
        accelerator.get_state_dict(model),
        final_save_dir / "model_state.pt",
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
