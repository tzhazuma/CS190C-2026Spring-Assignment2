# CS190C HW2 Runbook

This repository is ready for the assignment workflow:

1. install dependencies,
2. run the 3 learning-rate pilots on model S,
3. choose the best learning rate,
4. run the 3 model-size pilots,
5. launch the 2-GPU final run,
6. evaluate the final checkpoint and export TensorBoard logs.

## 1. Environment setup

```bash
git clone https://github.com/tzhazuma/CS190C-2026Spring-Assignment2.git
cd CS190C-2026Spring-Assignment2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your server has a shared filesystem, set a Hugging Face cache directory before training:

```bash
export HF_HOME=$PWD/.hf-cache
```

You can also write the same cache path into the `hf_cache_dir` field of any experiment YAML.

## 2. Dataset and tokenizer download

No separate download script is required.

- `scripts/train.py` and `scripts/evaluate.py` download the tokenizer from `roneneldan/TinyStories-33M` automatically.
- `src/hw2/data.py` downloads the official `roneneldan/TinyStories` dataset automatically.
- The code uses only the `train` and `validation` splits, which matches the assignment.
- Downloaded artifacts are cached by Hugging Face, so later runs reuse the local cache.

## 3. Learning-rate sweep on model S

Run these three pilots on a single GPU:

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_s_lr3e4.yaml \
  --model-config configs/models/s.json

accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_s_lr5e4.yaml \
  --model-config configs/models/s.json

accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_s_lr8e4.yaml \
  --model-config configs/models/s.json
```

Compare validation loss curves in TensorBoard and choose the best learning rate.

## 4. Model-size sweep

The repository now includes three model sizes:

- `configs/models/xs.json`
- `configs/models/s.json`
- `configs/models/m.json`

Default size-sweep configs currently use learning rate `5e-4`.
If your learning-rate sweep picks a different best value, update `learning_rate` in:

- `configs/experiments/pilot_xs.yaml`
- `configs/experiments/pilot_s.yaml` or one of the `pilot_s_lr*.yaml` files
- `configs/experiments/pilot_m.yaml`

Then run:

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_xs.yaml \
  --model-config configs/models/xs.json

accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_s.yaml \
  --model-config configs/models/s.json

accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_m.yaml \
  --model-config configs/models/m.json
```

## 5. Final 2-GPU run

The provided final config already increases:

- number of GPUs,
- effective batch size,
- total training steps,
- model size relative to model S.

Run:

```bash
accelerate launch --config_file accelerate_configs/two_gpu_ddp.yaml scripts/train.py \
  --experiment-config configs/experiments/scaleup_m.yaml \
  --model-config configs/models/m.json
```

Training writes checkpoints under `outputs/scaleup_m/` and always saves a final checkpoint at:

```text
outputs/scaleup_m/checkpoint-final
```

## 6. Evaluate a checkpoint

Evaluate the final model with:

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/evaluate.py \
  --experiment-config configs/experiments/scaleup_m.yaml \
  --model-config configs/models/m.json \
  --checkpoint-path outputs/scaleup_m/checkpoint-final
```

You can also point `--checkpoint-path` directly to a `model_state.pt` file.

## 7. TensorBoard

Launch TensorBoard from the repository root:

```bash
tensorboard --logdir outputs
```

## 8. What is already implemented

- `Accelerator` initialization for training and evaluation
- dataset loading, tokenization, and packing to fixed-length language-modeling blocks
- dataloaders for train and validation
- gradient accumulation and optimizer scheduling
- validation loss and perplexity reporting
- TensorBoard logging
- periodic checkpoint saving
- final checkpoint saving for evaluation

## 9. What you still need to do manually

- actually run the 6 pilot experiments and the final 2-GPU experiment
- select the best learning rate from the first sweep
- write the pilot results table and short report
- optionally try FSDP and compare memory/throughput