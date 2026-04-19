# CS190C HW2 Remote Server Workflow

This file is the command-oriented workflow for running long experiments on a remote server.
It focuses on:

1. `nohup` background launch
2. log redirection
3. checking status after SSH disconnects
4. TensorBoard on a remote machine

If your compute node cannot access `huggingface.co`, use the offline workflow in section 11.

## 1. One-time setup on the server

```bash
git clone https://github.com/tzhazuma/CS190C-2026Spring-Assignment2.git
cd CS190C-2026Spring-Assignment2

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

mkdir -p logs
export HF_HOME=$PWD/.hf-cache
```

If your cluster already provides a prepared image or conda environment, keep that environment but still create the `logs/` directory.

## 2. Launch a single pilot run with `nohup`

Example for `pilot_s_lr5e4`:

```bash
RUN_NAME=pilot_s_lr5e4

nohup bash -lc "
  cd /path/to/CS190C-2026Spring-Assignment2 && \
  source .venv/bin/activate && \
  export HF_HOME=\$PWD/.hf-cache && \
  accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
    --experiment-config configs/experiments/pilot_s_lr5e4.yaml \
    --model-config configs/models/s.json
" > logs/${RUN_NAME}.log 2>&1 < /dev/null &

echo $! > logs/${RUN_NAME}.pid
disown
```

What this does:

- stdout and stderr both go to `logs/pilot_s_lr5e4.log`
- the process id is stored in `logs/pilot_s_lr5e4.pid`
- the job keeps running after your SSH session disconnects

## 3. Launch the final 2-GPU run with `nohup`

```bash
RUN_NAME=scaleup_m

nohup bash -lc "
  cd /path/to/CS190C-2026Spring-Assignment2 && \
  source .venv/bin/activate && \
  export HF_HOME=\$PWD/.hf-cache && \
  accelerate launch --config_file accelerate_configs/two_gpu_ddp.yaml scripts/train.py \
    --experiment-config configs/experiments/scaleup_m.yaml \
    --model-config configs/models/m.json
" > logs/${RUN_NAME}.log 2>&1 < /dev/null &

echo $! > logs/${RUN_NAME}.pid
disown
```

## 4. How to check whether the job is still running

After reconnecting to the server:

```bash
cd /path/to/CS190C-2026Spring-Assignment2
```

Check the process:

```bash
ps -fp $(cat logs/pilot_s_lr5e4.pid)
```

Watch the latest log lines:

```bash
tail -n 40 logs/pilot_s_lr5e4.log
```

Follow the log in real time:

```bash
tail -f logs/pilot_s_lr5e4.log
```

Only show the training summary lines:

```bash
grep -E "\[step|\[final" logs/pilot_s_lr5e4.log
```

Check GPU usage:

```bash
nvidia-smi
```

Refresh GPU usage every 5 seconds:

```bash
watch -n 5 nvidia-smi
```

## 5. How to stop a run manually

```bash
kill $(cat logs/pilot_s_lr5e4.pid)
```

If the process does not exit cleanly:

```bash
kill -9 $(cat logs/pilot_s_lr5e4.pid)
```

## 6. How to continue viewing results after SSH disconnect

Disconnecting from SSH does not stop a `nohup` job.
After reconnecting, you only need to:

1. enter the repository directory
2. inspect the `.log` file with `tail`
3. inspect the saved checkpoints under `outputs/...`
4. reconnect TensorBoard if needed

Useful commands:

```bash
ls outputs
ls outputs/pilot_s_lr5e4
ls outputs/scaleup_m
```

## 7. Run TensorBoard in the background on the server

```bash
nohup bash -lc "
  cd /path/to/CS190C-2026Spring-Assignment2 && \
  source .venv/bin/activate && \
  tensorboard --logdir outputs --host 0.0.0.0 --port 6006
" > logs/tensorboard.log 2>&1 < /dev/null &

echo $! > logs/tensorboard.pid
disown
```

Check it later:

```bash
ps -fp $(cat logs/tensorboard.pid)
tail -n 20 logs/tensorboard.log
```

Stop it:

```bash
kill $(cat logs/tensorboard.pid)
```

## 8. SSH tunnel for TensorBoard

On your local machine:

```bash
ssh -L 6006:localhost:6006 your_username@your_server
```

Then open:

```text
http://localhost:6006
```

If port `6006` is already in use, pick another port such as `16006` on both sides.

## 9. Recommended remote workflow

1. Start one pilot run with `nohup`.
2. Save the PID file and log file name.
3. Reconnect later and inspect the log using `tail -n 40`.
4. Use TensorBoard to compare `val_loss` across runs.
5. Copy final metrics into `EXPERIMENT_LOG_TEMPLATE.md`.
6. Copy the final selected rows into `RESULTS_SUMMARY_TEMPLATE.csv`.
7. Use `REPORT_TEMPLATE.md` to write the final report.

## 10. Important limitation

Current training code saves checkpoints periodically and at the end, but it does not expose a dedicated `--resume-from-checkpoint` training flag.
That means this guide covers background running and re-checking progress after disconnects, not automatic mid-run resume.

You can still:

- evaluate any saved checkpoint with `scripts/evaluate.py`
- inspect `outputs/.../checkpoint-*`
- restart a run manually if a job is interrupted

## 11. Offline workflow for servers that cannot reach Hugging Face

The error below means your server failed before training even started, because it could not download the tokenizer or dataset:

```text
[Errno 99] Cannot assign requested address
```

This is a network problem, not a model-config problem.

Typical reasons:

- the compute node has no outbound internet access
- `huggingface.co` is blocked from the server network
- the node cannot open external IPv4 or IPv6 connections correctly

### Step 1. Download assets on a machine that has internet

Run this on your laptop, login node, or any machine that can reach Hugging Face:

```bash
python scripts/prepare_offline_assets.py --output-dir offline_assets
```

This creates:

- `offline_assets/tokenizer/TinyStories-33M`
- `offline_assets/datasets/TinyStories`

### Step 2. Copy the assets to the server

Example with `scp`:

```bash
scp -r offline_assets your_username@your_server:/path/to/CS190C-2026Spring-Assignment2/
```

### Step 3. Point the experiment config to local paths

For example, edit your YAML like this:

```yaml
dataset_name: /path/to/CS190C-2026Spring-Assignment2/offline_assets/datasets/TinyStories
dataset_config_name: null
tokenizer_name_or_path: /path/to/CS190C-2026Spring-Assignment2/offline_assets/tokenizer/TinyStories-33M
hf_cache_dir: /path/to/CS190C-2026Spring-Assignment2/.hf-cache
hf_local_files_only: true
```

The training code now supports:

- a local tokenizer directory in `tokenizer_name_or_path`
- a local dataset directory in `dataset_name`
- offline tokenizer loading through `hf_local_files_only: true`

### Step 4. Re-run training

```bash
accelerate launch --config_file accelerate_configs/single_gpu.yaml scripts/train.py \
  --experiment-config configs/experiments/pilot_s_lr3e4.yaml \
  --model-config configs/models/s.json
```

### Optional workaround if your network can reach a mirror but not the official site

If your environment allows it and you trust the mirror you use, you can try setting a Hugging Face mirror endpoint before running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

This is optional and environment-dependent. The most reliable solution for clusters is still the offline asset workflow above.