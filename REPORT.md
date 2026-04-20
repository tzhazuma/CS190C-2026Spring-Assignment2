# CS190C HW2: Scaling from Pilot Runs to Multi-GPU Training

Name: [ZhiHao Tang]

Student ID: [2022533131]

Date: 2026-04-20

## 1. Setup

### Hardware

- Pilot stage GPU: 1x NVIDIA H20
- Final stage GPUs: 2x NVIDIA H20
- Approximate GPU memory per device: 97871 MiB

### Software

- Python version: 3.13.9
- PyTorch version: 2.11.0+cu126
- Transformers version: 5.5.4
- Accelerate version: 1.13.0

### Dataset and tokenizer

- Dataset: roneneldan/TinyStories
- Tokenizer: roneneldan/TinyStories-33M
- Sequence length: 512

## 2. Pilot Experiments

### 2.1 Learning-rate sweep

I ran the required 3-run learning-rate sweep on model S using the single-GPU setting and the same optimizer, schedule, tokenizer, and sequence length.

| Run name | Model | Learning rate | Tokens/steps budget | Final val loss | Final val perplexity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| pilot_s_lr3e4 | S | 3e-4 | 1500 steps (~24.6M tokens) | 2.1685 | 8.7455 | Stable, but clearly underfit relative to the other two learning rates |
| pilot_s_lr5e4 | S | 5e-4 | 1500 steps (~24.6M tokens) | 2.0595 | 7.8424 | Best validation loss in the LR sweep |
| pilot_s_lr8e4 | S | 8e-4 | 1500 steps (~24.6M tokens) | 2.0843 | 8.0391 | Still stable, but slightly worse than 5e-4 |

What trends did I observe?

- 3e-4 was too conservative for this pilot budget and ended with the worst validation loss.
- 5e-4 produced the best validation loss and perplexity on model S.
- 8e-4 did not diverge, but it also did not beat 5e-4, so there was no reason to prefer the larger step size.

Which learning rate did I choose, and why?

I chose 5e-4 for the size sweep because it gave the lowest final validation loss on the S model while remaining stable throughout the pilot run.

### 2.2 Model-size sweep

Using the best pilot learning rate from above, I compared three model sizes while keeping the tokenizer, sequence length, schedule family, and pilot training budget fixed.

| Run name | Model | Hidden size | Layers | Heads | Final val loss | Final val perplexity | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_xs | XS | 384 | 8 | 6 | 2.1663 | 8.7257 | Smallest model, worst validation result in the size sweep |
| pilot_s | S | 512 | 12 | 8 | 2.0679 | 7.9085 | Stronger than XS, but still behind M |
| pilot_m | M | 640 | 14 | 10 | 1.9979 | 7.3734 | Best validation result in the size sweep |

What trends did I observe?

- Validation loss improved monotonically as model size increased from XS to S to M.
- The M model achieved the best generalization under the same pilot token budget.
- Because M was already best at the pilot budget, it was the most reasonable candidate for a longer 2-GPU scale-up run.

## 3. Final 2-GPU Run

I selected the M model for the final run because it won the size sweep. The final experiment increased GPU count from 1 to 2, effective batch size from 32 to 128, and total optimizer steps from 1500 to 10000.

| Item | Value |
| --- | --- |
| Experiment config | configs/experiments/scaleup_m.yaml |
| Model config | configs/models/m.json |
| Number of GPUs | 2 |
| Per-device batch size | 8 |
| Gradient accumulation steps | 8 |
| Effective batch size | 128 |
| Max train steps | 10000 |
| Learning rate | 3e-4 |
| Why this configuration | M was the best pilot model. For the much longer 2-GPU run with a 4x larger effective batch, I used a more conservative learning rate than the pilot winner to keep the long run stable. |

How did the pilot study lead to this final decision?

The pilot study established two facts. First, 5e-4 was the best single-GPU learning rate on model S under the pilot budget. Second, the M model was the best model size under the same controlled budget. That made M the correct scale-up target. For the final run, I kept the same overall recipe but reduced the learning rate to 3e-4 because the final run was much longer and used a substantially larger effective batch. The final result validated that choice: the run converged cleanly and produced a much lower validation loss than any pilot experiment.

## 4. Final Results

| Metric | Value |
| --- | --- |
| Final validation loss | 1.3831 |
| Final validation perplexity | 3.9873 |
| Best checkpoint path | outputs/scaleup_m/checkpoint-final |

Did the large run behave as expected?

- Yes. The final 2-GPU run substantially improved over the best pilot M run, reducing validation loss from 1.9979 to 1.3831.
- Yes. The final validation perplexity dropped from 7.3734 in pilot_m to 3.9873 in scaleup_m, which is consistent with the expectation that more compute and a larger total training budget should help.

## 5. TensorBoard Evidence

Static figures exported from the TensorBoard event files:

- [report_assets/lr_sweep_val_loss.png](report_assets/lr_sweep_val_loss.png)
- [report_assets/size_sweep_val_loss.png](report_assets/size_sweep_val_loss.png)
- [report_assets/scaleup_m_curves.png](report_assets/scaleup_m_curves.png)

What each figure shows and why it matters:

- Learning-rate sweep figure: compares validation loss curves for 3e-4, 5e-4, and 8e-4 on model S. The important trend is that 5e-4 finishes lowest, which is why I used it for the controlled size sweep.
- Model-size sweep figure: compares validation loss curves for XS, S, and M at the same learning rate and pilot budget. The important trend is that M stays best by the end, which is why I selected it for scale-up.
- Final 2-GPU curve figure: shows the scaleup_m training and validation loss curves over the long run. The important trend is continued improvement through the final checkpoint, which supports the decision to allocate more compute to the M model.

TensorBoard log locations used for these figures:

- outputs/pilot_s_lr3e4/cs190c-hw2
- outputs/pilot_s_lr5e4/cs190c-hw2
- outputs/pilot_s_lr8e4/cs190c-hw2
- outputs/pilot_xs/cs190c-hw2
- outputs/pilot_s/cs190c-hw2
- outputs/pilot_m/cs190c-hw2
- outputs/scaleup_m/cs190c-hw2

## 6. Discussion

1. What pilot experiments did I run?

I ran the required six pilots: three learning-rate sweeps on S and three model-size sweeps using the chosen pilot learning rate.

2. What trends did I observe?

The main trends were that 5e-4 was the best pilot learning rate on S and that validation performance improved consistently with model size from XS to M.

3. How did those trends affect my final scale-up decision?

They pointed to M as the strongest candidate for a longer run. I therefore scaled up M to 2 GPUs, increased the effective batch size, and increased the total training budget.

4. What were my final validation results?

The final 2-GPU run reached validation loss 1.3831 and validation perplexity 3.9873 at outputs/scaleup_m/checkpoint-final.

5. Did the large run behave as expected?

Yes. It clearly outperformed every pilot run and achieved a large perplexity improvement over the single-GPU pilot_m baseline.

6. If I tried FSDP or DeepSpeed, how much memory did they save?

I did not run the optional FSDP or DeepSpeed experiments in the current artifact set.

## 7. Optional Bonus

Optional bonus experiments were not attempted in the current workspace outputs.

| Backend | Max memory per GPU | Tokens/sec | Larger model or batch possible? | Notes |
| --- | --- | --- | --- | --- |
| DDP | N/A from current artifacts | N/A from current artifacts | N/A | Only default DDP-style 2-GPU run is present |
| FSDP | Not run | Not run | Not evaluated | Optional bonus not attempted |

## 8. Conclusion

- The required six pilot runs and one final 2-GPU run are present in the outputs directory.
- The pilot study identified 5e-4 as the strongest pilot learning rate on S and M as the strongest model size.
- The final scale-up run on 2 GPUs achieved the best result overall with validation loss 1.3831 and perplexity 3.9873.
