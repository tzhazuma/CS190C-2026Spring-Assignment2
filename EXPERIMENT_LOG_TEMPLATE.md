# CS190C HW2 Experiment Log Template

Use this file during training so you do not lose details needed later for the report.

## Global Notes

- Server name:
- Python environment:
- Start date:
- TensorBoard port:
- Hugging Face cache path:

## Run Checklist

Before each run, record:

- exact experiment config path
- exact model config path
- accelerate config path
- launch command
- git commit hash
- GPU type and count

## Per-Run Record

Copy this block for each run.

### Run name

- Date:
- Purpose:
- Accelerate config:
- Experiment config:
- Model config:
- Git commit:
- Command:
- GPU count:
- GPU model:
- Per-device batch size:
- Gradient accumulation steps:
- Effective batch size:
- Max train steps:
- Learning rate:
- Warmup ratio:

TensorBoard observations:

- Early training trend:
- Mid training trend:
- Final validation trend:
- Was the run stable:

Final metrics:

- Final train loss:
- Final val loss:
- Final val perplexity:
- Best checkpoint path:

Decision:

- Keep / reject / promote to next stage:
- Reason:

## Learning-Rate Sweep Summary

| Run | LR | Final val loss | Final val perplexity | Stability | Decision |
| --- | --- | --- | --- | --- | --- |
| pilot_s_lr3e4 | 3e-4 | | | | |
| pilot_s_lr5e4 | 5e-4 | | | | |
| pilot_s_lr8e4 | 8e-4 | | | | |

Chosen learning rate:

Reason:

## Model-Size Sweep Summary

| Run | Model | Final val loss | Final val perplexity | Speed impression | Decision |
| --- | --- | --- | --- | --- | --- |
| pilot_xs | XS | | | | |
| pilot_s | S | | | | |
| pilot_m | M | | | | |

Chosen final model size:

Reason:

## Final 2-GPU Run Summary

| Item | Value |
| --- | --- |
| Run name | |
| GPUs | |
| Effective batch size | |
| Total training steps | |
| Final val loss | |
| Final val perplexity | |
| Final checkpoint | |

Was the final run consistent with pilot expectations?

- 
- 

## TensorBoard Screenshot Checklist

Collect these screenshots before writing the report:

- `val_loss` comparison for 3 learning-rate pilot runs
- `val_loss` comparison for XS, S, M model-size runs
- `val_perplexity` for the final 2-GPU run
- optional `train_loss` stability comparison if one run was noisy