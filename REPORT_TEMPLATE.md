# CS190C HW2 Report Template

Use this file as the starting structure for your final submission report.

## Title

CS190C HW2: Scaling from Pilot Runs to Multi-GPU Training

Name:

Student ID:

Date:

## 1. Setup

### Hardware

- Pilot stage GPU:
- Final stage GPUs:
- Approximate GPU memory per device:

### Software

- Python version:
- PyTorch version:
- Transformers version:
- Accelerate version:

### Dataset and tokenizer

- Dataset: `roneneldan/TinyStories`
- Tokenizer: `roneneldan/TinyStories-33M`
- Sequence length:

## 2. Pilot Experiments

### 2.1 Learning-rate sweep

Summarize the 3 pilot runs on model `S`.

| Run name | Model | Learning rate | Tokens/steps budget | Final val loss | Final val perplexity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| pilot_s_lr3e4 | S | 3e-4 | | | | |
| pilot_s_lr5e4 | S | 5e-4 | | | | |
| pilot_s_lr8e4 | S | 8e-4 | | | | |

What trends did you observe?

- 
- 
- 

Which learning rate did you choose, and why?

## 2.2 Model-size sweep

Use the best learning rate above and compare model sizes.

| Run name | Model | Hidden size | Layers | Heads | Final val loss | Final val perplexity | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pilot_xs | XS | 384 | 8 | 6 | | | |
| pilot_s | S | 512 | 12 | 8 | | | |
| pilot_m | M | 640 | 14 | 10 | | | |

What trends did you observe?

- 
- 
- 

## 3. Final 2-GPU Run

Describe the final configuration you selected.

| Item | Value |
| --- | --- |
| Experiment config | `configs/experiments/scaleup_m.yaml` or your modified version |
| Model config | |
| Number of GPUs | |
| Per-device batch size | |
| Gradient accumulation steps | |
| Effective batch size | |
| Max train steps | |
| Learning rate | |
| Why this configuration | |

Explain how the pilot study led to this final decision.

## 4. Final Results

| Metric | Value |
| --- | --- |
| Final validation loss | |
| Final validation perplexity | |
| Best checkpoint path | |

Did the large run behave as expected?

- 
- 

## 5. TensorBoard Evidence

Include or reference:

- learning-rate sweep comparison figure
- model-size sweep comparison figure
- final 2-GPU run curve figure

For each figure, state:

- what is being compared
- what trend matters
- how it affected your decision

## 6. Discussion

Answer the report questions directly.

1. What pilot experiments did you run?
2. What trends did you observe?
3. How did those trends affect your final scale-up decision?
4. What were your final validation results?
5. Did the large run behave as expected?
6. If you tried FSDP or DeepSpeed, how much memory did they save?

## 7. Optional Bonus

If you tried FSDP or another backend, summarize it here.

| Backend | Max memory per GPU | Tokens/sec | Larger model or batch possible? | Notes |
| --- | --- | --- | --- | --- |
| DDP | | | | |
| FSDP | | | | |

## 8. Conclusion

Write a short final conclusion.

- 
- 
- 