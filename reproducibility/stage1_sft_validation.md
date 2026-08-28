# Stage-1 SFT validation result

The completed checkpoint is `checkpoint-267-bf16`. It reached 267/267 steps
in 2 h 49 min 56 s on four L40S GPUs. Training loss moved from 1.3836 at step
1 to 0.0813 at step 267, while token accuracy moved from 0.7065 to 0.9771.

Held-out greedy generation used every official validation example and the same
decoding settings for the base and tuned checkpoints:

| Validation set | Samples | Base accuracy | SFT accuracy | Delta | SFT strict-format rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| ScienceQA image-only val | 2,097 | 77.78% | 86.50% | +8.73 pp | 99.43% |
| TQA diagram-MC val | 2,781 | 61.38% | 65.66% | +4.28 pp | 99.17% |

The preregistered severe-overfitting rule was not triggered: accuracy improved
on both the in-domain ScienceQA validation set and the independent TQA set.
The automatic decision was therefore `proceed_to_grpo`. The full predictions,
qualitative five-correct/five-incorrect gallery, TensorBoard events, and model
artifacts remain outside Git because they contain bulky dataset or checkpoint
content.
