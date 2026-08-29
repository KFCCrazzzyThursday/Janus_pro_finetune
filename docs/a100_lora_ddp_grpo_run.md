# 8×A100 Janus-Pro-7B LoRA GRPO run

This note records the operational profile validated on 2026-08-29. It is a
systems/runbook snapshot, not a final convergence report.

## Validated configuration

| Item | Value |
|---|---|
| GPUs | 8 × NVIDIA A100-SXM4-40GB, one host, NVLink |
| Model | `Billyshears/Janus_pro_finetune`, downloaded as `models/Janus-Pro-7B-stage1-sft` |
| Dataset | 6,501 TQA training prompts with model difficulty annotations |
| Parallelism | PyTorch DDP, 8 ranks, BF16, SDPA |
| Tuning | LoRA r=32, alpha=64, dropout=0.05 |
| LoRA targets | `q/k/v/o_proj`, `gate/up/down_proj` |
| Trainable parameters | 74.9568M / 7.495B (1.00%) |
| Optimizer | AdamW, LR 1e-5; GaLore disabled |
| GRPO | DAPO loss, G=16, clip=0.2/0.28, KL beta=0.04 |
| Reward weighting | stabilized range-normalized dispersion, 50% scheduled-prior mix (`paper` remains available) |
| Batch | per-device=1, GA=16, generation batch=128, steps/generation=16 |
| Generation | Transformers backend, rollout chunk=4, max completion=384 |
| Checkpointing | gradient checkpointing enabled; save every 500 steps |

For LoRA, TRL computes reference log probabilities with the policy adapter
disabled. DDP therefore holds one BF16 7B base model per rank rather than a
second reference-model copy. The frozen vision tower and aligner remain part of
the forward graph but are not trainable.

## Initial observed result

The formal run passed checkpoint loading, PEFT injection, rollout generation,
all four reward functions, reference/policy log-probability computation,
backward, AdamW, and repeated optimizer updates. Over the first 26 completed
steps:

- trainer-reported peak memory was 21.6 GiB per rank;
- mean compute `step_time` was 124.65 seconds (median 124.37 seconds);
- observed wall-clock mean was about 138.4 seconds per step;
- completion length averaged about 70 tokens, with the configured cap of 384;
- the early 3,000-step ETA was approximately 4 days 18 hours.

These figures include variable autoregressive generation and dynamic resampling,
so short-window ETAs should not be treated as a convergence guarantee.

## Why this profile uses LoRA DDP

Full-parameter FSDP2 probes completed rollout and backward on this host, but the
GaLore projection workspace exhausted a 40GB rank at the optimizer boundary.
LoRA removes that projection and reduces trainable state to about 75M
parameters. With that memory profile, data-parallel replicas fit comfortably
and DDP avoids the complexity of sharding a custom multimodal wrapper.

The repository retains optional FSDP2 compatibility code for other profiles,
but `deploy/local_a100/run_stage1_grpo.sh` defaults to LoRA, AdamW, and DDP.
The A100 launcher also defaults to the stabilized reward weighting described in
`docs/reward_variance_weighting_revision.md`; export
`JANUS_REWARD_WEIGHTING=paper` for the original equations-(3.9)-(3.10) ablation.

## Operation

Start a formal run in a detached tmux session. The helper prompts for the judge
key if it is not already in the calling environment and passes it through pane
stdin without embedding it in a command or file:

```bash
bash deploy/local_a100/start_grpo_tmux.sh
tmux attach -t janus_grpo_lora_ddp
```

Detach with `Ctrl-b d`. For an offline one-step plumbing check:

```bash
JANUS_GRPO_MAX_STEPS=1 JANUS_GRPO_SAVE_STEPS=1 \
  bash deploy/local_a100/start_grpo_tmux.sh --offline
```

Start TensorBoard separately:

```bash
bash deploy/local_a100/start_tensorboard_tmux.sh
```

TensorBoard listens on port 6006. Prefer an authenticated platform port proxy
or SSH forwarding. An account-less public tunnel exposes the dashboard without
authentication and should only be used deliberately for non-sensitive logs.

## Artifacts and security

Models, datasets, outputs, TensorBoard events, judge caches, logs, environment
files, crash dumps, and credentials are excluded from Git. Only source,
versioned upstream patches, launchers, tests, and documentation belong in the
repository. Rotate any credential that has been pasted into a chat or terminal
transcript.
