# Reproduction environment

## Original L40S full-tuning profile

- Date: 2026-08-27 (UTC)
- Host Python: 3.12.3, GCC 13.3.0
- Virtual environment: `/root/.venvs/janus-repro-py312`
- GPUs used: physical 0, 1, 3, 4; NVIDIA L40S 46,068 MiB each
- Unused GPU: physical 2
- NVIDIA driver: 580.173.02
- PyTorch: 2.6.0+cu124
- torchvision: 0.21.0+cu124
- transformers: 4.57.6
- TRL: 0.29.1
- ms-swift: 4.6.0.dev0, commit
  `087aa1cc481f97fb7f69f12dc7c224fb95857e86`
- DeepSpeed: not active. Version 0.18.9 was tested, but removed from this
  virtual environment because merely being discoverable makes Accelerate
  import its op builder, which fails without a CUDA Toolkit, `CUDA_HOME`, or
  `nvcc`, even during native DDP setup. The ZeRO configuration files are kept
  for a host with a complete CUDA Toolkit.
- accelerate: 1.14.0
- TensorBoard: 2.21.0; all trainer and resource events are stored below the
  NFS `outputs/` tree and served on port 6006.
- datasets: 4.8.4
- PEFT: 0.20.0
- safetensors: 0.8.0
- DeepSeek Janus code: 1.0.0, commit
  `1daa72fa409002d40931bd7b36a9280362469ead`
- ScienceQA code/data commit:
  `2cbf8318e07b9ece895bb2ae605e71e38d623264`
- Janus-Pro-7B Hugging Face revision:
  `5c3eb3fb2a3b61094328465ba61fcd4272090d67`
- FlashAttention and vLLM: not installed. The Janus composite wrapper now
  declares the SDPA support already provided by its stock Llama language
  model, so SFT and GRPO use PyTorch SDPA; GRPO uses Transformers rollout.
- Distributed launcher adaptation: four-process native FSDP2 `full_shard` is
  the default on physical GPUs 0, 1, 3, and 4. Plain DDP completed forward and
  backward but exhausted a 46 GiB L40S at the first GaLore optimizer update.
  The paper used four 80 GiB A100 GPUs; FSDP2 is therefore an explicit hardware
  adaptation for these four 46 GiB cards. DeepSpeed configurations remain for
  a host with a complete CUDA Toolkit.
- FSDP2 loads one model per rank directly from NFS before sharding. Its
  CPU-RAM-efficient rank-0 broadcast path cannot be used for this Janus
  composite model because rank-local state dictionaries contain different
  persistent-buffer entries. Peak verified host availability during a four-rank
  load was 51.7 GiB with zero swap use.
- Accelerate uses FP32 master parameters for trainable weights under BF16
  FSDP2. A narrow compatibility hook also promotes the 510,068,363 frozen
  floating-point elements to FP32 before sharding, preventing a mixed-dtype
  root FSDP group; forward/backward computation remains BF16.
- GaLore targets are passed explicitly as the Llama attention and MLP
  projections (`q/k/v/o_proj`, `gate/up/down_proj`). The current ms-swift
  revision computes its implicit default after constructing a detached
  `TrainingArguments` object, leaving the optimizer-side value as `None`.
- External judge substitution: `deepseek-v4-flash-vision-exp` at
  `https://api.deepseek.com` via the OpenAI-compatible client. Credentials are
  runtime-only and are never stored in this project.

The environment and all packages remain outside the NFS project only where a
Python virtual environment normally lives. Model weights, datasets,
checkpoints, logs, predictions, TensorBoard events, and telemetry CSV files are
canonical on NFS. GPU jobs use the NFS model directly and do not keep a local
or `/dev/shm` checkpoint copy.

## Local A100 LoRA profile (current validated run)

- Date validated: 2026-08-29 (UTC)
- GPUs: 8 x NVIDIA A100-SXM4-40GB on one NVLink host
- Host RAM: 251 GiB plus 63 GiB swap
- Python: 3.12
- PyTorch: 2.6.0+cu124; torchvision: 0.21.0+cu124
- transformers: 4.57.6; TRL: 0.29.1; PEFT: 0.20.0
- accelerate: 1.14.0; TensorBoard: 2.21.0
- Model: downloaded stage-1 SFT checkpoint from
  `Billyshears/Janus_pro_finetune`
- Dataset: 6,501 TQA prompts with model difficulty annotations
- Parallelism: 8-rank DDP, BF16, SDPA, Transformers rollout
- Tuning: LoRA r=32/alpha=64/dropout=0.05; 74.9568M trainable parameters
- Optimizer: AdamW, LR 1e-5; GaLore explicitly disabled
- Batch: per-device 1, gradient accumulation 16, generation batch 128,
  steps-per-generation 16, G=16, rollout chunk 4
- Observed peak trainer memory: 21.6 GiB/rank
- First 26-step mean compute time: 124.65 s/step; observed wall-clock mean
  approximately 138.4 s/step
- Process supervision: GRPO and TensorBoard run in separate tmux sessions

The Janus upstream patch exposes nested input/output embeddings, delegates
weight tying to the nested Llama configuration, and delegates
`prepare_inputs_for_generation`. The latter is required for PEFT's causal-LM
wrapper. The ms-swift patch bounds Transformers rollout batch size and avoids
forking dataset preprocessing when `num_proc=1`.

See `docs/a100_lora_ddp_grpo_run.md` and `deploy/local_a100/README.md` for the
exact runbook. The older L40S and Blackwell sections remain historical profiles
and do not describe the local A100 launcher's defaults.

## Blackwell migration target

- Date provisioned: 2026-08-28 (UTC)
- GPUs: 4 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each
- Driver: 580.173; driver-reported maximum CUDA 13.0
- System CUDA toolkit: 12.8
- Host RAM: 755 GiB total
- Workspace capacity at provisioning: 355 GiB free
- Python: 3.12.3
- Deployment root, virtual environment, cache, data, checkpoint, and outputs:
  `/workspace/Janus_pro_finetune`
- PyTorch target: 2.7.1 CUDA 12.8; torchvision 0.22.1 CUDA 12.8
- GRPO target mapping: logical GPUs `0,1,2,3`, FSDP2, BF16, SDPA,
  Transformers rollout
- Checkpoint transport: Hugging Face Xet to
  `Billyshears/Janus_pro_finetune`, then direct download into `/workspace`

The Vast `/workspace` filesystem persists across stop/start but is not a named
persistent volume: recycling or destroying the instance erases it. Formal jobs
therefore remain Supervisor-managed, and checkpoints should be synced out
before instance destruction.
