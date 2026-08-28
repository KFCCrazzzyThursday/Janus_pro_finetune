# Janus-Pro thesis reproduction

This repository reproduces the experiments in `final_print.pdf`. The first
run used physical L40S GPUs `0,1,3,4`; the current deployment also supports
four RTX PRO 6000 Blackwell 96 GB GPUs as logical devices `0,1,2,3`. Raw
datasets, model weights, credentials, caches, and generated outputs are
intentionally excluded from Git.

Current pinned upstream revisions:

- DeepSeek Janus: `1daa72fa409002d40931bd7b36a9280362469ead`
- ScienceQA: `2cbf8318e07b9ece895bb2ae605e71e38d623264`
- ms-swift: `087aa1cc481f97fb7f69f12dc7c224fb95857e86`

The paper settings and all identified ambiguities are recorded in
`configs/paper.yaml` and `reproducibility/paper_audit.md`. Exact software and
hardware versions are in `reproducibility/environment.md`.

For the single-host `8 x A100-SXM4-40GB` deployment, see
`deploy/local_a100/README.md`. Its validated profile is LoRA + 8-rank DDP with
AdamW (GaLore disabled), launched and monitored in tmux. The measured run
snapshot is in `docs/a100_lora_ddp_grpo_run.md`.

Prepare data:

```bash
/root/.venvs/janus-repro-py312/bin/python scripts/prepare_data.py
```

GPU launchers default to the original L40S device list and allow a deployment
to override it explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,3,4 ...
```

On the original host, the canonical checkpoint, datasets, logs, and experiment
outputs stay on NFS. GPU launchers read that model directly by default
(`JANUS_STAGE_MODEL_TO_RAM=0`). Optional RAM staging checks both available host
RAM and tmpfs capacity before copying, and keeps a configurable 64 GiB safety
margin.

## Clean Blackwell deployment

The remote deployment keeps every large or persistent artifact below
`/workspace`:

```bash
cd /workspace
git clone https://github.com/KFCCrazzzyThursday/Janus_pro_finetune.git
cd Janus_pro_finetune
bash scripts/setup_remote_blackwell.sh
```

The setup script disables the generic proxy only in its own process, checks out
the three pinned upstream revisions, applies the versioned compatibility
patches, creates `/workspace/Janus_pro_finetune/.venv`, installs PyTorch 2.7.1
with CUDA 12.8 support, verifies the Blackwell compute capability, and runs the
test suite.

The completed SFT checkpoint is hosted separately from Git. Download it
directly into `/workspace`:

```bash
HF_HOME=/workspace/Janus_pro_finetune/.hf_home \
  /workspace/Janus_pro_finetune/.venv/bin/python \
  scripts/download_sft_checkpoint.py \
  --local-dir /workspace/Janus_pro_finetune/models/Janus-Pro-7B-stage1-sft
```

After copying the TQA JSONL and images, rewrite the source-host prefix and
validate every referenced image:

```bash
.venv/bin/python scripts/relocate_jsonl_paths.py \
  data/processed/tqa/train_prompt_model_difficulty.jsonl \
  data/processed/tqa/train_prompt_model_difficulty.remote.jsonl \
  --from-prefix /root/nfs/LiYJ/Janus \
  --to-prefix /workspace/Janus_pro_finetune \
  --check-images
```

The full-shape preflight uses all production rollout dimensions but replaces
the paid reasoning judge with a logged stub and performs one optimizer step:

```bash
JANUS_STAGE1_GRPO_OUTPUT=/workspace/Janus_pro_finetune/outputs/smoke/grpo_full_shape \
  bash deploy/remote/run_stage1_grpo.sh --memory-smoke
```

For the formal run, place only `export OPENAI_API_KEY=...` in the runtime-only,
mode-0600 file `/dev/shm/janus-grpo-secret.env`, install
`deploy/remote/janus-grpo.supervisor.conf` into Supervisor, and start the
`janus-grpo` program. No credential belongs in this repository or its logs.

Real-time curves are written for every SFT and GRPO run. Start the shared
TensorBoard server with:

```bash
bash scripts/run_tensorboard.sh
```

It watches the complete NFS `outputs/` tree, listens on port 6006, and reloads
every five seconds. Use `http://<training-host>:6006`, or forward remote port
6006 to your workstation and open `http://127.0.0.1:6006`.

SFT reports loss, token accuracy, learning rate, gradient norm, input tokens,
runtime, and throughput. GRPO additionally reports the curves shown in the
paper: overall reward mean/variance and the Accuracy, Format, Length, and
Reasoning reward means/variances. Extra GRPO diagnostics include component
contributions, dynamic reward weights, correct/strict-format/reasoning-active
fractions, retained groups and advantages, KL and its scheduled coefficient,
entropy, low/high clipping ratios, completion length/truncation, and rollout
throughput. A separate resource run records physical GPUs 0, 1, 3, and 4
(memory, utilization, power, temperature) plus host RAM, swap, and load to both
TensorBoard and `resource_metrics.csv`; physical GPU 2 is never sampled.

The optimized and original-GRPO ablation runs use separate run directories so
TensorBoard can overlay their 3,000-step reward curves for Figure 5.1.

Run the four-card baseline smoke test:

```bash
bash scripts/run_baseline.sh tqa val --max-samples 4 --max-new-tokens 64
```

ScienceQA is prepared both as the full official split (with passage/hint) and
as an image-only view because the paper does not state which one underlies its
reported score. Run the primary full-split baseline with:

```bash
bash scripts/run_baseline.sh scienceqa/full test
```

Collected base-model scores and paper deltas are written to
`outputs/baseline/summary.json` by:

```bash
python scripts/collect_baseline_results.py
```

The deterministic proxy for the paper's unreported CR perturbation set is
prepared alongside each split. For example:

```bash
bash scripts/run_baseline.sh tqa test_consistency
```

Validate one optimizer step, then run the paper's otherwise-unreported
ScienceQA SFT warm-up with the assumptions documented in the audit:

```bash
bash scripts/run_stage1_sft.sh --smoke
bash scripts/run_stage1_sft.sh
```

Run a one-step GRPO plumbing check (its GPT reward is replaced by a clearly
logged zero-valued stub only in this smoke mode):

```bash
bash scripts/run_annotate_tqa_difficulty.sh --smoke
bash scripts/run_annotate_tqa_difficulty.sh
bash scripts/run_stage1_grpo.sh --smoke
```

The real GRPO run uses the configured DeepSeek replacement for the paper's
retired external judge. Export the key in the calling shell without storing it
in the repository:

```bash
export OPENAI_API_KEY='...'
bash scripts/run_stage1_grpo.sh
```

On the local A100 host, prefer the tmux helper. It reads the key silently and
does not embed it in the pane command or a file:

```bash
bash deploy/local_a100/start_grpo_tmux.sh
bash deploy/local_a100/start_tensorboard_tmux.sh
```

Generate one CoT for every TQA training item and run the auditable automatic
portion of the paper's two-stage filter:

```bash
bash scripts/run_stage1_synthesis.sh
bash scripts/run_filter_tqa_synthesis.sh
```

The filter is resumable and keeps every raw judge response. It reports its
automatic accepted count alongside the paper's 5,307; it does not fabricate
the unavailable human screening decisions.

After filtering, validate and run the paper's stage-2 understanding SFT:

```bash
bash scripts/run_stage2_understanding_sft.sh --smoke
bash scripts/run_stage2_understanding_sft.sh
```
