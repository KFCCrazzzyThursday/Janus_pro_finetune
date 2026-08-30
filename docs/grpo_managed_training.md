# GRPO 30-step checkpoint / validation loop

## Cadence and validation definition

`scripts/run_stage1_grpo_managed.sh` divides the 3,000-step run into 30-step
segments. Training releases all five L40S GPUs at each boundary, then
`scripts/run_stage1_grpo_validation.sh` performs deterministic greedy inference
on all 2,781 examples in `data/processed/tqa/val_prompt.jsonl`. The evaluator
uses the training-compatible `<think>` response prefix and records:

- answer accuracy (the primary best-checkpoint metric);
- strict-format and parse-failure rates;
- mean completion and `<think>` reasoning lengths in tokens;
- validation runtime and sample count.

An exact tie in accuracy is broken by strict-format rate and then lower parse
failure. Validation never calls the paid reasoning judge and never updates the
model.

## Recovery state

The Trainer retains the two newest `checkpoint-N` directories. Each contains
the LoRA adapter, optimizer, scheduler, Trainer state and the RNG state of all
five ranks. `scripts/grpo_run_state.py` opens the safetensors header and stores
the size and SHA-256 hash of every required file in
`janus_checkpoint_manifest.json` before the checkpoint becomes eligible for
resume.

The run root contains atomic pointers and machine-readable state:

- `last-checkpoint`: newest verified rolling checkpoint;
- `previous-checkpoint`: previous verified fallback;
- `best-checkpoint`: independent full copy of the best validated checkpoint;
- `resume_state.json`: verified candidates, rejected/corrupt candidates and
  the selected steps;
- `best.json`: best metric, source summary and checkpoint path;
- `validation/history.jsonl`: one authoritative validation record per step.

If training fails between boundaries, rerunning the managed launcher selects
the newest checkpoint whose hashes still match, trims `logging.jsonl` and
`completions.jsonl` after that step, rebuilds canonical TensorBoard events, and
replays only the interrupted segment. If validation fails, the checkpoint is
kept and validation runs again before further training.

## TensorBoard hygiene

Training and validation JSONL files are authoritative. At every completed
boundary the canonical event files are rebuilt from them, eliminating duplicate
future steps left by an interrupted attempt. Resource monitoring continues its
event step counter across segment restarts.

`scripts/run_tensorboard.sh` defaults to a scoped server with only `train` and
`val`. It enables TensorBoard's multi-file reload so checkpoint resumes can
rotate the canonical event files without leaving the live dashboard attached
to an obsolete file. The launcher also refreshes the TensorBoard child process
when the event filename set changes, clearing cached files from the preceding
segment instead of accumulating duplicate history. Resource events remain on
disk and can be included with `JANUS_TENSORBOARD_INCLUDE_RESOURCES=1`. Old
smoke/aborted runs remain on disk for audit but are not loaded. Set
`JANUS_TENSORBOARD_ALL_RUNS=1` only for deliberate comparison.

## Launch

```bash
export OPENAI_API_KEY='...'
bash scripts/run_stage1_grpo_managed.sh
```

The output directory defaults to
`outputs/stage1/tqa_grpo_lora_managed30`. It must be empty for a new run or
contain at least one verified managed checkpoint for a resume; the launcher
refuses to mix a fresh run with partial or historical event data. The original
50-step trial remains separately auditable in
`outputs/stage1/tqa_grpo_lora` and is excluded from the scoped TensorBoard
server.
