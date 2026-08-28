# Paper reproduction audit

The target is the thesis `final_print.pdf`, *Design of Downstream Fine-Tuning
Methods for Unified Image Generation and Understanding Models* (2025). The
paper trains `deepseek-ai/Janus-Pro-7B` in two stages and reports results on TQA
and ScienceQA.

## Directly recoverable settings

- TQA's 6,501 examples are exactly all `Diagram Multiple Choice` questions in
  the official training split. The official validation and test splits contain
  2,781 and 3,285 such questions.
- ScienceQA contains 12,726/4,241/4,241 total questions and
  6,218/2,097/2,017 image questions in train/val/test. The corresponding counts
  with non-empty `solution` fields are 11,515/3,848/3,839 and
  5,678/1,922/1,836, respectively.
- Stage-1 GRPO: LR 1e-6, constant scheduler, zero weight decay, gradient clip
  1.0, GaLore rank 512/update gap 128, batch size 8, four epochs/3,000 steps,
  G=16, asymmetric clip 0.2/0.28, cubic length scale 5e-7, and advantage
  threshold 0.2.
- Stage-2 SFT/joint-SFT/generation-GRPO settings are transcribed in
  `configs/paper.yaml`.

## Blocking ambiguities for an exact numerical reproduction

1. No source-code or checkpoint URL is given for the thesis implementation.
2. The ScienceQA SFT warm-up is described but has no hyperparameters or exact
   subset definition.
3. Equation (3.11) gives reward priors `[0.30, 0.20, 0.45, 0.05]`, while Table
   3.1 gives `[0.25, 0.25, 0.45, 0.05]`.
4. Initial KL beta, reward decay lambda, decoding temperature, completion
   length, gradient accumulation, and random seeds are omitted.
5. The format-reward prose names underscore tags (`choice_text`), while the
   appendix prompt and examples use space-separated labels (`choice text`).
6. TQA evaluation split, whether ScienceQA evaluation uses its full split or
   image-only subset, consistency perturbations, generation evaluation
   prompts/reference set, and FID implementation are not identified.
7. The image-GRPO equation adds a normalized L2 distance as a reward although
   lower distance is desirable; its alpha schedule description also conflicts
   with the written equation.
8. GPT-4o and GPT-4o-mini are required for filtering/judging, so exact runs need
   an `OPENAI_API_KEY` and a pinned model snapshot, neither of which is supplied.
9. The paper says a base model assigns each VQA item a 0-11 difficulty level,
   but does not report that grader's prompt or decoding settings.
10. Section 1.5 says the solution-bearing ScienceQA subset is used for SFT and
    then the full training set is used for GRPO, while Figure 3.1 labels the
    GRPO input as TQA. The thesis does not resolve this dataset contradiction.
11. The automatic threshold and all human accept/reject decisions that reduce
    6,501 synthesized TQA outputs to 5,307 are not reported.

The reproduction therefore records every chosen resolution of these
ambiguities and keeps raw IDs/predictions, instead of treating a single final
score as an exact match by construction.

## Explicit reproduction choices

- The unreported ScienceQA warm-up uses all 5,678 image-training examples with
  non-empty official solutions. Its optimizer settings reuse the paper's
  stage-2 understanding SFT settings: three epochs, global batch 64, LR 2e-5,
  constant schedule, GaLore rank 256/update gap 128, and seeds 42.
- The primary stage-1 GRPO run uses all 6,501 TQA training prompts. This is
  supported by Section 2.3, Figure 3.1, the middle-school characterization in
  the difficulty-reward discussion, and the subsequent 6,501-output synthesis.
  The conflicting Section 1.5 ScienceQA interpretation remains selectable with
  `JANUS_STAGE1_GRPO_DATA`. Before GRPO, the base model assigns each row a difficulty
  using the explicit prompt stored in `src/janus_repro/difficulty.py`. The
  primary two-turn response is parsed as a 1-12 grade; only an unparseable
  response falls back to a constrained, single-token `one`-through-`twelve`
  classification. Raw responses, fallback flags, and restricted probabilities
  are retained. This resolves omitted settings rather than claiming to recover
  the paper's grader prompt.
- Baseline reporting retains two ScienceQA views: the complete official split
  (including its supplied `hint`/passage) and the image-only split without
  context. The complete split is the primary table-comparison view because the
  paper explicitly restricts only TQA to image-question-answer triples. Both
  raw prediction sets are kept so this unreported choice remains auditable.
- The CR view applies both perturbation types the thesis names elsewhere:
  deterministic option reordering and the irrelevant prefix `Here is the
  question for you:`. Seed 42 and each new-to-old option map are stored. This is
  a reproducible proxy because the paper's actual CR perturbations are absent.
- Unless an experiment overrides them, omitted decoding values are temperature
  1.0 and maximum completion length 384; the initial stage-1 KL coefficient is
  0.04 (the trainer default) and decays over the first 500 steps.
- The paper's literal `<|Assistant|>: <think>` synthesis prefix is implemented
  as the Janus chat template's existing assistant marker plus
  `--response_prefix '<think>'`; decoded completions include the prefix for
  reward parsing.
- Table 3.1's reward prior `[0.25, 0.25, 0.45, 0.05]` is the default. The
  equation's alternative can be selected explicitly for the corresponding
  ablation.
- The appendix's space-separated output labels (`choice text`, `choice index`)
  are canonical. Parsing remains permissive only for reporting parse failures;
  the strict format reward requires the appendix form.
- The checkpoint remains canonical on NFS. At runtime it may be copied once to
  `/dev/shm` after capacity checks, solely to avoid concurrent FUSE page faults;
  this is volatile RAM, not a local-disk model directory.
- Stage-1 GRPO interprets the reported batch size 8 as eight prompts per update:
  with G=16 this is 128 completions, split into sixteen global micro-batches of
  eight completions. This interpretation makes 3,000 updates approximately
  consistent with approximately four passes over the 6,501-prompt data plus
  dynamic resamples.
- vLLM is not used in the L40S launcher because the thesis depends on an
  unpublished modified vLLM/TRL implementation and stock ms-swift does not
  provide a verified Janus multimodal vLLM path. Rollouts use Transformers;
  this is a documented infrastructure deviation, not presented as exact code
  parity with the unavailable implementation.
- At the user's direction, the unavailable GPT-4o/GPT-4o-mini services are
  replaced by `deepseek-v4-flash-vision-exp` through DeepSeek's OpenAI-compatible
  endpoint. Appendix A.1/A.2 prompts remain unchanged where the paper supplies
  them; the provider/model substitution is recorded in every judge log.
- CoT filtering first requires the exact answer text/index and Appendix-A.2
  strict format. Appendix A.1 is then run three times; the automatic filter
  requires mean overall score >=4 and each dimension mean >=3. These unreported
  thresholds are explicit reproduction choices. The unavailable manual phase
  is never simulated by truncating or ranking to the paper's target count of
  5,307.
