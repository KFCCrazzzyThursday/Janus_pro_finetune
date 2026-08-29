# TQA validation：原始模型、ScienceQA SFT 与 GRPO step 50 对比

> 生成时间：2026-08-29 15:26 UTC  
> 数据：TQA val，共 2,781 题；三组预测按样本 ID 一一配对。

## 结论摘要

- ScienceQA SFT 将 TQA 准确率从 61.38% 提高到 65.66%（+4.28 个百分点）。
- GRPO step 50 的准确率为 65.91%，相对 SFT 为 +0.25 个百分点；它改对 126 题，同时改错 119 题，净变化 +7 题。
- SFT 与 step 50 的配对 McNemar 精确检验 p=0.702；当前没有证据表明 50 步带来了整体准确率变化，但样本级答案发生了明显置换。
- step 50 的平均完整回答长度为 82.04 token，相比 SFT 的 78.31 增加 4.77%。

## 评测口径

- 原始模型：`models/Janus-Pro-7B`，未经过本项目 SFT/GRPO。
- SFT：`outputs/stage1/scienceqa_sft/checkpoint-267-bf16`；这是 GRPO 的初始化模型。
- step 50：上述 SFT 模型叠加 `outputs/stage1/tqa_grpo_lora/checkpoint-50` LoRA。
- 三者均使用同一 TQA val、greedy decoding、seed 42、最多 384 个新 token。step 50 的 `<think>` 前缀作为未闭合的 Assistant 上下文输入，prompt 尾部不含 EOS。
- `严格格式率`沿用项目解析器；`规范单标签率`进一步要求 `<think>`、`</think>`、`<choice text>`、`<choice index>` 各恰好出现一次。
- 长度由同一 Janus tokenizer 在保存的完整 response 上统一重算；推理长度只对能完整解析 `<think>...</think>` 的回答统计。

## Response prefix / EOS 修正记录

最初的 step-50 验证把 `<think>` 放入最后一条 Assistant 消息，但 Janus 对话模板在非空 Assistant 内容后自动追加 EOS，实际 prompt 尾部成为 `<think><EOS>`。模型因此把前一回答视为已经结束，并在新生成中再次输出 `<think>`。解码器再补回输入前缀后，保存结果出现 `<think><think>`。

修正后的验证在生成前只移除该模板附带的末尾 EOS，保留 `<think>` 作为开放的 Assistant 上下文；解码后补回前缀仅用于还原完整 response。summary 通过 `response_prefix_conditioning=assistant_context_without_terminal_eos` 记录这一口径。

- 修正后的 2,781 条输出中，双 `<think>` 开头为 0 条。
- 旧错误结果保存在 `/root/nfs/LiYJ/Janus/outputs/stage1/tqa_grpo_lora/validation/checkpoint-000050/tqa_val_invalid_terminal_eos_20260829`，只用于审计，不应用于模型比较或 best-checkpoint 判断。

| step-50 协议 | 准确率 | 严格格式率 | 解析失败率 | 平均推理 token | 平均完整回答 token |
|---|---:|---:|---:|---:|---:|
| 旧协议：`<think><EOS>`（无效） | 65.37% | 91.73% | 2.09% | 80.99 | 103.65 |
| 修正协议：开放 `<think>` | 65.91% | 98.89% | 0.72% | 59.23 | 82.04 |

## 总体指标

| 指标 | 原始模型 | ScienceQA SFT | GRPO step 50 | step 50 vs 原始 | step 50 vs SFT |
|---|---:|---:|---:|---:|---:|
| 回答准确率 | 61.38% (1,707) | 65.66% (1,826) | 65.91% (1,833) | +4.53 pp | +0.25 pp |
| 严格格式率 | 0.00% (0) | 99.17% (2,758) | 98.89% (2,750) | +98.89 pp | -0.29 pp |
| 规范单标签率 | 0.00% (0) | 99.17% (2,758) | 98.89% (2,750) | +98.89 pp | -0.29 pp |
| 答案索引解析失败率 | 0.00% (0) | 0.68% (19) | 0.72% (20) | +0.72 pp | +0.04 pp |
| 平均完整回答长度（token） | 33.79 | 78.31 | 82.04 | +48.25 | +3.73 |
| 平均可解析推理长度（token） | — | 56.37 | 59.23 | — | +2.86 |

### 长度分布

| 模型 | 完整回答 mean / median / p90 / max | 可解析推理 mean / median / p90 / max | 推理可解析覆盖率 |
|---|---:|---:|---:|
| 原始 Janus-Pro-7B | 33.79 / 37 / 60 / 173 | — | 0.00% (0/2,781) |
| ScienceQA SFT | 78.31 / 65 / 129 / 384 | 56.37 / 45 / 107 / 227 | 99.17% (2,758/2,781) |
| SFT + GRPO step 50 | 82.04 / 67 / 139 / 387 | 59.23 / 46 / 114 / 328 | 98.89% (2,750/2,781) |

## 正确性转移

`对→错`和`错→对`是同一题在两模型之间的配对变化。

| 对比 | 对→对 | 对→错 | 错→对 | 错→错 | 净增正确题 |
|---|---:|---:|---:|---:|---:|
| 原始 Janus-Pro-7B → ScienceQA SFT | 1,457 (52.39%) | 250 (8.99%) | 369 (13.27%) | 705 (25.35%) | +119 |
| ScienceQA SFT → SFT + GRPO step 50 | 1,707 (61.38%) | 119 (4.28%) | 126 (4.53%) | 829 (29.81%) | +7 |
| 原始 Janus-Pro-7B → SFT + GRPO step 50 | 1,448 (52.07%) | 259 (9.31%) | 385 (13.84%) | 689 (24.78%) | +126 |

### 三模型结果组合

| 原始/SFT/step50 | 含义 | 数量 | 占比 |
|---|---|---:|---:|
| `000` | 三者都错 | 627 | 22.55% |
| `001` | 仅 step 50 正确 | 78 | 2.80% |
| `010` | 仅 SFT 正确 | 62 | 2.23% |
| `011` | SFT 改对，step 50 保持 | 307 | 11.04% |
| `100` | 仅原始模型正确 | 202 | 7.26% |
| `101` | SFT 改错，step 50 恢复 | 48 | 1.73% |
| `110` | 原始与 SFT 正确，step 50 改错 | 57 | 2.05% |
| `111` | 三者都对 | 1,400 | 50.34% |

## 代表性样例

样例按结果桶自动选取，并优先选择选项不是纯数字、索引有效且回答较短的题；它们用于解释变化类型，不替代总体统计。

### GRPO 新改对：原始错、SFT 错、step 50 对

- ID：`DQ_009107`；lesson：`cell structures`
- 问题：The vacuoles float inside which part of the cell?
- 选项：["N", "W", "P", "C"]
- Gold：`2` — P

| 模型 | 预测 | 正确 | 严格格式 |
|---|---|:---:|:---:|
| 原始 Janus-Pro-7B | 1 — W | ✗ | ✗ |
| ScienceQA SFT | 1 — W | ✗ | ✓ |
| SFT + GRPO step 50 | 2 — P | ✓ | ✓ |

原始 Janus-Pro-7B 输出：

```text
Answer: 1) W. The vacuoles float inside the cell's cytoplasm, which is labeled as 'W' in the diagram.
```

ScienceQA SFT 输出：

```text
<think>The vacuoles float inside the cell membrane. The cell membrane is labeled as W in the diagram.</think>
<choice text>: W
<choice index>: 1
```

SFT + GRPO step 50 输出：

```text
<think>The vacuoles float inside the cell membrane. The cell membrane is labeled with the letter P in the diagram.</think>
<choice text>: P
<choice index>: 2
```

### SFT 改对且 GRPO 保持：原始错、SFT 对、step 50 对

- ID：`DQ_002293`；lesson：`soils`
- 问题：What is the name of the surface litter?
- 选项：["E horizon", "B horizon", "A horizon", "O horizon"]
- Gold：`3` — O horizon

| 模型 | 预测 | 正确 | 严格格式 |
|---|---|:---:|:---:|
| 原始 Janus-Pro-7B | 0 — E horizon | ✗ | ✗ |
| ScienceQA SFT | 3 — O horizon | ✓ | ✓ |
| SFT + GRPO step 50 | 3 — O horizon | ✓ | ✓ |

原始 Janus-Pro-7B 输出：

```text
Answer: 0) E horizon
```

ScienceQA SFT 输出：

```text
<think>The O horizon is the surface litter.</think>
<choice text>: O horizon
<choice index>: 3
```

SFT + GRPO step 50 输出：

```text
<think>The O horizon is the surface litter.</think>
<choice text>: O horizon
<choice index>: 3
```

### GRPO 回归：原始对、SFT 对、step 50 错

- ID：`DQ_009180`；lesson：`cell structures`
- 问题：Which letter is Nucleus represented in the Diagram?
- 选项：["L", "U", "S", "N"]
- Gold：`3` — N

| 模型 | 预测 | 正确 | 严格格式 |
|---|---|:---:|:---:|
| 原始 Janus-Pro-7B | 3 — N | ✓ | ✗ |
| ScienceQA SFT | 3 — N | ✓ | ✓ |
| SFT + GRPO step 50 | 2 — S | ✗ | ✓ |

原始 Janus-Pro-7B 输出：

```text
Answer: 3) N
```

ScienceQA SFT 输出：

```text
<think>The nucleus is labeled with the letter N in the diagram.</think>
<choice text>: N
<choice index>: 3
```

SFT + GRPO step 50 输出：

```text
<think>The nucleus is the part of the cell that contains the cell's genetic material. It is often referred to as the control center of the cell. The nucleus is shown in the center of the cell in the diagram.</think>
<choice text>: S
<choice index>: 2
```

### SFT 的改进被 GRPO 撤销：原始错、SFT 对、step 50 错

- ID：`DQ_001582`；lesson：`inside earth`
- 问题：How many layers of the earth are shown here?
- 选项：["three", "one", "four", "two"]
- Gold：`2` — four

| 模型 | 预测 | 正确 | 严格格式 |
|---|---|:---:|:---:|
| 原始 Janus-Pro-7B | 3 — two | ✗ | ✗ |
| ScienceQA SFT | 2 — four | ✓ | ✓ |
| SFT + GRPO step 50 | 0 — three | ✗ | ✓ |

原始 Janus-Pro-7B 输出：

```text
Answer: 3
```

ScienceQA SFT 输出：

```text
<think>The diagram shows the crust, the solid inner core, the liquid outer core, and the mantle. So, there are four layers of the earth shown here.</think>
<choice text>: four
<choice index>: 2
```

SFT + GRPO step 50 输出：

```text
<think>This diagram shows the Earth's crust, the solid inner core, and the liquid outer core. So, it shows three layers of the Earth.</think>
<choice text>: three
<choice index>: 0
```

### 持续失败：三者都错

- ID：`DQ_010339`；lesson：`protists`
- 问题：Where is the membrane?
- 选项：["E", "U", "X", "P"]
- Gold：`0` — E

| 模型 | 预测 | 正确 | 严格格式 |
|---|---|:---:|:---:|
| 原始 Janus-Pro-7B | 1 — U | ✗ | ✗ |
| ScienceQA SFT | 2 — X | ✗ | ✓ |
| SFT + GRPO step 50 | 2 — X | ✗ | ✓ |

原始 Janus-Pro-7B 输出：

```text
Answer: 1) U. The membrane is labeled as 'U' in the diagram, which is the outer boundary of the cell structure.
```

ScienceQA SFT 输出：

```text
<think>The X is the membrane.</think>
<choice text>: X
<choice index>: 2
```

SFT + GRPO step 50 输出：

```text
<think>The X is the membrane.</think>
<choice text>: X
<choice index>: 2
```

## 解读

1. **SFT 的跨数据集收益明确。** 相比原始模型，SFT 净增加 119 道正确题。
2. **step 50 的总体准确率基本持平，但不是“模型没变”。** 它相对 SFT 改对 126 题、改错 119 题；两者几乎抵消。
3. **回答明显变长。** step 50 的完整回答平均比 SFT 多 3.73 token；需要继续观察更长推理是否真正提高视觉计数/定位，而不是增加冗余或最终答案不一致。
4. **最佳 checkpoint 应继续按全量 val 选择。** 仅凭训练 batch 的 reward 或准确率不能判断泛化；后续每 30 步的验证应重点同时看 `val/accuracy`、`val/strict_format_rate` 和 `val/mean_reasoning_tokens`。

## 可复核输入

- Base summary：`/root/nfs/LiYJ/Janus/outputs/baseline/tqa_val/summary.json`
- SFT summary：`/root/nfs/LiYJ/Janus/outputs/stage1/scienceqa_sft_validation/sft/tqa_val/summary.json`
- Step-50 summary：`/root/nfs/LiYJ/Janus/outputs/stage1/tqa_grpo_lora/validation/checkpoint-000050/tqa_val/summary.json`
- Prompt 数据：`/root/nfs/LiYJ/Janus/data/processed/tqa/val_prompt.jsonl`
