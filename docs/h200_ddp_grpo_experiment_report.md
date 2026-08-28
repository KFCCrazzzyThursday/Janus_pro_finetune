# Janus-Pro-7B TQA GRPO：H200 DDP 部署与加速实验报告

- 实验日期：2026-08-28（UTC）
- 代码版本：`328b32c`
- 实验目录：`/workspace/Janus_pro_finetune`
- 报告范围：验证远程 4 卡 DDP GRPO 的正确启动、定位吞吐瓶颈，并筛选不改变训练语义的加速项；不评价最终收敛质量。

## 1. 结论摘要

实验已在 4 张 NVIDIA H200 上完成并恢复正式训练。当前主要瓶颈不是显存、CPU、磁盘或 DDP 通信，而是每轮生成后的外部 LLM judge 请求及动态重采样；judge 阶段可观察到 GPU 空闲和活跃 HTTPS 连接。

已上线的有效优化是把每个 rank 的 judge 并发从 8 提至 16。相同 seed、相同首批 completions 的受控实验中，单步从 124.01 s 降至 108.85 s，提升 12.2%，160 次 judge 调用零错误。当前正式任务前 5 步的 `step_time` 均值为 103.09 s；旧配置前 5 步均值为 124.58 s。后一个跨多步比较会受生成长度和动态重采样波动影响，因此以 12.2% 的受控结果作为主要证据。

增大 micro-batch 最多只带来约 3% 的 GPU-only 加速，同时改变 DAPO 梯度缩放和梯度范数；关闭 gradient checkpointing 在排除 GaLore 首步后没有加速，却增加约 8 GiB/卡显存。因此正式配置保持 batch=2、gradient checkpointing 开启，仅应用 judge 并发 16。

## 2. 环境、配置与测量方法

| 项目 | 配置 |
|---|---|
| GPU | 4 × NVIDIA H200，143771 MiB/卡，NVLink |
| Driver | 580.159.03 |
| 模型 | `models/Janus-Pro-7B-stage1-sft` |
| 数据 | `data/processed/tqa/train_prompt_model_difficulty.remote.jsonl` |
| 并行 | PyTorch DDP，world size=4，BF16，SDPA |
| 训练 | LLM 全参训练；冻结 ViT、aligner 及生成视觉模块 |
| 优化器 | GaLore，rank=512，projection gap=128，LR=1e-6 |
| GRPO | DAPO loss，16 generations，max completion=384，动态采样 |
| Batch | per-device=2，gradient accumulation=16，有效 batch=128 |
| 生成调度 | generation batch=128，steps/generation=16，rollout chunk=16 |
| 其他 | gradient checkpointing=true，entropy logging=true，vLLM=false |

速度以训练日志中的 `step_time` 为主；它不包含进程启动和首次 GaLore 投影初始化。需要比较真实墙钟时，用累计 `train_speed(s/it)` 反推每一步耗时。所有对照固定 seed/data seed=42；GPU-only 实验用等形状 stub judge 去除外部 API 延迟。单步 sweep 只作为方向性证据，checkpointing 则补做了开/关各 4 步，并排除首步。

## 3. 瓶颈定位与 judge 并发实验

### 3.1 基线拆解

旧正式任务（judge 并发 8）前 5 步为 124.01、126.06、79.18、170.25、123.38 s，均值 124.58 s。相同首批数据的 GPU-only stub 为 71.36 s，说明真实 judge 为首步增加约 52.66 s，占该步总时长约 42.5%。训练/生成阶段 4 卡 GPU 利用率接近 100%；judge 阶段 GPU 空闲，因此继续扩 batch 或优化 NCCL 不能解决主要等待。

### 3.2 并发 8 → 16

| 条件 | 首步 `step_time` | Judge 调用 | 错误 | 相对变化 |
|---|---:|---:|---:|---:|
| 并发 8 | 124.01 s | 同批次 | 0 | 基线 |
| 并发 16 | 108.85 s | 160 | 0 | -15.16 s / -12.2% |
| GPU-only stub | 71.36 s | 0 | — | judge 下界参考 |

优化后的正式任务前 5 步为 105.05、115.13、59.65、115.57、120.06 s，均值 103.09 s。按目前 103–109 s/步粗略外推，3000 步纯训练墙钟约 86–91 小时，另需计入每 500 步保存、API 抖动和重试。Trainer 早期 ETA 会把约 242 s 的一次性启动/GaLore 开销均摊到每步，前几步显示的 5–12 天不能作为总时长预测。

## 4. Batch、显存及其他加速项

### 4.1 Micro-batch sweep（GPU-only、checkpointing 开启）

保持有效 batch=128，分别令 `(per-device batch, GA)` 为 `(2,16)`、`(4,8)`、`(8,4)`、`(16,2)`、`(32,1)`：

| Batch / GA | `step_time` | 相对 B2 | 记录显存 | Grad norm |
|---|---:|---:|---:|---:|
| 2 / 16 | 71.36 s | 基线 | 63.28 GiB | 0.107 |
| 4 / 8 | 70.11 s | -1.7% | 63.20 GiB | 0.293 |
| 8 / 4 | 69.52 s | -2.6% | 63.22 GiB | 0.824 |
| 16 / 2 | 69.19 s | -3.0% | 63.10 GiB | 2.547 |
| 32 / 1 | 72.57 s | +1.7% | 63.10 GiB | 4.969 |

显存峰值由生成/rollout 等阶段主导，因此表中 batch 增大后记录显存没有同步增长。更重要的是，loss 随 micro-batch 近似线性放大，grad norm 从 0.107 增至 4.969；B16/B32 已触发 `max_grad_norm=1` 裁剪。这来自 DAPO 全局 token normalizer 与 Trainer gradient-accumulation 缩放的组合，说明 sweep 并非严格等价配方。为最多 3% 加速改变优化轨迹不划算，正式任务保持 B2/GA16。

关闭 checkpointing 后，B8 单步记录 99.24 GiB、`nvidia-smi` 观察峰值约 115.2 GiB；B16 在 rank 1 达到约 139.7/139.8 GiB 后 OOM。因此“空闲显存很多”只适用于部分阶段，不能据此直接把正式 batch 提到 16。

### 4.2 Gradient checkpointing 多步对照

下表为真实墙钟；首步包含相同的 GaLore 初始化，仅第 2–4 步用于稳态判断。

| 设置 | Step 1 | Step 2 | Step 3 | Step 4 | 最终记录显存 |
|---|---:|---:|---:|---:|---:|
| 开启 | 313.1 s | 65.4 s | 49.7 s | 83.9 s | 64.84 GiB |
| 关闭 | 312.0 s | 65.6 s | 50.2 s | 64.9 s | 72.97 GiB |

第 2 步工作量完全一致，第 3 步近似一致：关闭后均未加速，两步平均反而慢约 0.6%。第 4 步两组的 dynamic-group keep fraction 分别为 0.375 和 0.6875，开启组触发更多重采样，不能把 19 s 差值归因于 checkpointing。结论是保留 checkpointing。

### 4.3 其余尝试

| 尝试 | 结果 | 决策 |
|---|---|---|
| Rollout chunk 16 → 32 | 单步超过 4 分钟仍在计算，已中止 | 保持 16 |
| 关闭 entropy logging | 71.48 s vs 71.36 s | 保持开启 |
| Max completion 384 → 256 | B16 单步 69.19 → 66.96 s，但 completion/reward/重采样均改变 | 不应用 |
| vLLM | 环境未安装，Janus/Janus-Pro 尚无可直接使用的官方支持 | 不作为短期优化 |
| Judge 结果缓存 | 历史调用约 10.5% 重复 | 已实现内容寻址的 SQLite WAL 共享缓存 |

vLLM 兼容性参考：官方 Janus-Pro 请求 [#12479](https://github.com/vllm-project/vllm/issues/12479) 及关联 Janus 请求 [#12538](https://github.com/vllm-project/vllm/issues/12538)。

### 4.4 Judge 成本优化与流水线实验

后续实验实现了以下可配置优化：

- 四个 DDP rank 共享的 SQLite WAL 精确缓存，键包含模型、prompt 版本、题目、参考答案和完整 completion；
- 同题候选批量评分，正式配置每个请求合并 4 个候选；
- 紧凑 JSON 协议及 2048 字符 reasoning 上限；
- 确定性 50% 候选抽样，未抽样候选使用同组已评分候选均值；
- reasoning judge 激活阈值由 60% 调至 62.5%；
- 在调用外部 Judge 前完成 answer-homogeneous 动态重采样过滤，避免为必然丢弃的组付费。

真实 API 小测中，同一题 4 个候选使用 4 个独立并发请求耗时 8.73 s，合并为单请求耗时 6.00 s，评分一致；第二次精确缓存命中耗时约 1 ms。历史有效批次回放显示，仅阈值与 50% 抽样即可使外评候选数下降约 55%，尚未计入缓存与批量请求节省。首批正式对照中，新外评约为 24 个候选、6 个批量请求，另有 8 个缓存命中；旧路径约为 160 个独立候选请求。

另实现了实验性 Judge/GPU 流水线：上一动态采样轮的有效组通过独立 Gloo 进程组后台评分，同时 GPU 生成下一轮，以避免与默认 NCCL collective 交叉。该路径能够正确运行，但两步实测为 109.0 s、127.8 s，未优于非流水线基线 103.4 s 及旧优化配置第二步 115.1 s。批量化和抽样后剩余 Judge 延迟不足以抵消额外同步开销，因此代码保留、默认关闭（`JANUS_JUDGE_PIPELINE=0`）。

## 5. 最终配置、运行状态与证据

最终决策：4 卡 DDP；B2/GA16；generation batch=128；rollout chunk=16；max completion=384；checkpointing 与 entropy 开启；judge 并发 16、batch size=4、抽样率 50%、激活阈值 62.5%；共享精确缓存和动态预过滤开启；Judge/GPU 流水线关闭；不启用 vLLM。抽样、阈值和组均值估计会改变 reasoning reward 的采样语义，因此通过环境变量完整保留了回退能力。

正式任务由 Supervisor 管理，服务名为 `janus-grpo`，报告编写时状态为 `RUNNING`。输出目录为：

```text
/workspace/Janus_pro_finetune/outputs/stage1/tqa_grpo_h200_ddp
```

关键证据目录：

```text
outputs/stage1/tqa_grpo_h200_ddp_pre_tune_step5_20260828T0622Z  # 并发 8 基线
outputs/stage1/tqa_grpo_h200_ddp_judge16_probe_step1_20260828   # 并发 16 受控探针
outputs/smoke/grpo_full_shape_h200_ddp                           # GPU-only stub
outputs/bench/b2_r16_gc_4step                                   # checkpointing 开启
outputs/bench/b2_r16_nogc_4step                                 # checkpointing 关闭
```

常用检查命令：

```bash
supervisorctl status janus-grpo
tail -f /var/log/portal/janus-grpo.log
tail -1 outputs/stage1/tqa_grpo_h200_ddp/logging.jsonl | jq
nvidia-smi
```

后续若仍需显著加速，优先方向是把 judge 改造成低延迟批处理服务或部署到独立推理节点；其次才是实现 Janus 专用 vLLM 适配。继续增大训练 batch、关闭 checkpointing 或缩短 completion 都没有得到“快且语义等价”的证据。
