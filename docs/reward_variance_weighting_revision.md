# GRPO 奖励权重稳定化实现

## 动机

论文在同一 prompt 的 `G=16` 个候选内计算 Accuracy、Length、Format、
Reasoning 四项奖励的方差，并按 `beta_i * Var(r_i)` 归一化。高分散项因此
获得更多权重，符合 GRPO 优先利用组内正负差异的思想，但原公式有三项偏差：

1. Accuracy 的理论范围是 `[-1, 1]`，其余奖励为 `[0, 1]`，原始方差不能直接比较；
2. 方差会使分量的典型影响近似按 `sigma^3` 放大；
3. Reasoning 每组只实际 judge 8/16 条，其余候选填入实测均值，若把填充值
   计入方差会系统性压低 Reasoning 分散度。

## 稳定化公式

设奖励理论范围宽度为：

```text
Delta = [2, 1, 1, 1]
```

先计算范围归一化后的标准差：

```text
dispersion_i = std(r_i / Delta_i)
adaptive_i = beta_i * dispersion_i / sum_k(beta_k * dispersion_k)
weight_i = (1 - rho) * normalized_beta_i + rho * adaptive_i
```

本机正式训练固定使用 `rho=0.5`。因此一半权重来自计划先验，一半来自当前
组内分散度；即使某一项当前方差为零，也不会立即失去全部训练信号。所有项
分散度均为零时，权重自动退回计划先验。

Reasoning 分散度只使用真正经过 judge 或命中 cache 的候选。若实测数量小于
`G` 且至少为 2，使用 Bessel 校正的样本方差（除以 `n-1`）；均值填充候选不
参与权重估计。完整测量的 Accuracy、Length、Format 仍使用组内总体方差。

## 困难组处理

全对组视为已掌握并重采样。混合正确组保留 Accuracy 对比；全错组也保留，
让 Reasoning、Length 和 Format 提供 dense relative signal。默认 judge 所有
未掌握组，而不是仅 judge 正确率高于阈值的组。

低均值、低方差的全错组仍可能缺少有效对比；保留它们并不保证产生梯度。
这类组应通过更高探索度、更密集的奖励或额外 SFT 处理，而不能只凭零方差
判断为已经掌握。

## 对照模式与启动默认值

插件保留两种显式模式：

- `JANUS_REWARD_WEIGHTING=paper`：复现 `beta * raw_variance`；
- `JANUS_REWARD_WEIGHTING=stabilized`：使用上述校正公式。

`scripts/run_stage1_grpo.sh` 明确默认 `stabilized`，并通过
`JANUS_REWARD_VARIANCE_MIX` 配置 `rho`。`paper` 模式只用于同初始权重、
数据顺序和随机种子的消融实验。

## TensorBoard 指标

保留论文曲线 `paper/*_reward_mean`、`paper/*_reward_variance` 和总体奖励曲线，
并额外记录：

- 每项计划先验、实际动态权重及其标准差；
- 每项 observed reward variance、normalized std 和实际 weighting dispersion；
- Reasoning 实际 judge 比例；
- 全错、混合、全对组比例及保留比例；
- weighting 模式、`rho`、KL 系数、advantage 保留率、entropy 和 clipping。

论文曲线中的 component variance 保持对全部 `G` 个 reward 槽位计算的总体
方差；用于权重计算的校正分散度单独记录，避免把两个统计口径混为一谈。

## 验证

合并后全套 42 项单元测试通过。正式结论仍需使用固定验证集对比 `paper` 与
`stabilized`，训练 batch reward 只作为诊断，不能单独证明准确率或推理质量改善。
