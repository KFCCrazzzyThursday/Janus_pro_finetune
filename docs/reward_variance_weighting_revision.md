# GRPO 奖励方差加权的稳定化修改说明

## 1. 背景与修改动机

论文在每个 prompt 的 `G=16` 个候选回答内计算 Accuracy、Length、Format、
Reasoning 四项奖励的方差，再按 `beta_i * Var(r_i)` 归一化得到动态权重。其核心
思想是：同一项奖励在候选间越分散，模型在该维度越不稳定，也越可能存在可利用的
正负样本对比，因此应该把更多优化强度分配给该项。这个思想与 GRPO 的组内相对学习
是一致的；方差为零时，组内中心化后的该项也无法产生 advantage。

但原公式直接比较不同量纲的原始方差，并把方差本身再次乘入总奖励。2026-08-29 的
8×A100 LoRA-DDP 运行在 97 步时暴露了明显失衡：全程平均实际动态权重约为
Accuracy 91.57%、Length 6.42%、Format 1.26%、Reasoning 0.75%；第 97 步的
Reasoning 权重只有 0.31%。与此同时，外部 judge 的答对条件分数没有明显改善。

这并不否定“优先训练高分散项”的思想，而是说明当前分散度估计受到四类偏差影响：

1. Accuracy 奖励范围为 `[-1, 1]`，其范围宽度是其余 `[0, 1]` 奖励的两倍；未经
   校准的方差因此天然可放大四倍。
2. Reasoning 仅在组内索引正确率超过阈值时启用。未启用组全部为零，低方差既可能
   表示已经掌握，也可能表示尚未得到测量。
3. 为节省 API 成本，每组只实际 judge 一半候选，另一半填入已测样本均值。把这些
   均值填充值纳入方差会系统性压缩 Reasoning 的分散度。
4. 原权重满足 `w_i ∝ beta_i * sigma_i^2`。考虑该分量自身的典型偏差为 `sigma_i`，
   它对组内总奖励方向的典型影响近似为 `beta_i * sigma_i^3`，会非常激进地放大
   高方差分量。高方差可能来自有效的学习边界，也可能来自离散尺度或 judge 噪声。

论文的基础权重还存在记录歧义：公式 3.11 为 `[0.30, 0.20, 0.45, 0.05]`，表 3.1
为 `[0.25, 0.25, 0.45, 0.05]`。本项目延续表格版本，并保留 `paper` 模式用于严格
对照。论文没有报告衰减系数；复现仍采用 `lambda=0.2/3000`，使 Format 先验从
0.45 线性降至 0.25，Reasoning 先验从 0.05 线性升至 0.25。

## 2. 稳定化权重设计

修改后的 `stabilized` 模式保留方差课程思想，但将“任务重要性”和“当前分散度”分开。
首先按每个奖励的理论范围宽度 `Delta_i` 校准方差：

```text
normalized_variance_i = Var(r_i) / Delta_i^2
dispersion_i = sqrt(normalized_variance_i)
```

本项目使用 `Delta=[2, 1, 1, 1]`。使用标准差而不是方差可保留“越分散越加强”的
单调关系，同时避免额外的平方放大。由分散度得到的自适应分布为：

```text
adaptive_i = beta_i * dispersion_i / sum_k(beta_k * dispersion_k)
```

最终权重不再完全由一个小批次的方差决定，而是将计划先验与自适应分布凸组合：

```text
weight_i = (1 - rho) * beta_i + rho * adaptive_i
```

默认 `rho=0.5`，即一半权重表达论文预定的训练阶段，一半表达当前组内的学习边界。
所有分散度为零时自动退回基础 `beta`。该设计仍会加强高分散项，但不会让低分散项
立刻失去全部训练信号。`rho` 可通过 `JANUS_REWARD_VARIANCE_MIX` 配置，并限制在
`[0, 1]`。

Reasoning 的分散度只使用真正经过 judge 或命中 judge cache 的候选。内容哈希抽样
提供可复现的近似随机子集；对至少两个实测分数使用 Bessel 校正的样本方差。未实测
候选仍填入样本均值，以保持现有 reward 接口和 API 成本，但这些填充值不再参与
Reasoning 的权重估计。这样修正的是“课程调度依据”，并不伪造未观测样本间的排序。

## 3. 兼容性、监控与验证标准

通用复现脚本默认保持 `JANUS_REWARD_WEIGHTING=paper`，数值行为与原公式一致。本机
A100 实验启动器默认使用 `stabilized`；设置下列环境变量即可执行论文对照：

```bash
JANUS_REWARD_WEIGHTING=paper bash deploy/local_a100/run_stage1_grpo.sh --smoke
```

稳定化实验为：

```bash
JANUS_REWARD_WEIGHTING=stabilized \
JANUS_REWARD_VARIANCE_MIX=0.5 \
bash deploy/local_a100/run_stage1_grpo.sh --smoke
```

TensorBoard 除原有 raw reward、dynamic weight 和 contribution 外，新增每项计划先验、
用于加权的校准分散度、Reasoning 实际 judge 比例、权重模式标记和 `rho`。这使后续
结果可以回答三个不同问题：原始奖励是否改善、候选是否仍有分散度、动态分配是否符合
预期，而不再只看一个合成 reward。

本修改不预先声称能够提高最终准确率或推理质量。正式结论应来自至少两个使用相同初始
SFT 权重、数据顺序、随机种子和固定验证面板的对照：`paper` 与 `stabilized`。主要判断
指标应是固定题集上的答案准确率和答对条件 Reasoning judge 分数；训练 batch reward
仅作为诊断。若稳定化模式提高 Reasoning 权重却降低固定集准确率，可以调高或调低
`rho`，而不改变奖励定义。对于“低均值、零方差”的一致失败组，单纯提高权重仍不会
产生 GRPO 对比信号，应通过增加探索、改善 dense reward 或单独 SFT 处理，而不是把
低方差误判为已经掌握。

本次实现已通过 42 项单元测试，并在 8×A100-SXM4-40GB 上完成两次完整的一步 LoRA-DDP
smoke：均包含 checkpoint 装载、16 个候选生成、奖励计算、DAPO advantage、反向传播和
AdamW 更新，单步训练约 10.5 秒，峰值显存约 21--23 GiB/卡。第二次将 judge 激活阈值
临时设为零，离线 stub 的实际抽样比例在跨 rank 汇总后为 0.5，验证了“实测/填充”掩码
与全局 reward 顺序一致。stub 的实测分数全部相同，因此 Reasoning 分散度为零、最终权重
保留为一半先验 `0.025`，也符合上述退化行为。smoke 只证明代码和分布式链路可运行，不能
替代使用真实 judge 的长期 `paper`/`stabilized` 收敛对照。
