# A100 训练续传与代码备份说明（2026-09-01）

## 目标与位置

本次备份计划在北京时间 **2026-09-01 10:00**（UTC 02:00）自动执行。GitHub
目标分支固定为：

`backup/a100-resume-20260901-1000-cst`

该分支从到点时仓库的已提交 `HEAD` 创建，并由脚本在推送后反查远端 commit，避免只在本地
建分支而未真正上传。Hugging Face 目标仓库通过 `JANUS_HF_REPO_ID` 指定，默认创建或使用
model repo；本次使用 `Billyshears/Janus_pro_finetune` 的
`backup/a100-resume-20260901-1000-cst` revision。HF token 只从运行时凭据文件读取，不写入
Git、日志或备份内容。该 HF 仓库目前是公开仓库，因此此 revision 上传完成后也可公开访问；
若需私有备份，应在执行前更换目标 private repo。

## 本阶段代码和训练策略变更

训练从 L40S 上的完整 step 270 状态迁移到两张 A100，保留了 LoRA 权重、AdamW optimizer、
constant scheduler、trainer state 和 RNG 状态。当前阶段只保留两项 reward：

- `janus_accuracy`：答案索引正确性；
- `janus_format`：严格输出格式。

模型仍采用语言模型 LoRA（rank 16、alpha 32），视觉塔和 aligner 冻结。训练入口使用
`loss_type=dapo`、16 个 generation、temperature 1.0、KL beta 0.04，并维持每 30 个 optimizer
step 保存一次完整 checkpoint、释放训练进程后跑一次 2,781 题确定性验证。运行目录保留最新两个
完整 checkpoint，另外独立保留当前验证准确率最佳 checkpoint。

奖励调度已在 commit `508f647` 后改为可持久化的线性方案：step 391–480 内，格式权重从
0.75 线性下降到 0.10，正确性权重互补地从 0.25 上升到 0.90；step 480 后保持最终权重。
旧的 step 331–390 余弦配置保存在运行目录 `schedule_history/`，TensorBoard 不重写历史点，因而
策略切换会以真实时间顺序显示。

## 自动备份包含什么

到点时脚本不会中断正在运行的训练，而是先刷新 `resume_state.json`，选择最新**完整且已通过
manifest 校验**的 checkpoint。对不可变 checkpoint 使用硬链接建立独立快照，因此训练进程随后
清理旧 checkpoint 时，备份仍然有效，同时不会瞬间复制大量磁盘数据。快照至少包括：

1. 最新完整 checkpoint：LoRA、optimizer、scheduler、trainer state、training args、两张 A100
   的 Python/NumPy/CPU/CUDA RNG 状态；
2. 上一个完整 checkpoint，作为最新状态损坏时的回退点；
3. 当前验证准确率最佳 checkpoint；若与前两者是同一步，只保存一份并在 roles 中复用；
4. `resume_state.json`、`best.json`、reward schedule 及旧调度历史；
5. 最新/上一个/最佳 step 的验证结果、全量验证 history、TensorBoard event、训练 JSONL 日志、
   completions、资源监控和 launcher 日志；
6. 远端实际使用的处理后 train/val JSONL（原始图片和 SFT 基座不重复上传，路径与依赖记录在
   manifest 中）；
7. Git commit、当前分支、DeepSeek-Janus 与 ms-swift 上游 commit、Python 包版本、GPU/驱动信息；
8. `backup_manifest.json`：所有上传文件的相对路径、字节数和 SHA-256；以及逐 checkpoint 的
   `janus_checkpoint_manifest.json`。

Hugging Face 使用可断点续传的 `upload_large_folder`。上传完成后脚本重新列出远端文件，确认
adapter、optimizer、scheduler、trainer state 和所有 rank RNG 文件均存在，才写入成功回执。
上传中断时，本地一致性快照和 HF 上传缓存都会保留；定时包装脚本默认每 10 分钟重试一次、
最多尝试 6 次，重跑时复用同一个快照和 revision，不重复占用 checkpoint 空间。

## 恢复方式与限制

下载 HF 仓库中 `backups/a100-resume-20260901-1000-cst/`，并检出上述 GitHub 分支。恢复前先
依据两个 manifest 校验哈希，再把选定 checkpoint 放回输出目录，提供相同的 ScienceQA SFT
基座与 TQA 图片数据，然后通过 managed launcher 指定 `JANUS_RESUME_FROM_CHECKPOINT`。

该 checkpoint 对 **world size 2** 可以恢复 optimizer、scheduler 和已保存 RNG 状态。若切回
五张 L40S，模型和 optimizer 仍可继续，但新增 rank 的 CUDA RNG 无法从两张 A100 的状态精确
还原，必须进行确定性重新播种；因此可续训，但随机轨迹不会逐 bit 等同于两卡继续训练。

自动任务的主日志位于：

`outputs/stage1/tqa_grpo_accfmt_a100_from270_managed30/scheduled_backups/logs/`

机器可读状态位于同目录上一级的
`a100-resume-20260901-1000-cst.status.json`。其中分别记录 GitHub、快照和 Hugging Face 三个
阶段的成功或失败，便于只重试失败部分。
