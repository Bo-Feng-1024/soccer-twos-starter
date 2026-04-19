# 训练实验日志

> 记录每次 PACE 训练的配置、状态和结果。按时间倒序。

## 实验记录

| # | Job ID | 日期 | Git Commit | Ray | 脚本 | 配置摘要 | 状态 | 结果 |
|---|--------|------|-----------|-----|------|---------|------|------|
| 18 | TBD | 2026-04-19 | TBD | 1.4.0 | `train_experiment_d.py` | 分离 policy/value 网络 (vf_share=False), vf_loss=1.0, 从零训练 | 待提交 | — |
| 17 | TBD | 2026-04-19 | TBD | 1.4.0 | `train_experiment_c.py` | Reward 微调（kick 2x, offensive 2x, defensive 0.5x, 去 time penalty），从 checkpoint-5122 | 待提交 | — |
| 16 | 4955780 | 2026-04-18 | `2c48a51` | 1.4.0 | `train_experiment_b.py` | [512,512] 网络 + 近默认超参数，从零训练，50% ceia + 50% selfplay | 完成 | checkpoint-591: **6%**（训练不足，591 轮 vs 需要 5000+） |
| 15 | 4955779 | 2026-04-18 | `2c48a51` | 1.4.0 | `train_experiment_a.py` | 从 checkpoint-5122 续训，entropy=0.01, lr=1e-4, 60% ceia + 40% selfplay | 完成 | checkpoint-5650: **84%**（高 entropy 导致退化） |
| 14 | 4914451 | 2026-04-17 | `ed08910` | 1.4.0 | `train_focused_ceia.py` | 从 checkpoint-5122 续训，混合对手（70% ceia + 30% selfplay），entropy=0.003, lr=5e-5 | 完成 | checkpoint-5750: **86%**（防止退化但未突破 87%） |
| 13 | 4903675 | 2026-04-16 | `460f07f` | 1.4.0 | `train_focused_ceia.py` | 从 checkpoint-5122 续训，100% ceia + 优化超参数 | 完成 | checkpoint-5749: **83%**（退化，过拟合 ceia） |
| 12 | 4903420 | 2026-04-16 | `d1af6cb` | 1.4.0 | `train_bc_finetune.py` | BC 权重注入 + PPO fine-tuning vs 100% ceia, lr=5e-5, entropy=0.01 | 完成 | reward 0.078, 621 iter / 12.5M steps, BC 初始化未能有效对抗 ceia（distribution shift） |
| 11 | 4903211 | 2026-04-16 | `d1af6cb` | 1.4.0 | `collect_expert_data.py` + `train_bc.py` | BC 预训练：从 checkpoint-5050 收集专家数据 + 监督学习 | 完成 | val_acc=75.7%, val_loss=0.579, 12min |
| 10 | 4896565 | 2026-04-16 | `67680dc` | 1.4.0 | `train_focused_ceia.py` | 同 #9 但修复 lr_schedule（从 21M 开始，1e-4 → 1e-5） | 完成 | checkpoint-4650: **80%**, checkpoint-5122: **87%** |
| 9 | 4895541 | 2026-04-16 | `c0524fa` | 1.4.0 | `train_focused_ceia.py` | 100% ceia, 优化超参数, 从 checkpoint-4500 续训, 4 policy 结构 | 已取消 | lr_schedule 从 21M 起为 3e-5，太低，reward 停滞在 0.175-0.188 |
| 8 | 4895469 | 2026-04-15 | `927f7b6` | 1.4.0 | `train_focused_ceia.py` | 同上，4 policy 结构修复 | 完成 1 轮 | stop 条件立即满足（time_total_s=142521 > 41400） |
| 7 | 4895458 | 2026-04-15 | `f00439a` | 1.4.0 | `train_focused_ceia.py` | 100% ceia, 优化超参数, 从 checkpoint-4500 续训, 2 policy 结构 | 失败 | AssertionError: filter keys 不匹配（2 policy vs 4 policy） |
| 6 | 4845199 | 2026-04-14 | `3c31f5f` | 1.4.0 | `train_focused_ceia.py` | 100% ceia, 优化超参数, 从零训练 | 完成 617 轮 | 失败：reward 仅 0.078，从零打满强度 ceia 学不到东西 |
| 5 | 4843426 | 2026-04-14 | `b0ae740` | 1.13.0 | `train_focused_ceia.py` | 同上 | 已取消 | ray 版本不对，改用 1.4.0 |
| 4 | 4843257 | 2026-04-14 | `98a559f` | 1.13.0 | `train_focused_ceia.py` | 同上 | 已取消 | 误判为卡住（实为输出缓冲） |
| 3 | 4843025 | 2026-04-14 | `98a559f` | 1.13.0 | `train_focused_ceia.py` | 同上 | 已取消 | 同上 |
| 2 | 4840700 | 2026-04-14 | `3fef6d3` | 1.13.0 | `train_focused_ceia.py` | 100% ceia, 优化超参数, 无 ceia checkpoint | 完成 157 轮 | ceia 权重未加载，reward 仅 0.076 |
| 1 | 4840691 | 2026-04-14 | `6b0736a` | 1.13.0 | `train_focused_ceia.py` | 同上 | 失败 | `np.bool` 报错，需 numpy==1.23.5 |

## 踩坑记录

| 问题 | 原因 | 解决 | 对应 Job |
|------|------|------|---------|
| `np.bool` AttributeError | numpy 版本太新 | `pip install "numpy==1.23.5"` | #1 |
| ceia 权重未加载 | `ceia_baseline_agent.zip` 未解压到 PACE | 从 Google Drive 下载解压 | #2 |
| 训练看似卡住 | Python stdout 缓冲，`tail` 看不到输出 | batch 脚本加 `PYTHONUNBUFFERED=1` + `python -u` | #3, #4 |
| ray 1.13.0 评估不兼容 | `BaseEnv()` 和 checkpoint 格式不兼容 | 改用 ray==1.4.0（Frank 的 environment.yml） | #5 |
| `conda env create -f` 失败 | pip 版本太新，gym metadata 无效 | 手动建环境 + 降级 pip + `--no-deps` 安装 | #6 环境搭建 |
| 从零训练 100% ceia 无效 | agent 太弱打不过 ceia，shaped reward 太小学不到东西 | 从 Frank 的 checkpoint-4500 续训 | #6 |
| restore 后 policy 数不匹配 | checkpoint 有 4 个 policy，脚本只定义 2 个 | 保持 4 policy 结构，全部加载 ceia 权重 | #7 |
| restore 后 stop 条件立即满足 | checkpoint 的 time_total_s=142521 已超 41400 | 增大 stop 到 184200s / 60M steps | #8 |
| lr_schedule 基于绝对 timesteps | checkpoint 恢复后 timesteps=21M，超过 schedule 最后节点 15M，lr 锁在 3e-5 | schedule 从 21M 开始设置 | #9 |

## 如何添加新记录

训练提交后，添加一行到实验记录表：

```bash
# 获取当前 git commit
git rev-parse --short HEAD

# 获取 job ID
squeue -u $USER
```
