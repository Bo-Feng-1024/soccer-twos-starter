# 训练实验日志

> 记录每次 PACE 训练的配置、状态和结果。按时间倒序。

## 实验记录

| # | Job ID | 日期 | Git Commit | Ray | 脚本 | 配置摘要 | 状态 | 结果 |
|---|--------|------|-----------|-----|------|---------|------|------|
| 18 | 5015762 | 2026-04-19 | `2c4c330` | 1.4.0 | `train_experiment_d.py` | 分离 policy/value 网络 (vf_share=False), vf_loss=1.0, 从零训练 | 完成 | checkpoint-555: 训练不足（555 轮），未评估 |
| 23 | 5030847 | 2026-04-21 | `66a53e2` | 1.4.0 | `train_experiment_h.py` | 强防守 reward（3x defensive penalty + goalie positioning），从 checkpoint-5122 | 完成（超时） | iter 5754, reward 0.098, 46.5M steps, 12h 时间限制到期，待查 checkpoint 并评估 |
| — | anal | 2026-04-21 | — | — | analyze_losses.py | 输球模式分析：输球 avg 24.5 步 vs 赢球 41.0 步，被快速进球 | 完成 | 防守漏洞是主要输球原因 |
| 22 | 5022630 | 2026-04-20 | `4f9e20e` | 1.4.0 | `train_experiment_g.py` | 默认超参数 + 从 checkpoint-5122 续训，grad_clip=0.5 | 完成 | checkpoint-6600: **75%**（严重退化，blue 88% vs orange 62% 不对称） |
| 25 | 5031679 | 2026-04-21 | `a72ab3d` | 1.4.0 | `train_experiment_d.py` | 续训实验 D（分离网络），从 checkpoint-1122，排除问题节点 | 运行中 | — |
| 24 | 5031584 | 2026-04-21 | `a72ab3d` | 1.4.0 | `train_experiment_d.py` | 续训实验 D（分离网络），从 checkpoint-1122 继续 | 失败 | 端口占用（Address already in use），walltime 仅 2min |
| 21 | 5022518 | 2026-04-19 | `4f40a3a` | 1.4.0 | `train_experiment_d.py` | 续训实验 D（分离网络 vf_share=False），从 checkpoint-555 继续 | 完成 | checkpoint-1122（新目录 29b63），reward 0.15-0.16，待评估 |
| 20 | 5022442 | 2026-04-19 | `1302de0` | 1.4.0 | `train_experiment_f.py` | MeanStdFilter + 从零训练，50% ceia + 50% selfplay | 完成 | checkpoint-593, reward 0.075（MeanStdFilter 从零训练太慢） |
| 19 | 5022441 | 2026-04-19 | `1302de0` | 1.4.0 | `train_experiment_e.py` | MeanStdFilter + 从 checkpoint-5122 续训，70% ceia | 失败 | NoFilter/MeanStdFilter 不兼容，restore 时 filter sync 报错 |
| — | eval | 2026-04-21 | — | — | 500 局评估 checkpoint-5050 | 扫描中间 checkpoint | 完成 | **81.4%**（407/500） |
| — | eval | 2026-04-21 | — | — | 100 局评估 checkpoint-5050 | 扫描中间 checkpoint | 完成 | 87%（87/100） |
| — | eval | 2026-04-21 | — | — | 100 局评估 checkpoint-5000 | 扫描中间 checkpoint | 完成 | 77%（77/100） |
| — | eval | 2026-04-19 | — | — | 500 局评估 checkpoint-5122 | 精确测量真实胜率 | 完成 | **83%**（415/500），之前 100 局测的 87% 有统计偏差 |
| 17 | 5015761 | 2026-04-19 | `2c4c330` | 1.4.0 | `train_experiment_c.py` | Reward 微调（kick 2x, offensive 2x, defensive 0.5x, 去 time penalty），从 checkpoint-5122 | 完成 | checkpoint-5200: **81%**, checkpoint-5750: **84%**（reward 变化导致退化） |
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
| 端口占用反复出现（4次） | 失败 job 或交互式 session 的 Unity 进程残留在节点上 | **每次 sbatch 必须加 `--exclude`**，排除所有正在用和最近失败的节点 | #23(x2), #24 |

## 如何添加新记录

训练提交后，添加一行到实验记录表：

```bash
# 获取当前 git commit
git rev-parse --short HEAD

# 获取 job ID
squeue -u $USER
```
