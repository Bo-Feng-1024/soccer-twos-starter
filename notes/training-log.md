# Training Experiment Log

> A record of every PACE training run: configuration, status, and result. Reverse chronological order.

## Experiment Records

| # | Job ID | Date | Git Commit | Ray | Script | Configuration Summary | Status | Result |
|---|--------|------|-----------|-----|--------|-----------------------|--------|--------|
| 28 | 5038579 | 2026-04-22 | `70be782` | 1.4.0 | `train_experiment_d.py` | Experiment D v3: separated networks + 70% self-play + lr=3e-4, resumed from checkpoint-1700 | Done (timed out) | checkpoint-2300: **90%** (100 games). Best to date — separated networks were the key. |
| 29 | 5038807 | 2026-04-22 | `dc04f67` | 1.4.0 | `train_experiment_j.py` | 4-frame stack + weight injection (fixed paths + ceia shape adapter), 70% self-play | Done | checkpoint-587: reward 0.079, severely undertrained (587 iter), discarded. |
| 27 | 5037897 | 2026-04-22 | `ee63f03` | 1.4.0 | `train_experiment_j.py` | 4-frame stack + weight injection + 70% self-play | Failed | Weight injection failed (relative path) + ceia shape mismatch — effectively training from scratch, cancelled. |
| 26 | 5037675 | 2026-04-22 | `e4f9286` | 1.4.0 | `train_experiment_i.py` | Self-play dominant (30% ceia + 70% selfplay), lr=3e-4, 5 epochs, resumed from checkpoint-5122 (following Brandão 2022) | Done (timed out) | checkpoint-5750: **85%** (100 games), no breakthrough. |
| — | eval | 2026-04-23 | — | — | eval_vs_ceia.py | Evaluate exp I checkpoint-5750 vs ceia | Done | **85%** (85/100) |
| — | eval | 2026-04-23 | — | — | eval_vs_ceia.py | Evaluate exp D v3 checkpoint-2300 vs ceia | Done | **90%** (90/100) |
| 18 | 5015762 | 2026-04-19 | `2c4c330` | 1.4.0 | `train_experiment_d.py` | Separated policy/value networks (vf_share=False), vf_loss=1.0, from scratch | Done | checkpoint-555: undertrained (555 iter), not evaluated. |
| 23 | 5030847 | 2026-04-21 | `66a53e2` | 1.4.0 | `train_experiment_h.py` | Strong defensive reward (3x defensive penalty + goalie positioning), resumed from checkpoint-5122 | Done (timed out) | checkpoint-5750: **80%** (regressed — defensive reward too conservative). |
| — | anal | 2026-04-21 | — | — | analyze_losses.py | Loss-pattern analysis: avg 24.5 steps when conceding vs 41.0 steps when scoring — fast counter-goals | Done | Defensive gaps are the dominant failure mode. |
| 22 | 5022630 | 2026-04-20 | `4f9e20e` | 1.4.0 | `train_experiment_g.py` | Default hyperparameters + resumed from checkpoint-5122, grad_clip=0.5 | Done | checkpoint-6600: **75%** (severe regression, asymmetric: blue 88% vs orange 62%). |
| 25 | 5031679 | 2026-04-21 | `a72ab3d` | 1.4.0 | `train_experiment_d.py` | Resume experiment D (separated networks) from checkpoint-1122, excluding the problematic node | Done (timed out) | checkpoint-1700: **78%** (undertrained, asymmetric: blue 82% vs orange 74%). |
| 24 | 5031584 | 2026-04-21 | `a72ab3d` | 1.4.0 | `train_experiment_d.py` | Resume experiment D (separated networks) from checkpoint-1122 | Failed | Port already in use (Address already in use), walltime only 2 min. |
| 21 | 5022518 | 2026-04-19 | `4f40a3a` | 1.4.0 | `train_experiment_d.py` | Resume experiment D (separated networks, vf_share=False) from checkpoint-555 | Done | checkpoint-1122 (new dir 29b63), reward 0.15-0.16, pending evaluation. |
| 20 | 5022442 | 2026-04-19 | `1302de0` | 1.4.0 | `train_experiment_f.py` | MeanStdFilter + from scratch, 50% ceia + 50% selfplay | Done | checkpoint-593, reward 0.075 (MeanStdFilter from scratch is too slow). |
| 19 | 5022441 | 2026-04-19 | `1302de0` | 1.4.0 | `train_experiment_e.py` | MeanStdFilter + resumed from checkpoint-5122, 70% ceia | Failed | NoFilter / MeanStdFilter incompatible — filter sync error during restore. |
| — | eval | 2026-04-21 | — | — | 500-game evaluation of checkpoint-5050 | Sweeping intermediate checkpoints | Done | **81.4%** (407/500) |
| — | eval | 2026-04-21 | — | — | 100-game evaluation of checkpoint-5050 | Sweeping intermediate checkpoints | Done | 87% (87/100) |
| — | eval | 2026-04-21 | — | — | 100-game evaluation of checkpoint-5000 | Sweeping intermediate checkpoints | Done | 77% (77/100) |
| — | eval | 2026-04-19 | — | — | 500-game evaluation of checkpoint-5122 | Precise win-rate measurement | Done | **83%** (415/500); the earlier 100-game 87% had statistical bias. |
| 17 | 5015761 | 2026-04-19 | `2c4c330` | 1.4.0 | `train_experiment_c.py` | Reward tuning (kick 2x, offensive 2x, defensive 0.5x, drop time penalty), resumed from checkpoint-5122 | Done | checkpoint-5200: **81%**, checkpoint-5750: **84%** (reward changes caused regression). |
| 16 | 4955780 | 2026-04-18 | `2c48a51` | 1.4.0 | `train_experiment_b.py` | [512,512] network + near-default hyperparameters, from scratch, 50% ceia + 50% selfplay | Done | checkpoint-591: **6%** (severely undertrained — 591 iter vs the 5000+ needed). |
| 15 | 4955779 | 2026-04-18 | `2c48a51` | 1.4.0 | `train_experiment_a.py` | Resumed from checkpoint-5122, entropy=0.01, lr=1e-4, 60% ceia + 40% selfplay | Done | checkpoint-5650: **84%** (high entropy caused regression). |
| 14 | 4914451 | 2026-04-17 | `ed08910` | 1.4.0 | `train_focused_ceia.py` | Resumed from checkpoint-5122, mixed opponents (70% ceia + 30% selfplay), entropy=0.003, lr=5e-5 | Done | checkpoint-5750: **86%** (prevented regression but did not break 87%). |
| 13 | 4903675 | 2026-04-16 | `460f07f` | 1.4.0 | `train_focused_ceia.py` | Resumed from checkpoint-5122, 100% ceia + tuned hyperparameters | Done | checkpoint-5749: **83%** (regressed — overfit to ceia). |
| 12 | 4903420 | 2026-04-16 | `d1af6cb` | 1.4.0 | `train_bc_finetune.py` | BC weight injection + PPO fine-tune vs 100% ceia, lr=5e-5, entropy=0.01 | Done | reward 0.078, 621 iter / 12.5M steps; BC initialisation could not effectively counter ceia (distribution shift). |
| 11 | 4903211 | 2026-04-16 | `d1af6cb` | 1.4.0 | `collect_expert_data.py` + `train_bc.py` | BC pretraining: collect expert data from checkpoint-5050 + supervised learning | Done | val_acc=75.7%, val_loss=0.579, 12 min. |
| 10 | 4896565 | 2026-04-16 | `67680dc` | 1.4.0 | `train_focused_ceia.py` | Same as #9 but fixed lr_schedule (starts from 21M steps, 1e-4 → 1e-5) | Done | checkpoint-4650: **80%**, checkpoint-5122: **87%**. |
| 9 | 4895541 | 2026-04-16 | `c0524fa` | 1.4.0 | `train_focused_ceia.py` | 100% ceia, tuned hyperparameters, resumed from checkpoint-4500, 4-policy structure | Cancelled | lr_schedule from 21M onwards was 3e-5 — too low; reward stalled at 0.175-0.188. |
| 8 | 4895469 | 2026-04-15 | `927f7b6` | 1.4.0 | `train_focused_ceia.py` | Same as above, fixed 4-policy structure | Done 1 iter | Stop condition met immediately (time_total_s=142521 > 41400). |
| 7 | 4895458 | 2026-04-15 | `f00439a` | 1.4.0 | `train_focused_ceia.py` | 100% ceia, tuned hyperparameters, resumed from checkpoint-4500, 2-policy structure | Failed | AssertionError: filter keys mismatch (2-policy vs 4-policy). |
| 6 | 4845199 | 2026-04-14 | `3c31f5f` | 1.4.0 | `train_focused_ceia.py` | 100% ceia, tuned hyperparameters, from scratch | Done 617 iter | Failed: reward only 0.078 — full-strength ceia is not learnable from scratch. |
| 5 | 4843426 | 2026-04-14 | `b0ae740` | 1.13.0 | `train_focused_ceia.py` | Same as above | Cancelled | Wrong Ray version, switched to 1.4.0. |
| 4 | 4843257 | 2026-04-14 | `98a559f` | 1.13.0 | `train_focused_ceia.py` | Same as above | Cancelled | Misjudged as stuck (was actually output buffering). |
| 3 | 4843025 | 2026-04-14 | `98a559f` | 1.13.0 | `train_focused_ceia.py` | Same as above | Cancelled | Same as above. |
| 2 | 4840700 | 2026-04-14 | `3fef6d3` | 1.13.0 | `train_focused_ceia.py` | 100% ceia, tuned hyperparameters, no ceia checkpoint loaded | Done 157 iter | ceia weights not loaded, reward only 0.076. |
| 1 | 4840691 | 2026-04-14 | `6b0736a` | 1.13.0 | `train_focused_ceia.py` | Same as above | Failed | `np.bool` error, requires numpy==1.23.5. |

## Pitfalls Encountered

| Issue | Cause | Fix | Job # |
|-------|-------|-----|-------|
| `np.bool` AttributeError | numpy version too new | `pip install "numpy==1.23.5"` | #1 |
| ceia weights not loaded | `ceia_baseline_agent.zip` not extracted on PACE | Download and extract from Google Drive | #2 |
| Training appears stuck | Python stdout buffering — `tail` cannot see output | Add `PYTHONUNBUFFERED=1` + `python -u` to the batch script | #3, #4 |
| Ray 1.13.0 incompatible with evaluation | `BaseEnv()` and checkpoint format mismatch | Switch to ray==1.4.0 (Frank's environment.yml) | #5 |
| `conda env create -f` fails | pip too new, gym metadata invalid | Build env manually + downgrade pip + install with `--no-deps` | #6 environment setup |
| 100% ceia from scratch does not learn | Agent too weak to beat ceia; shaped reward too small to gain traction | Resume from Frank's checkpoint-4500 | #6 |
| Policy count mismatch on restore | Checkpoint has 4 policies, script defined only 2 | Keep the 4-policy structure and load ceia weights into all of them | #7 |
| Stop condition met immediately on restore | Checkpoint's time_total_s=142521 already exceeds the 41400 budget | Increase stop budget to 184200s / 60M steps | #8 |
| lr_schedule based on absolute timesteps | After restore timesteps=21M exceeds the schedule's last anchor at 15M, locking lr at 3e-5 | Re-anchor schedule to start from 21M | #9 |
| Recurring port-in-use failures (×4) | Leftover Unity processes from failed jobs / interactive sessions on the node | **Always pass `--exclude` to sbatch**, listing all nodes currently busy or recently failed | #23 (×2), #24 |

## How to Add a New Record

After submitting a training run, append a new row to the experiment record table:

```bash
# Get the current git commit
git rev-parse --short HEAD

# Get the job ID
squeue -u $USER
```
