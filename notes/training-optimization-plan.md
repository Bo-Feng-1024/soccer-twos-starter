# Training Optimization Plan: 68% → 90% vs CEIA Baseline

## Current Status (2026-04-22)

**Best model**: checkpoint-5122, **83% win rate** vs ceia_baseline (500-game evaluation).

### Key Findings

**1. The CEIA baseline is a deterministic policy** (confirmed by the instructor)
- The theoretical ceiling is 100% — every loss is a deficiency in our own policy.
- The baseline's offensive pattern is fixed and can be fully predicted and defended.

**2. Brandão 2022's central conclusion: Self-Play >> training against a fixed opponent**
- The paper shows that training only against a fixed heuristic **never improves** (Fig 6-8), while self-play steadily climbs from scratch to 93%.
- **This explains all of our regressions when continuing from checkpoint-5122** — heavy ceia training prevents exploration of new strategies.
- Self-play creates an automatic curriculum: the opponent scales with the agent.
- The paper trained for 320M steps to converge; we have run at most 46M.

**3. Other key design choices in the paper**
- 8-frame stacking (encodes velocity / direction information)
- lr=0.0004, fixed (no decay)
- 5 epochs (not 10)
- Separated policy / value networks
- Goal reward weighted by time (earlier goals are worth more)

### Strategies Tried, with Results

| # | Strategy | Result | Conclusion |
|---|----------|--------|-----------|
| 1 | PPO hyperparameter tuning + 100% ceia | checkpoint-5122: **83%** (500 games) | Effective: 68%→83%, but hit a ceiling |
| 2 | Continue 100% ceia training | checkpoint-5749: 83% | Overfit-driven regression |
| 3 | 70% ceia + 30% selfplay | checkpoint-5750: 86% (100 games) | Prevented regression but did not break through |
| 4 | BC pretraining + PPO fine-tune | reward 0.078 | Total failure (distribution shift) |
| 5 | High entropy (0.01) + high lr (1e-4) | checkpoint-5650: 84% (100 games) | Regressed |
| 6 | [512,512] wide network from scratch | checkpoint-591: 6% | Undertrained |
| 7 | Reward fine-tuning (exp C) | checkpoint-5750: 84% (100 games) | Slight gain, no breakthrough |
| 8 | Separated networks (exp D) | checkpoint-1700: 78% | Undertrained (1700 iter vs 5000+ needed) |
| 9 | Default-hyperparameter resume (exp G) | checkpoint-6600: 75% | Severe regression |
| 10 | Heavy defensive reward (exp H) | checkpoint-5750: 80% | Defensive reward too conservative |
| 11 | MeanStdFilter on resume | — | Failed (NoFilter incompatible) |
| 12 | MeanStdFilter from scratch | checkpoint-593: reward 0.075 | Too slow |

### Next Steps (paper-inspired)

| Experiment | Core change | Starting point | Hypothesis |
|-----------|-------------|----------------|------------|
| **I (priority)** | Invert opponent ratio (30% ceia + 70% selfplay), lr=3e-4, 5 epochs | checkpoint-5122 | Paper's central finding: self-play dominance is needed to break through |
| **J (follow-up)** | Frame stacking (4 frames) + self-play + larger network | From scratch | Frame stacking provides temporal information, improving defensive reactions |

### Directions Already Ruled Out

- Observation normalization (MeanStdFilter) — observations are already in [0,1], and it is incompatible with restore.
- 100% ceia training — the paper proves training against a fixed opponent cannot break through.
- Default hyperparameters — empirically lead to severe regression.

---

## Original Plan (2026-04-14)

The original optimization plan, mostly executed by now. Retained for reference.

---

## Context (original)

The strongest checkpoint at the time was checkpoint-2429 (`train_PPO_team.py`, 50% ceia opponent mix), with a 68% win rate vs ceia_baseline; the target was 90%. The training history was severely unstable (62% → 15% → 56% → 68%).

**Root cause**: none of the training scripts set the key PPO hyperparameters; they used the Ray 1.13.0 defaults, several of which are seriously inappropriate for this task (verified against `DEFAULT_CONFIG`).

## Background: Frank's Training Iteration History

Starting on April 5, Frank iterated continuously on PACE through four phases. The analysis below summarises the strategy, results, and lessons of each phase.

### Phase 1: Reward Shaping + Self-Play (4/5 — 4/10)

- Rewrote `RewardShapingWrapper` (six reward signals); trained from scratch with `train_ray_selfplay.py`.
- Found that GPU offered no benefit for this task (Unity simulation does not use GPU; the MLP is too small) — switched to CPU thereafter.
- checkpoint-700 vs ceia: **46%** win rate.

**Lesson**: a pure-self-play agent learns to play itself but does not learn to play CEIA.

### Phase 2: Continued Training + Opponent-Update Threshold Tuning (4/10 — 4/12)

- Resumed from checkpoint-700, adjusted the opponent-update threshold.
- checkpoint-1000: **62%** (best)
- checkpoint-1900: **15%** (collapse)
- checkpoint-2000: **15%** (sustained low)

**Lesson**: self-play training is extremely unstable — the win rate plunged from 62% to 15%. The cause was opponent updates that were too frequent combined with no `grad_clip`: a single large gradient update can wipe out the learned policy.

### Phase 3: League Self-Play — Adding CEIA Opponents (4/13)

- After identifying the pure-self-play problem, created `train_ray_selfplay_league.py`, which faces CEIA with 15% probability.
- However, the signal turned out to be diluted: any of agents 0-3 could be mapped to "default", so the default policy spent half its time playing itself, and reward stayed near 0.
- checkpoint-1000 (league): **43%** (worse than before).

**Lesson**: 15% ceia is too low, and the policy mapping design diluted the training signal.

### Phase 4: Team Training — 50% ceia + cooldown (4/13 — 4/14)

- Created `train_PPO_team.py` with key improvements:
  - Team 0 (agents 0, 1) is fixed to "default"; Team 1 (agents 2, 3) is sampled from the opponent pool.
  - 50% ceia + 30% selfplay_1 + 20% selfplay_2.
  - Added a cooldown (every 20 iterations) to prevent over-frequent opponent updates.
- checkpoint-2296: **56%**
- checkpoint-2429: **68%** (best to date).

**Lesson**: increasing the ceia ratio + fixing policy mapping + adding cooldown all helped, but 68% is still far from 90%.

### Training Curve Analysis

![PPO Self-Play vs Reward Shaped](../training_analysis.png)

- **Top-left (Reward)**: RS converges faster (~1M steps) and higher (2.3 vs 1.9), confirming that reward shaping accelerates learning.
- **Top-right (Policy Loss)**: SP shows sharp early spikes — without `grad_clip`, gradients are unstable.
- **Bottom-left (VF Loss)**: SP fluctuates wildly (0.05–0.35); inaccurate value estimates interfere with policy learning.
- **Bottom-right (Entropy)**: both drop sharply from ~3.0 to ~0.5 — the policy locks in early, and `entropy_coeff=0.0` cannot prevent it.

### Summary: Optimization Directions Derived from the History

| Observation | Conclusion | Optimization |
|-------------|-----------|--------------|
| 62% → 15% collapse | No `grad_clip` + no entropy → catastrophic forgetting | Add `grad_clip=0.5`, `entropy_coeff=0.005` |
| 15% ceia not enough → 50% better | The higher the ceia ratio, the more targeted | Move to 100% ceia |
| Diluted-signal problem | Policy mapping must cleanly separate trainee / opponent | Simplify to 2 policies (default + opponent_ceia) |
| All scripts using defaults | Many PPO hyperparameters are untuned | Systematically fix 8 hyperparameters |

## Core Finding: The Default-Hyperparameter Problem

| Parameter | Current (default) | Issue | Target |
|-----------|------------------|-------|--------|
| `entropy_coeff` | **0.0** | No exploration bonus; policy converges prematurely | 0.005 |
| `lambda` (GAE) | **1.0** | Pure Monte Carlo returns — high variance | 0.95 |
| `grad_clip` | **None** | No gradient clipping; large reward swings cause catastrophic updates | 0.5 |
| `clip_param` | **0.3** | Wider than the PPO paper's 0.2 — updates too aggressive | 0.2 |
| `train_batch_size` | **4000** | Collects ~105k steps but only uses 30×128=3840 — wastes 96% of the data | 20000 |
| `sgd_minibatch_size` | **128** | Too small; gradient estimates are noisy | 2048 |
| `num_sgd_iter` | **30** | 30 SGD epochs over 128 samples overfits a single batch | 10 |
| `vf_loss_coeff` | **1.0** | With `vf_share_layers=True`, VF loss dominates the policy gradient | 0.5 |

## Implementation Steps

### Step 1: Create `train_focused_ceia.py`

Copy `train_PPO_team.py` and modify:

**1a. Training strategy: 100% CEIA opponents (drop self-play)**

```python
def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0 or agent_id == 1:
        return "default"
    return "opponent_ceia"  # 100% CEIA
```

Simplify to 2 policies (drop opponent_1/2/3) and simplify the callback (only initialise ceia weights — no self-play update).

**1b. PPO hyperparameter fixes**

Add to the config:
```python
"entropy_coeff": 0.005,
"lambda": 0.95,
"grad_clip": 0.5,
"clip_param": 0.2,
"train_batch_size": 20000,
"rollout_fragment_length": 1000,
"sgd_minibatch_size": 2048,
"num_sgd_iter": 10,
"vf_loss_coeff": 0.5,
"lr_schedule": [
    [0, 3e-4],
    [5_000_000, 1e-4],
    [15_000_000, 3e-5],
],
```

**1c. Training configuration**
- `restore` from checkpoint-2429.
- `stop`: 30M timesteps / 23h (leave a 1h buffer).
- `checkpoint_freq`: 50 (save more frequently).
- Everything else unchanged: 7 workers, 3 envs/worker, [256,256] relu, no GPU.

### Step 2: Create `scripts/focused_ceia.batch`

Copy `scripts/team_ceia50.batch` and change the training script to `train_focused_ceia.py`.

### Step 3: (Optional) Fine-tune reward weights

In `utils.py`:
- kick reward: 0.001 → 0.002 (encourage more aggressive shooting)
- offensive reward: 0.0004 → 0.0008 (encourage pushing the ball toward goal)
- Everything else unchanged.

Lower priority than the hyperparameter fixes; skip if Step 1 already reaches 90%.

### Step 4: Update the evaluation agent

After training, update the `CHECKPOINT_PATH` in `reward_shaping_ppo_agent/agent.py` to point at the new checkpoint.

## Key Files

| File | Action |
|------|--------|
| `train_PPO_team.py` | Copy as template (do not modify) |
| `train_focused_ceia.py` | **New** — 100% ceia + hyperparameter fixes |
| `scripts/focused_ceia.batch` | **New** — PACE batch script |
| `utils.py` | Optional reward weight tuning |
| `reward_shaping_ppo_agent/agent.py` | Update checkpoint path after training |

## Expected Effects

| Change | Expected impact | Confidence |
|--------|-----------------|------------|
| entropy_coeff=0.005 | +3-5% | High |
| grad_clip=0.5 | +3-5% | High (eliminates catastrophic updates) |
| lambda=0.95 | +2-4% | High |
| Fix batch / minibatch | +3-5% | High |
| clip_param=0.2 | +1-3% | High |
| 100% CEIA training | +5-10% | Medium-high |
| lr_schedule | +1-3% | Medium |
| Total | 68% → ~85-93% | |

## Verification

```bash
# Submit training on PACE
sbatch scripts/focused_ceia.batch

# Monitor the training log; watch for:
# - policy_reward_mean/default rising steadily
# - entropy > 0 (confirms entropy_coeff is in effect)
# - grad_global_norm being clipped (confirms grad_clip is in effect)

# After training, evaluate
conda run -n soccertwos python -m soccer_twos.evaluate \
    -m1 reward_shaping_ppo_agent \
    -m2 ceia_baseline_agent \
    -e 100
```

## Risks

1. **Overfitting to CEIA** — training 100% against ceia could yield a narrow policy that only beats ceia. But the grading rubric is "9/10 wins vs the Baseline Agent", so overfit = scoring.
2. **lr=3e-4 may be too high initially** — `grad_clip=0.5` provides protection; if uncomfortable, start from 1e-4.
3. **checkpoint-2429 lives on PACE** — confirm the restore path is reachable from PACE scratch.
