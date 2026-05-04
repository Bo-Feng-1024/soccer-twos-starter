"""Generate Figure 1: Reward-vs-steps training curve for each of the three agents.

Each subplot has its own y-axis because the per-step reward signal differs across
agents (Agent 1: sparse goal reward in self-play; Agent 2: shaped reward against
70% self-play / 30% CEIA mix; Agent 3: sparse goal reward against fixed CEIA).
"""

import gzip
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/Users/bofeng/Documents/CS8803_DRL"
OUT = f"{ROOT}/final_project/report/figures/training_curves.pdf"
DATA = f"{ROOT}/soccer-twos-starter/report/training_data"

# Agent 1: no-shaping pure self-play, local progress.csv
AGENT1 = (f"{ROOT}/soccer-twos-starter/ray_results/PPO_SP/"
          "PPO_Soccer_58633_00000_0_2026-03-31_15-10-56/progress.csv")

# Agent 2: chained from random-init through to checkpoint-2300 (the submitted)
AGENT2_PARTS = [f"{DATA}/agent2_{h}.csv.gz" for h in ("63f11", "29b63", "6047f", "6aa55")]

AGENT3 = f"{DATA}/agent3_bc.csv.gz"


def load(path):
    return pd.read_csv(path, compression="gzip" if path.endswith(".gz") else None)


def smooth(y, k=20):
    return pd.Series(y).rolling(k, min_periods=1).mean().values


def plot_agent(ax, df, title, color):
    x = df["timesteps_total"].values / 1e6  # show steps in millions
    y = df["episode_reward_mean"].values
    ax.plot(x, y, color=color, alpha=0.20, linewidth=0.6)
    ax.plot(x, smooth(y, 25), color=color, linewidth=1.6)
    ax.set_xlabel("Environment Steps (millions)")
    ax.set_ylabel("Episode Reward Mean")
    ax.set_title(title, fontsize=8.5)
    ax.grid(True, alpha=0.3)


agent1 = load(AGENT1)
agent2 = pd.concat([load(p) for p in AGENT2_PARTS], ignore_index=True)
agent3 = load(AGENT3)

fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.4))

plot_agent(axes[0], agent1, "Agent 1: PPO Self-Play\n(no reward shaping)", "tab:gray")
plot_agent(axes[1], agent2, "Agent 2: Reward Shaping + 70/30\n(submitted, full lineage)", "tab:blue")
plot_agent(axes[2], agent3, "Agent 3: BC + PPO Fine-tune\n(vs CEIA)", "tab:red")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)

for name, df in [("Agent 1", agent1), ("Agent 2", agent2), ("Agent 3", agent3)]:
    last = df.iloc[-1]
    best = df["episode_reward_mean"].max()
    print(f"{name}: iters={len(df)}, steps={int(last['timesteps_total']):,}, "
          f"reward_last={last['episode_reward_mean']:.3f}, reward_best={best:.3f}")

print(f"\nSaved: {OUT}")
