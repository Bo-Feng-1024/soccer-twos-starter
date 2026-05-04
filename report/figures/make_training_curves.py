"""Generate Figure 1: Reward vs Steps comparison for the report."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/bofeng/Documents/CS8803_DRL/soccer-twos-starter/ray_results"
OUT = "/Users/bofeng/Documents/CS8803_DRL/final_project/report/figures/training_curves.pdf"

CSVS = [
    ("PPO Self-Play (no reward shaping)",
     f"{ROOT}/PPO_SP/PPO_Soccer_58633_00000_0_2026-03-31_15-10-56/progress.csv",
     "tab:gray"),
    ("PPO + Reward Shaping (self-play)",
     f"{ROOT}/PPO_reward_shaped/PPO_Soccer_62a5a_00000_0_2026-03-30_21-31-50/progress.csv",
     "tab:blue"),
]


def smooth(y, k=10):
    return pd.Series(y).rolling(k, min_periods=1).mean().values


fig, ax = plt.subplots(figsize=(6.0, 3.4))

for label, path, color in CSVS:
    df = pd.read_csv(path)
    x = df["timesteps_total"].values
    y = df["episode_reward_mean"].values
    ax.plot(x, y, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(x, smooth(y, 15), color=color, linewidth=2.0, label=label)

ax.set_xlabel("Environment Steps")
ax.set_ylabel("Episode Reward Mean")
ax.set_title("Training Curves: Effect of Reward Shaping")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight")
plt.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)

for label, path, _ in CSVS:
    df = pd.read_csv(path)
    last = df.iloc[-1]
    best = df["episode_reward_mean"].max()
    print(f"{label}")
    print(f"  iters={len(df)}, steps={int(last['timesteps_total']):,}, "
          f"reward_last={last['episode_reward_mean']:.3f}, reward_best={best:.3f}")

print(f"\nSaved: {OUT}")
