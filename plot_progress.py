"""
Plot training progress from Ray results.
Works for both local and PACE results (copy progress.csv back first).

Usage:
    python plot_progress.py                          # auto-find all progress.csv
    python plot_progress.py <path/to/progress.csv>   # specific file
"""

import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt


def find_csv_files():
    pattern = "ray_results/**/progress.csv"
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("No progress.csv found under ray_results/")
        sys.exit(1)
    return sorted(files)


def load(path):
    df = pd.read_csv(path)
    return df[["timesteps_total", "episode_reward_mean"]].dropna()


def summarize(path, df):
    last = df.iloc[-1]
    best = df["episode_reward_mean"].max()
    converged = df["episode_reward_mean"].tail(50).std() < 0.05
    print(f"\n{'─'*60}")
    print(f"File   : {path}")
    print(f"Iters  : {len(df)}")
    print(f"Steps  : {int(last['timesteps_total']):,}")
    print(f"Reward (last): {last['episode_reward_mean']:.3f}")
    print(f"Reward (best): {best:.3f}")
    print(f"Converged?   : {'YES ✓' if converged else 'NO  (still training)'}")


def plot(files):
    plt.figure(figsize=(9, 5))
    for path in files:
        df = load(path)
        # use folder name as label
        label = path.split("/")[-2][:50]
        plt.plot(df["timesteps_total"], df["episode_reward_mean"], label=label)
        summarize(path, df)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward Mean")
    plt.title("Training Progress")
    plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = "training_progress.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = find_csv_files()
    plot(files)
