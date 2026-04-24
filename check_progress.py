"""Quick progress checker for Ray training results."""
import csv
import sys
import os
import glob


def find_progress(name="PPO_bc_finetune"):
    """Find the progress.csv file for a given experiment."""
    pattern = "ray_results/%s/*/progress.csv" % name
    files = glob.glob(pattern)
    if not files:
        print("No progress.csv found for %s" % name)
        sys.exit(1)
    return sorted(files)[-1]


def show(path, step=50):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print("File: %s" % path)
    print("Total iterations: %d" % len(rows))
    print("")
    print("iter\ttimesteps\treward\t\tep_len")
    print("----\t---------\t------\t\t------")
    for i, r in enumerate(rows):
        if i % step == 0 or i >= len(rows) - 3:
            print("%s\t%s\t%.4f\t\t%.1f" % (
                r["training_iteration"],
                r["timesteps_total"],
                float(r["episode_reward_mean"]),
                float(r["episode_len_mean"]),
            ))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "PPO_bc_finetune"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    path = find_progress(name)
    show(path, step)
