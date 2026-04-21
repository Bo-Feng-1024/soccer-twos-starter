"""Quick check: what's the observation range? If already [0,1], MeanStdFilter won't help."""
import numpy as np
import soccer_twos

env = soccer_twos.make()
obs = env.reset()
o = obs[0]
print(f"Shape: {o.shape}")
print(f"Min: {o.min():.4f}, Max: {o.max():.4f}, Mean: {o.mean():.4f}, Std: {o.std():.4f}")
print(f"First 20: {o[:20]}")
print(f"Values > 1: {(o > 1).sum()}, Values < 0: {(o < 0).sum()}")
env.close()
