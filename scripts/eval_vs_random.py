"""Evaluate reward_shaping_ppo_agent vs selfmade_random_agent (headless).

Usage: python scripts/eval_vs_random.py [N]
Default N=50 (yields decent statistical power for ~90% win rate claim).
"""
import resource
# Cap RLIMIT_NOFILE so ray 1.4.0 doesn't pass sys.maxsize to Redis CONFIG SET.
# Only relevant on macOS where ulimit defaults to unlimited; on Linux this is a no-op.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft > 65535 or _hard > 65535:
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, min(_hard, 8192)))

import soccer_twos
import sys

sys.path.insert(0, ".")

env = soccer_twos.make(render=False)
from reward_shaping_ppo_agent.agent import RewardShapingPPOAgent
from selfmade_random_agent.agent import RandomAgent

m1 = RewardShapingPPOAgent(env)
m2 = RandomAgent(env)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
wins = draws = losses = 0

for i in range(N):
    obs = env.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        act = {}
        act.update(m1.act({k: obs[k] for k in [0, 1]}))
        act.update(m2.act({k: obs[k] for k in [2, 3]}))
        obs, rew, done, info = env.step(act)
    r = rew[0] + rew[1]
    if r > 0:
        wins += 1
    elif r < 0:
        losses += 1
    else:
        draws += 1
    print(f"Game {i+1}: W={wins} L={losses} D={draws} ({wins/(i+1)*100:.0f}%)", flush=True)

print(f"\nFinal: {wins}W {losses}L {draws}D out of {N} ({wins/N*100:.1f}%)")
env.close()
