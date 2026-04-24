"""Evaluate reward_shaping_ppo_agent vs ceia_baseline_agent (headless)."""
import soccer_twos
import sys

sys.path.insert(0, ".")

env = soccer_twos.make(render=False)
from reward_shaping_ppo_agent.agent import RewardShapingPPOAgent
from ceia_baseline_agent.agent_ray import RayAgent as CEIAAgent

m1 = RewardShapingPPOAgent(env)
m2 = CEIAAgent(env)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
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
