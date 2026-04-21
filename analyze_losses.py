"""Analyze when the agent loses — what patterns emerge?"""
import pickle, os, ray, numpy as np
from ray import tune
from ray.tune.registry import get_trainable_cls
import soccer_twos

ray.init(ignore_reinit_error=True)

# Load our agent
CHECKPOINT = "./ray_results/PPO_focused_ceia/PPO_Soccer_396a8_00000_0_2026-04-16_07-55-55/checkpoint_005122/checkpoint-5122"
config_path = os.path.join(os.path.dirname(CHECKPOINT), "../params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)
config["num_workers"] = 0
config["num_gpus"] = 0

from soccer_twos.utils import DummyEnv
from utils import RLLibWrapper
env_tmp = soccer_twos.make()
obs_space = env_tmp.observation_space
act_space = env_tmp.action_space
tune.registry.register_env("DummyEnv", lambda *_: RLLibWrapper(DummyEnv(obs_space, act_space)))
config["env"] = "DummyEnv"
env_tmp.close()

cls = get_trainable_cls("PPO")
agent = cls(env=config["env"], config=config)
with open(CHECKPOINT, "rb") as f:
    data = pickle.load(f)
worker_state = pickle.loads(data["worker"])
weights = {pid: {k: v for k, v in state.items() if k != "_optimizer_variables"} for pid, state in worker_state["state"].items()}
agent.workers.local_worker().set_weights(weights)
policy = agent.get_policy("default")

# Load ceia
ceia_ckpt = "./ceia_baseline_agent/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449"
ceia_config_path = os.path.join(os.path.dirname(ceia_ckpt), "../params.pkl")
with open(ceia_config_path, "rb") as f:
    ceia_config = pickle.load(f)
ceia_config["num_workers"] = 0
ceia_config["num_gpus"] = 0
tune.registry.register_env("DummyEnv2", lambda *_: RLLibWrapper(DummyEnv(obs_space, act_space)))
ceia_config["env"] = "DummyEnv2"
ceia_cls = get_trainable_cls("PPO")
ceia_agent = ceia_cls(env=ceia_config["env"], config=ceia_config)
ceia_agent.restore(ceia_ckpt)
ceia_policy = ceia_agent.get_policy("default")

# Run 50 episodes and track wins/losses
env = soccer_twos.make()
win_lengths = []
loss_lengths = []

for ep in range(50):
    obs = env.reset()
    done = {"__all__": False}
    steps = 0
    while not done["__all__"]:
        actions = {}
        for pid in [0, 1]:
            actions[pid], *_ = policy.compute_single_action(obs[pid])
        for pid in [2, 3]:
            actions[pid], *_ = ceia_policy.compute_single_action(obs[pid])
        obs, rewards, done, info = env.step(actions)
        steps += 1

    # Check who won (team 0 = our agent)
    team0_reward = rewards.get(0, 0) + rewards.get(1, 0)
    if team0_reward > 0:
        win_lengths.append(steps)
    else:
        loss_lengths.append(steps)

    if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1}/50: wins={len(win_lengths)}, losses={len(loss_lengths)}")

env.close()

print(f"\n=== Results ===")
print(f"Wins: {len(win_lengths)}, Losses: {len(loss_lengths)}")
print(f"Win rate: {len(win_lengths)/(len(win_lengths)+len(loss_lengths))*100:.1f}%")
if win_lengths:
    print(f"Win episode length: mean={np.mean(win_lengths):.1f}, std={np.std(win_lengths):.1f}")
if loss_lengths:
    print(f"Loss episode length: mean={np.mean(loss_lengths):.1f}, std={np.std(loss_lengths):.1f}")
