"""
Experiment C: Tuned reward shaping + from checkpoint-5122 (87%).

Hypothesis: Default reward weights may be suboptimal at high skill level.
- Increase offensive reward (encourage more aggressive goal-directed play)
- Reduce defensive penalty (stop being too conservative)
- Remove time-step penalty (stop rushing shots)
- Increase kick reward (reward stronger shots)

Uses a custom RewardShapingWrapper with modified weights, overriding utils.py.
"""
import os
import pickle

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
import soccer_twos


NUM_ENVS_PER_WORKER = 3
OPPONENT_UPDATE_COOLDOWN = 30

CEIA_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
)

RESTORE_CHECKPOINT = (
    "./ray_results/PPO_focused_ceia/"
    "PPO_Soccer_396a8_00000_0_2026-04-16_07-55-55/checkpoint_005122/checkpoint-5122"
)


# ── Tuned Reward Shaping ────────────────────────────────────────────────────
class TunedRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    """Modified reward shaping with tuned weights for high-level play."""
    TEAM0_GOAL_X = -13.0
    TEAM1_GOAL_X = 13.0

    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_vel = None

    def reset(self):
        self.prev_ball_vel = None
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped = self._shaping(info)
        combined = {aid: reward[aid] + shaped[aid] for aid in reward}
        return obs, combined, done, info

    def _shaping(self, info):
        shaped = {aid: 0.0 for aid in info}
        ball_pos = ball_vel = None
        for aid in info:
            if "ball_info" in info[aid]:
                ball_pos = np.array(info[aid]["ball_info"]["position"])
                ball_vel = np.array(info[aid]["ball_info"]["velocity"])
                break
        if ball_pos is None:
            return shaped

        for aid in info:
            agent_info = info[aid]
            if "player_info" not in agent_info:
                continue
            player_pos = np.array(agent_info["player_info"]["position"])
            player_vel = np.array(agent_info["player_info"]["velocity"])
            own_goal_x = self.TEAM0_GOAL_X if aid < 2 else self.TEAM1_GOAL_X
            attack_dir = 1.0 if aid < 2 else -1.0

            # Signal 1: approach — unchanged
            to_ball = ball_pos - player_pos
            dist = np.linalg.norm(to_ball) + 1e-6
            approach_reward = np.dot(player_vel, to_ball / dist)
            shaped[aid] += 0.0002 * approach_reward

            # Signal 2: kick — INCREASED 2x (encourage harder shots)
            if self.prev_ball_vel is not None:
                delta_speed = np.linalg.norm(ball_vel) - np.linalg.norm(self.prev_ball_vel)
                if delta_speed > 0 and dist < 1.5:
                    shaped[aid] += 0.002 * delta_speed  # was 0.001

            # Signal 3: offensive — INCREASED 2x (more aggressive)
            shaped[aid] += 0.0008 * ball_vel[0] * attack_dir  # was 0.0004

            # Signal 4: defensive — REDUCED 2x (less conservative)
            danger = max(0.0, 1.0 - abs(ball_pos[0] - own_goal_x) / 5.0)
            shaped[aid] -= 0.0005 * danger  # was 0.001

            # Signal 5: time penalty — REMOVED (stop rushing)
            # shaped[aid] -= 0.00002  # removed

        # Signal 6: separation — unchanged
        for team_start in (0, 2):
            ids = [team_start, team_start + 1]
            if all("player_info" in info.get(i, {}) for i in ids):
                pos0 = np.array(info[ids[0]]["player_info"]["position"])
                pos1 = np.array(info[ids[1]]["player_info"]["position"])
                sep = min(float(np.linalg.norm(pos0 - pos1)), 5.0)
                for i in ids:
                    shaped[i] += 0.0001 * sep

        self.prev_ball_vel = ball_vel
        return shaped


def create_tuned_env(env_config={}):
    """Create env with tuned reward shaping."""
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return TunedRewardWrapper(env)
# ────────────────────────────────────────────────────────────────────────────


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0 or agent_id == 1:
        return "default"
    return np.random.choice(
        ["opponent_1", "opponent_2", "opponent_3"],
        p=[0.15, 0.15, 0.70],  # 70% ceia
    )


def _load_weights(checkpoint_path, policy_name="default"):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker_state = pickle.loads(data["worker"])
    state = worker_state["state"]
    if policy_name not in state:
        policy_name = list(state.keys())[0]
    return {k: v for k, v in state[policy_name].items() if k != "_optimizer_variables"}


class MixedOpponentCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self._ceia_initialized = False
        self._last_update_iter = 0

    def on_train_result(self, **info):
        trainer = info["trainer"]
        if not self._ceia_initialized:
            print("=== Loading ceia_baseline into opponent_3 (fixed) ===")
            try:
                ceia_weights = _load_weights(CEIA_CHECKPOINT)
                trainer.set_weights({"opponent_3": ceia_weights})
                print("=== opponent_3 = ceia_baseline (fixed forever) ===")
            except Exception as e:
                print(f"WARNING: failed to load ceia_baseline: {e}")
            self._ceia_initialized = True

        current_iter = info["result"]["training_iteration"]
        default_reward = info["result"].get("policy_reward_mean", {}).get("default", -999)
        since_last = current_iter - self._last_update_iter
        if default_reward > 0.1 and since_last >= OPPONENT_UPDATE_COOLDOWN:
            print(f"---- Updating selfplay opponents (iter={current_iter}, reward={default_reward:.3f}) ----")
            trainer.set_weights({
                "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                "opponent_1": trainer.get_weights(["default"])["default"],
            })
            self._last_update_iter = current_iter


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_tuned_env)  # use tuned rewards
    tmp = create_tuned_env()
    obs_space = tmp.observation_space
    act_space = tmp.action_space
    tmp.close()

    analysis = tune.run(
        "PPO",
        name="PPO_exp_c",
        config={
            "num_gpus": 0,
            "num_workers": 7,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": MixedOpponentCallback,
            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            # Same hyperparameters as best run (#10)
            "entropy_coeff": 0.005,
            "lambda": 0.95,
            "grad_clip": 0.5,
            "clip_param": 0.2,
            "train_batch_size": 20000,
            "sgd_minibatch_size": 2048,
            "num_sgd_iter": 10,
            "vf_loss_coeff": 0.5,
            "lr_schedule": [
                [33_000_000, 5e-5],
                [45_000_000, 3e-5],
                [60_000_000, 1e-5],
            ],
            "rollout_fragment_length": 1000,
            "batch_mode": "complete_episodes",
        },
        stop={
            "timesteps_total": 100_000_000,
            "time_total_s": 267000,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=RESTORE_CHECKPOINT,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
    print(best_trial)
    print(best_ckpt)
    print("Done training")
