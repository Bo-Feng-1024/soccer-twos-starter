from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


class RewardShapingWrapper(gym.core.Wrapper):
    """
    Adds dense reward shaping on top of the sparse SoccerTwos goal reward.

    The default environment only rewards scoring a goal (+1) or conceding (-1),
    making learning extremely slow. This wrapper adds two continuous signals:

      1. Ball proximity bonus: small reward each step for being close to the ball.
         Encourages the agent to actively chase and engage with the ball instead
         of standing still.

      2. Ball progress bonus: reward proportional to the ball's velocity toward
         the opponent's goal (positive x-direction for the blue team).
         Encourages the agent to kick the ball in the right direction.

    Both signals are small relative to the goal reward (±1) so they guide
    exploration without dominating the true objective.

    This wrapper requires position info from the environment step. The info dict
    is populated when the environment binary sends 345-dim observations (the
    extra 9 values are player/ball absolute positions). When the info dict is
    empty, shaping returns 0.0 and the original reward is unchanged.

    Works with team_vs_policy + single_player=True (single-agent training mode).
    """

    # Weight for ball proximity bonus.
    # Max contribution per step = PROXIMITY_WEIGHT * 1.0
    PROXIMITY_WEIGHT = 0.001

    # Weight for ball progress bonus.
    # Contribution per step = PROGRESS_WEIGHT * ball_x_velocity
    PROGRESS_WEIGHT = 0.002

    # Approximate half-length of the field, used to normalize proximity distance.
    FIELD_HALF_LENGTH = 10.0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward + self._shaping(info), done, info

    def _shaping(self, info):
        # info dict is empty when the env binary doesn't send position data
        if not info or "ball_info" not in info or "player_info" not in info:
            return 0.0

        ball_pos = np.array(info["ball_info"]["position"])   # [x, z] world coords
        ball_vel = np.array(info["ball_info"]["velocity"])   # [vx, vz]
        player_pos = np.array(info["player_info"]["position"])

        # Signal 1: proximity — reward being close to the ball
        dist = np.linalg.norm(ball_pos - player_pos)
        proximity_bonus = self.PROXIMITY_WEIGHT * max(0.0, 1.0 - dist / self.FIELD_HALF_LENGTH)

        # Signal 2: progress — reward the ball moving toward opponent's goal.
        # Blue team attacks in the +x direction, so positive vx = good.
        # If your agent plays as orange team, flip the sign here.
        progress_bonus = self.PROGRESS_WEIGHT * ball_vel[0]

        return proximity_bonus + progress_bonus


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
            - reward_shaping: if True, wraps the env with RewardShapingWrapper. Defaults to False.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    # soccer_twos.make() accepts **env_config and ignores unknown keys,
    # so reward_shaping passes through harmlessly.
    env = soccer_twos.make(**env_config)

    if env_config.get("reward_shaping", False):
        env = RewardShapingWrapper(env)

    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
