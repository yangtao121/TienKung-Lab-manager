from __future__ import annotations

import torch

from rsl_rl.env import VecEnv


class ManagerRslVecEnvAdapter(VecEnv):
    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.num_actions = env.action_manager.total_action_dim
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf
        self.step_dt = env.step_dt
        self.cfg = env.cfg
        self.reset_env_ids = torch.zeros((0,), dtype=torch.long, device=self.device)

        self.env.reset()

    @property
    def unwrapped(self):
        return self.env

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)

    def get_observations(self):
        obs_dict = self.env.observation_manager.compute(update_history=False)
        policy_obs = obs_dict["policy"]
        critic_obs = obs_dict["critic"]
        extras = {"observations": {"critic": critic_obs}}
        return policy_obs, extras

    def reset(self):
        obs_dict, extras = self.env.reset()
        policy_obs = obs_dict["policy"]
        extras.setdefault("observations", {})
        extras["observations"]["critic"] = obs_dict["critic"]
        return policy_obs, extras

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).long()
        extras.setdefault("observations", {})
        extras["observations"]["critic"] = obs_dict["critic"]
        extras["time_outs"] = truncated

        self.episode_length_buf = self.env.episode_length_buf
        if hasattr(self.env, "reset_env_ids"):
            self.reset_env_ids = self.env.reset_env_ids

        return obs_dict["policy"], rewards, dones, extras

    def close(self):
        return self.env.close()
