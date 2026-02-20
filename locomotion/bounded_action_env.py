from __future__ import annotations

import gymnasium as gym
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv


class BoundedActionManagerBasedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with a bounded [-1, 1] single_action_space.

    This is useful for RL libraries (e.g., skrl) that rely on the Gym action_space
    bounds to clip actions. IsaacLab's default ManagerBasedRLEnv uses an unbounded
    action space for convenience.
    """

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()

        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
