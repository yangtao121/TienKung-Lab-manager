from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

from .walk_manager_env_cfg import WalkManagerEnvCfg


class WalkManagerRLEnv(ManagerBasedRLEnv):
    cfg: WalkManagerEnvCfg

    def __init__(self, cfg: WalkManagerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]

        self.feet_body_ids, _ = self.robot.find_bodies(name_keys=["ankle_roll_l_link", "ankle_roll_r_link"], preserve_order=True)
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "hip_roll_l_joint",
                "hip_pitch_l_joint",
                "hip_yaw_l_joint",
                "knee_pitch_l_joint",
                "ankle_pitch_l_joint",
                "ankle_roll_l_joint",
            ],
            preserve_order=True,
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                "hip_roll_r_joint",
                "hip_pitch_r_joint",
                "hip_yaw_r_joint",
                "knee_pitch_r_joint",
                "ankle_pitch_r_joint",
                "ankle_roll_r_joint",
            ],
            preserve_order=True,
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=["ankle_pitch_l_joint", "ankle_pitch_r_joint", "ankle_roll_l_joint", "ankle_roll_r_joint"],
            preserve_order=True,
        )

        self.gait_phase = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.gait_cycle = torch.full((self.num_envs,), self.cfg.gait_cycle, dtype=torch.float, device=self.device)
        self.phase_ratio = torch.tensor(
            [self.cfg.gait_air_ratio_l, self.cfg.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.gait_phase_offset_l, self.cfg.gait_phase_offset_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        self.avg_feet_force_per_step = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.avg_feet_speed_per_step = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)

        self.reset_env_ids = torch.zeros((0,), dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def _update_gait_phase(self):
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) > 0:
            self.avg_feet_force_per_step[env_ids] = 0.0
            self.avg_feet_speed_per_step[env_ids] = 0.0

    def step(self, action: torch.Tensor):
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        self.avg_feet_force_per_step[:] = 0.0
        self.avg_feet_speed_per_step[:] = 0.0

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)

            net_force = self.scene.sensors["contact_forces"].data.net_forces_w[:, self.feet_body_ids, :3]
            self.avg_feet_force_per_step += torch.norm(net_force, dim=-1)
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.decimation
        self.avg_feet_speed_per_step /= self.cfg.decimation

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._update_gait_phase()

        # direct-like order: update command/events first, then termination+reward
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.time_out_buf = self.reset_time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(self.reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(self.reset_env_ids)
            self._reset_idx(self.reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(self.reset_env_ids)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
