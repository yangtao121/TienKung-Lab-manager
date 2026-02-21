from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _get_phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    return env.episode_length_buf.float() * env.step_dt / cycle_time


def _get_stance_mask(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    """Return stance mask with shape (N, 2): [left_stance, right_stance]."""
    phase = _get_phase(env, cycle_time)
    sin_pos = torch.sin(2.0 * torch.pi * phase)

    stance_mask = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float)
    stance_mask[:, 0] = (sin_pos >= 0.0).float()
    stance_mask[:, 1] = (sin_pos < 0.0).float()
    stance_mask[torch.abs(sin_pos) < 0.1] = 1.0
    return stance_mask


def _get_ref_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_joint_pos_scale: float,
    cycle_time: float,
) -> torch.Tensor:
    """Build gait reference for the 12 leg joints in fixed order.

    Expected leg order:
    [
        hip_roll_l, hip_pitch_l, hip_yaw_l, knee_pitch_l, ankle_pitch_l, ankle_roll_l,
        hip_roll_r, hip_pitch_r, hip_yaw_r, knee_pitch_r, ankle_pitch_r, ankle_roll_r,
    ]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[1] < 12:
        raise RuntimeError("Leg reference reward expects 12 ordered leg joints.")

    ref_joint_pos = torch.zeros_like(joint_pos)
    phase = _get_phase(env, cycle_time)
    sin_pos = torch.sin(2.0 * torch.pi * phase)

    sin_pos_l = sin_pos.clone()
    sin_pos_r = sin_pos.clone()

    scale_1 = target_joint_pos_scale
    scale_2 = 2.0 * scale_1

    # left swing contribution on sagittal joints: hip_pitch, knee_pitch, ankle_pitch
    sin_pos_l[sin_pos_l > 0.0] = 0.0
    ref_joint_pos[:, 1] = sin_pos_l * scale_1
    ref_joint_pos[:, 3] = sin_pos_l * scale_2
    ref_joint_pos[:, 4] = sin_pos_l * scale_1

    # right swing contribution on sagittal joints
    sin_pos_r[sin_pos_r < 0.0] = 0.0
    ref_joint_pos[:, 7] = sin_pos_r * scale_1
    ref_joint_pos[:, 9] = sin_pos_r * scale_2
    ref_joint_pos[:, 10] = sin_pos_r * scale_1

    # double support phase -> neutral reference
    ref_joint_pos[torch.abs(sin_pos) < 0.1] = 0.0
    return ref_joint_pos


def joint_pos(
    env: ManagerBasedRLEnv,
    target_joint_pos_scale: float,
    cycle_time: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_joint_pos = _get_ref_joint_pos(env, asset_cfg, target_joint_pos_scale, cycle_time)

    diff = current_joint_pos - target_joint_pos
    diff_norm = torch.norm(diff, dim=1)
    return torch.exp(-2.0 * diff_norm) - 0.2 * torch.clamp(diff_norm, 0.0, 0.5)


def feet_distance(
    env: ManagerBasedRLEnv,
    min_dist: float,
    max_dist: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    foot_dist = torch.norm(foot_pos_xy[:, 0, :] - foot_pos_xy[:, 1, :], dim=1)

    d_min = torch.clamp(foot_dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(foot_dist - max_dist, 0.0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0


def knee_distance(
    env: ManagerBasedRLEnv,
    min_dist: float,
    max_dist: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    knee_pos_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    knee_dist = torch.norm(knee_pos_xy[:, 0, :] - knee_pos_xy[:, 1, :], dim=1)

    d_min = torch.clamp(knee_dist - min_dist, -0.5, 0.0)
    d_max = torch.clamp(knee_dist - max_dist / 2.0, 0.0, 0.5)
    return (torch.exp(-torch.abs(d_min) * 100.0) + torch.exp(-torch.abs(d_max) * 100.0)) / 2.0


def foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0

    asset: Articulation = env.scene[asset_cfg.name]
    foot_speed_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    reward = torch.sqrt(torch.clamp_min(foot_speed_xy, 0.0)) * contact.float()
    return torch.sum(reward, dim=1)


def feet_contact_number(
    env: ManagerBasedRLEnv,
    cycle_time: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0
    stance_mask = _get_stance_mask(env, cycle_time).bool()

    reward = torch.where(contact == stance_mask, 1.0, -0.3)
    return torch.mean(reward, dim=1)


def orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    quat_mismatch = torch.exp(-(torch.abs(roll) + torch.abs(pitch)) * 10.0)
    flatness = torch.exp(-torch.norm(asset.data.projected_gravity_b[:, :2], dim=1) * 20.0)
    return (quat_mismatch + flatness) / 2.0


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    max_contact_force: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_norm = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.sum((force_norm - max_contact_force).clip(0.0, 400.0), dim=1)


def default_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    left_yaw_roll = joint_diff[:, [0, 2]]
    right_yaw_roll = joint_diff[:, [6, 8]]
    yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
    yaw_roll = torch.clamp(yaw_roll - 0.1, 0.0, 50.0)

    return torch.exp(-yaw_roll * 100.0) - 0.01 * torch.norm(joint_diff, dim=1)


def base_height(
    env: ManagerBasedRLEnv,
    base_height_target: float,
    cycle_time: float,
    feet_body_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    stance_mask = _get_stance_mask(env, cycle_time)

    feet_z = asset.data.body_pos_w[:, feet_body_cfg.body_ids, 2]
    measured_height = torch.sum(feet_z * stance_mask, dim=1) / torch.sum(stance_mask, dim=1).clamp_min(1.0)
    relative_base_height = asset.data.root_pos_w[:, 2] - (measured_height - 0.05)

    return torch.exp(-torch.abs(relative_base_height - base_height_target) * 100.0)


def vel_mismatch_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    lin_mismatch = torch.exp(-torch.square(asset.data.root_lin_vel_b[:, 2]) * 10.0)
    ang_mismatch = torch.exp(-torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1) * 5.0)
    return (lin_mismatch + ang_mismatch) / 2.0


def track_vel_hard(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    lin_vel_error = torch.norm(command[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=1)
    lin_vel_error_exp = torch.exp(-lin_vel_error * 10.0)

    ang_vel_error = torch.abs(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    ang_vel_error_exp = torch.exp(-ang_vel_error * 10.0)

    linear_error = 0.2 * (lin_vel_error + ang_vel_error)
    return (lin_vel_error_exp + ang_vel_error_exp) / 2.0 - linear_error


def tracking_lin_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    lin_vel_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error * tracking_sigma)


def tracking_ang_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error * tracking_sigma)


def low_speed(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    absolute_speed = torch.abs(asset.data.root_lin_vel_b[:, 0])
    absolute_command = torch.abs(command[:, 0])

    speed_too_low = absolute_speed < 0.5 * absolute_command
    speed_too_high = absolute_speed > 1.2 * absolute_command
    speed_desired = ~(speed_too_low | speed_too_high)

    sign_mismatch = torch.sign(asset.data.root_lin_vel_b[:, 0]) != torch.sign(command[:, 0])

    reward = torch.zeros_like(asset.data.root_lin_vel_b[:, 0])
    reward[speed_too_low] = -1.0
    reward[speed_too_high] = 0.0
    reward[speed_desired] = 1.2
    reward[sign_mismatch] = -2.0
    return reward * (torch.abs(command[:, 0]) > 0.1)


def torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def dof_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def dof_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_norm = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.sum((contact_norm > 0.1).float(), dim=1)


class FeetAirTimeReward(ManagerTermBase):
    """Humanoid-gym style feet air-time reward with internal contact state."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]

        self._sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._body_ids = sensor_cfg.body_ids

        num_feet = len(self._body_ids)
        self._feet_air_time = torch.zeros((env.num_envs, num_feet), dtype=torch.float, device=env.device)
        self._last_contacts = torch.zeros((env.num_envs, num_feet), dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._feet_air_time[env_ids] = 0.0
        self._last_contacts[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv, cycle_time: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        contact = self._sensor.data.net_forces_w[:, self._body_ids, 2] > 5.0
        stance_mask = _get_stance_mask(env, cycle_time).bool()

        contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self._last_contacts)
        self._last_contacts[:] = contact

        first_contact = (self._feet_air_time > 0.0) * contact_filt
        self._feet_air_time += env.step_dt

        air_time = self._feet_air_time.clamp(0.0, 0.5) * first_contact.float()
        self._feet_air_time *= (~contact_filt).float()

        return air_time.sum(dim=1)


class FeetClearanceReward(ManagerTermBase):
    """Humanoid-gym style swing-foot clearance reward with internal height state."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]

        self._sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._sensor_body_ids = sensor_cfg.body_ids

        self._asset: Articulation = env.scene[asset_cfg.name]
        self._asset_body_ids = asset_cfg.body_ids

        num_feet = len(self._asset_body_ids)
        self._last_feet_z = torch.full((env.num_envs, num_feet), 0.05, dtype=torch.float, device=env.device)
        self._feet_height = torch.zeros((env.num_envs, num_feet), dtype=torch.float, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._last_feet_z[env_ids] = 0.05
        self._feet_height[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_feet_height: float,
        cycle_time: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        contact = self._sensor.data.net_forces_w[:, self._sensor_body_ids, 2] > 5.0

        feet_z = self._asset.data.body_pos_w[:, self._asset_body_ids, 2] - 0.05
        delta_z = feet_z - self._last_feet_z
        self._feet_height += delta_z
        self._last_feet_z[:] = feet_z

        swing_mask = 1.0 - _get_stance_mask(env, cycle_time)
        reward_pos = (torch.abs(self._feet_height - target_feet_height) < 0.01).float()
        reward_pos = torch.sum(reward_pos * swing_mask, dim=1)

        self._feet_height *= (~contact).float()
        return reward_pos


class BaseAccReward(ManagerTermBase):
    """Humanoid-gym style base acceleration reward from root velocity differences."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_root_vel = torch.zeros((env.num_envs, 6), dtype=torch.float, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._last_root_vel[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        root_vel = torch.cat([asset.data.root_lin_vel_w[:, :3], asset.data.root_ang_vel_w[:, :3]], dim=1)

        root_acc = self._last_root_vel - root_vel
        self._last_root_vel[:] = root_vel

        return torch.exp(-torch.norm(root_acc, dim=1) * 3.0)


class ActionSmoothnessReward(ManagerTermBase):
    """Humanoid-gym style action smoothness with second-order difference term."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_last_actions = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._last_last_actions[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        actions = env.action_manager.action
        last_actions = env.action_manager.prev_action

        term_1 = torch.sum(torch.square(last_actions - actions), dim=1)
        term_2 = torch.sum(torch.square(actions + self._last_last_actions - 2.0 * last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(actions), dim=1)

        self._last_last_actions[:] = last_actions
        return term_1 + term_2 + term_3


def positive_reward_clip(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Apply only-positive total reward behavior as a terminal term compensation."""
    reward_before_clip = env.reward_manager._reward_buf
    clipped_reward = torch.clamp(reward_before_clip, min=0.0)
    delta = clipped_reward - reward_before_clip
    return delta / max(env.step_dt, 1.0e-6)
