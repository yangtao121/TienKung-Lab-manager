from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _compute_gait_phase(env) -> torch.Tensor:
    t = env.episode_length_buf.float() * env.step_dt / env.cfg.gait_cycle
    phase = torch.zeros((env.num_envs, 2), dtype=torch.float, device=env.device)
    phase[:, 0] = (t + env.cfg.gait_phase_offset_l) % 1.0
    phase[:, 1] = (t + env.cfg.gait_phase_offset_r) % 1.0
    return phase


def _compute_phase_ratio(env) -> torch.Tensor:
    return torch.tensor(
        [env.cfg.gait_air_ratio_l, env.cfg.gait_air_ratio_r], dtype=torch.float, device=env.device
    ).repeat(env.num_envs, 1)


def _compute_avg_feet_force(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3]
    return torch.norm(net_force, dim=-1)


def _compute_avg_feet_speed(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)


def gait_sin(env) -> torch.Tensor:
    return torch.sin(2.0 * torch.pi * _compute_gait_phase(env))


def gait_cos(env) -> torch.Tensor:
    return torch.cos(2.0 * torch.pi * _compute_gait_phase(env))


def phase_ratio(env) -> torch.Tensor:
    return _compute_phase_ratio(env)


def feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
    return is_contact.float()


def avg_feet_force(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]),
) -> torch.Tensor:
    return _compute_avg_feet_force(env, sensor_cfg)


def avg_feet_speed(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]),
) -> torch.Tensor:
    return _compute_avg_feet_speed(env, asset_cfg)


def action_buffer_last(env) -> torch.Tensor:
    return env.action_manager.action


def body_force_z(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]


def joint_action_subset(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    action = env.action_manager.action
    return action[:, : asset.num_joints]
