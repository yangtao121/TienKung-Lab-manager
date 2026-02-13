from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def gait_sin(env) -> torch.Tensor:
    return torch.sin(2.0 * torch.pi * env.gait_phase)


def gait_cos(env) -> torch.Tensor:
    return torch.cos(2.0 * torch.pi * env.gait_phase)


def phase_ratio(env) -> torch.Tensor:
    return env.phase_ratio


def feet_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
    return is_contact.float()


def avg_feet_force(env) -> torch.Tensor:
    return env.avg_feet_force_per_step


def avg_feet_speed(env) -> torch.Tensor:
    return env.avg_feet_speed_per_step


def action_buffer_last(env) -> torch.Tensor:
    return env.action_manager.action


def body_force_z(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]


def joint_action_subset(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    action = env.action_manager.action
    return action[:, : asset.num_joints]
