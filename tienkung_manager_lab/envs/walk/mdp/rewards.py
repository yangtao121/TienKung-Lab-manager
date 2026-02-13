from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _compute_gait_phase(env):
    t = env.episode_length_buf.float() * env.step_dt / env.cfg.gait_cycle
    gait_phase = torch.zeros((env.num_envs, 2), dtype=torch.float, device=env.device)
    gait_phase[:, 0] = (t + env.cfg.gait_phase_offset_l) % 1.0
    gait_phase[:, 1] = (t + env.cfg.gait_phase_offset_r) % 1.0
    return gait_phase


def _compute_phase_ratio(env):
    return torch.tensor(
        [env.cfg.gait_air_ratio_l, env.cfg.gait_air_ratio_r], dtype=torch.float, device=env.device
    ).repeat(env.num_envs, 1)


def _feet_force(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3]
    return torch.norm(net_force, dim=-1)


def _feet_speed(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)


def track_lin_vel_xy_yaw_frame_exp(env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)


def action_rate_l2(env):
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def undesired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def body_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def body_force(env, sensor_cfg: SceneEntityCfg, threshold: float = 500.0, max_reward: float = 400.0):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward = torch.where(reward < threshold, torch.zeros_like(reward), reward - threshold)
    return reward.clamp(min=0.0, max=max_reward)


def feet_too_near_humanoid(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2):
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0.0)


def feet_stumble(env, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5.0 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    ).float()


def joint_deviation_l1_zero_command(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    cmd = env.command_manager.get_command(command_name)
    zero_flag = (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def ankle_torque(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def ankle_action(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def hip_roll_action(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def hip_yaw_action(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return torch.sum(torch.abs(env.action_manager.action[:, asset_cfg.joint_ids]), dim=1)


def feet_y_distance(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ankle_roll_l_link", "ankle_roll_r_link"])):
    asset: Articulation = env.scene[asset_cfg.name]
    leftfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :] - asset.data.root_link_pos_w[:, :]
    rightfoot = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :] - asset.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(asset.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(asset.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_manager.get_command("base_velocity")[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


def gait_clock(phase, air_ratio, delta_t):
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1.0 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1.0 - delta_t)

    i_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2.0 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1.0 + delta_t) / (2.0 * delta_t) * trans_flag3
    )
    i_spd = 1.0 - i_frc
    return i_frc, i_spd


def gait_feet_frc_perio(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]),
    delta_t: float = 0.02,
):
    gait_phase = _compute_gait_phase(env)
    phase_ratio = _compute_phase_ratio(env)
    feet_force = _feet_force(env, sensor_cfg)
    left_frc_swing_mask = gait_clock(gait_phase[:, 0], phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(gait_phase[:, 1], phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200.0 * torch.square(feet_force[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200.0 * torch.square(feet_force[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]),
    delta_t: float = 0.02,
):
    gait_phase = _compute_gait_phase(env)
    phase_ratio = _compute_phase_ratio(env)
    feet_speed = _feet_speed(env, asset_cfg)
    left_spd_support_mask = gait_clock(gait_phase[:, 0], phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(gait_phase[:, 1], phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100.0 * torch.square(feet_speed[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100.0 * torch.square(feet_speed[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["ankle_roll_l_link", "ankle_roll_r_link"]),
    delta_t: float = 0.02,
):
    gait_phase = _compute_gait_phase(env)
    phase_ratio = _compute_phase_ratio(env)
    feet_force = _feet_force(env, sensor_cfg)
    left_frc_support_mask = gait_clock(gait_phase[:, 0], phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(gait_phase[:, 1], phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1.0 - torch.exp(-10.0 * torch.square(feet_force[:, 0])))
    right_frc_score = right_frc_support_mask * (1.0 - torch.exp(-10.0 * torch.square(feet_force[:, 1])))
    return left_frc_score + right_frc_score


def is_terminated(env):
    return env.termination_manager.terminated.float()
