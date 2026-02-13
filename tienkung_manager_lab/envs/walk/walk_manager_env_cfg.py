from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity import mdp as vel_mdp

import tienkung_manager_lab.envs.walk.mdp.observations as walk_obs
import tienkung_manager_lab.envs.walk.mdp.rewards as walk_rew

from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG


@configclass
class WalkSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = TIENKUNG2LITE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.005,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=base_mdp.last_action)
        gait_sin = ObsTerm(func=walk_obs.gait_sin)
        gait_cos = ObsTerm(func=walk_obs.gait_cos)
        phase_ratio = ObsTerm(func=walk_obs.phase_ratio)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
        velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        actions = ObsTerm(func=base_mdp.last_action)
        gait_sin = ObsTerm(func=walk_obs.gait_sin)
        gait_cos = ObsTerm(func=walk_obs.gait_cos)
        phase_ratio = ObsTerm(func=walk_obs.phase_ratio)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        feet_contact = ObsTerm(func=walk_obs.feet_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_roll.*")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=base_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=walk_rew.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=walk_rew.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=walk_rew.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=walk_rew.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=walk_rew.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=base_mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=walk_rew.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=walk_rew.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"]
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=walk_rew.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
        weight=-2.0,
    )
    flat_orientation_l2 = RewTerm(func=base_mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=walk_rew.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=walk_rew.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=walk_rew.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_roll.*"),
            "threshold": 500.0,
            "max_reward": 400.0,
        },
    )
    feet_too_near = RewTerm(
        func=walk_rew.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=walk_rew.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-2.0)

    joint_deviation_hip = RewTerm(
        func=walk_rew.joint_deviation_l1_zero_command,
        weight=-0.15,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_yaw_.*_joint",
                    "hip_roll_.*_joint",
                    "shoulder_pitch_.*_joint",
                    "elbow_pitch_.*_joint",
                ],
            ),
        },
    )
    joint_deviation_arms = RewTerm(
        func=walk_rew.joint_deviation_l1_zero_command,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_roll_.*_joint", "shoulder_yaw_.*_joint"]),
        },
    )
    joint_deviation_legs = RewTerm(
        func=walk_rew.joint_deviation_l1_zero_command,
        weight=-0.02,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["hip_pitch_.*_joint", "knee_pitch_.*_joint", "ankle_pitch_.*_joint", "ankle_roll_.*_joint"],
            ),
        },
    )

    gait_feet_frc_perio = RewTerm(func=walk_rew.gait_feet_frc_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_spd_perio = RewTerm(func=walk_rew.gait_feet_spd_perio, weight=1.0, params={"delta_t": 0.02})
    gait_feet_frc_support_perio = RewTerm(
        func=walk_rew.gait_feet_frc_support_perio,
        weight=0.6,
        params={"delta_t": 0.02},
    )

    ankle_torque = RewTerm(func=walk_rew.ankle_torque, weight=-0.0005)
    ankle_action = RewTerm(func=walk_rew.ankle_action, weight=-0.001)
    hip_roll_action = RewTerm(func=walk_rew.hip_roll_action, weight=-1.0)
    hip_yaw_action = RewTerm(func=walk_rew.hip_yaw_action, weight=-1.0)
    feet_y_distance = RewTerm(func=walk_rew.feet_y_distance, weight=-2.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)


@configclass
class WalkManagerEnvCfg(ManagerBasedRLEnvCfg):
    scene: WalkSceneCfg = WalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    gait_cycle: float = 0.85
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.is_finite_horizon = False

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
