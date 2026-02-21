from __future__ import annotations

import math

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

import locomotion.observations as walk_obs
import locomotion.rewards as walk_rew

from assets.tienkung2_lite import TIENKUNG2LITE_CFG
from terrains import GRAVEL_TERRAINS_CFG


LEG_JOINT_NAMES = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
]

ARM_JOINT_NAMES = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]

FEET_BODY_NAMES = ["ankle_roll_l_link", "ankle_roll_r_link"]
KNEE_BODY_NAMES = ["knee_pitch_l_link", "knee_pitch_r_link"]


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
        joint_names=LEG_JOINT_NAMES,
        preserve_order=True,
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
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
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
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
        )
        actions = ObsTerm(func=base_mdp.last_action)
        gait_sin = ObsTerm(func=walk_obs.gait_sin)
        gait_cos = ObsTerm(func=walk_obs.gait_cos)
        phase_ratio = ObsTerm(func=walk_obs.phase_ratio)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        feet_contact = ObsTerm(
            func=walk_obs.feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True)},
        )

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
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        },
    )

    reset_arm_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True),
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
    joint_pos = RewTerm(
        func=walk_rew.joint_pos,
        weight=1.6,
        params={
            "target_joint_pos_scale": 0.17,
            "cycle_time": 0.64,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    feet_clearance = RewTerm(
        func=walk_rew.FeetClearanceReward,
        weight=1.0,
        params={
            "target_feet_height": 0.06,
            "cycle_time": 0.64,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    feet_contact_number = RewTerm(
        func=walk_rew.feet_contact_number,
        weight=1.2,
        params={
            "cycle_time": 0.64,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    feet_air_time = RewTerm(
        func=walk_rew.FeetAirTimeReward,
        weight=1.0,
        params={
            "cycle_time": 0.64,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    foot_slip = RewTerm(
        func=walk_rew.foot_slip,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    feet_distance = RewTerm(
        func=walk_rew.feet_distance,
        weight=0.2,
        params={
            "min_dist": 0.2,
            "max_dist": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    knee_distance = RewTerm(
        func=walk_rew.knee_distance,
        weight=0.2,
        params={
            "min_dist": 0.2,
            "max_dist": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=KNEE_BODY_NAMES, preserve_order=True),
        },
    )
    feet_contact_forces = RewTerm(
        func=walk_rew.feet_contact_forces,
        weight=-0.01,
        params={
            "max_contact_force": 700.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODY_NAMES, preserve_order=True),
        },
    )
    tracking_lin_vel = RewTerm(
        func=walk_rew.tracking_lin_vel,
        weight=1.2,
        params={
            "command_name": "base_velocity",
            "tracking_sigma": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    tracking_ang_vel = RewTerm(
        func=walk_rew.tracking_ang_vel,
        weight=1.1,
        params={
            "command_name": "base_velocity",
            "tracking_sigma": 5.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    vel_mismatch_exp = RewTerm(
        func=walk_rew.vel_mismatch_exp,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    low_speed = RewTerm(
        func=walk_rew.low_speed,
        weight=0.2,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    track_vel_hard = RewTerm(
        func=walk_rew.track_vel_hard,
        weight=0.5,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    default_joint_pos = RewTerm(
        func=walk_rew.default_joint_pos,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    orientation = RewTerm(
        func=walk_rew.orientation,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewTerm(
        func=walk_rew.base_height,
        weight=0.2,
        params={
            "base_height_target": 0.89,
            "cycle_time": 0.64,
            "feet_body_cfg": SceneEntityCfg("robot", body_names=FEET_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_acc = RewTerm(
        func=walk_rew.BaseAccReward,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_smoothness = RewTerm(
        func=walk_rew.ActionSmoothnessReward,
        weight=-0.002,
    )
    torques = RewTerm(
        func=walk_rew.torques,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    dof_vel = RewTerm(
        func=walk_rew.dof_vel,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    dof_acc = RewTerm(
        func=walk_rew.dof_acc,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    collision = RewTerm(
        func=walk_rew.collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["knee_pitch.*", "pelvis"]),
        },
    )
    positive_reward_clip = RewTerm(
        func=walk_rew.positive_reward_clip,
        weight=1.0,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis"]),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)


@configclass
class TienKung2LiteRoughEnvCfg(ManagerBasedRLEnvCfg):
    scene: WalkSceneCfg = WalkSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # kept for gait observation terms
    gait_cycle: float = 0.64
    gait_air_ratio_l: float = 0.5
    gait_air_ratio_r: float = 0.5
    gait_phase_offset_l: float = 0.0
    gait_phase_offset_r: float = 0.5

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

