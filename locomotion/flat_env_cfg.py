import math

from isaaclab.utils import configclass

from .rough_env_cfg import TienKung2LiteRoughEnvCfg


@configclass
class TienKung2LiteFlatEnvCfg(TienKung2LiteRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # ---------------------------------------------------------------------
        # Stage-1: make learning easier on flat terrain (stand first, then walk)
        # ---------------------------------------------------------------------

        # Reset base velocities to zero to avoid unstable initial conditions.
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        # Reset joints to their default pose (avoid hard-to-recover initial joint randomization).
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Disable randomization/pushes for the initial stage.
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None

        # Turn off observation corruption for the initial stage.
        self.observations.policy.enable_corruption = False

        # Simplify commands: forward + optional small yaw, with many standing envs.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.rel_standing_envs = 0.5

        # Use yaw-rate commands directly (disable heading targets) for stage-1.
        # self.commands.base_velocity.heading_command = True
        # self.commands.base_velocity.rel_heading_envs = 0.0
        # self.commands.base_velocity.ranges.heading = None

        # Relax termination: only terminate on pelvis contact (others are penalized).
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "pelvis"


@configclass
class TienKung2LiteFlatEnvCfg_V1(TienKung2LiteFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Use yaw-rate commands directly (disable heading targets).
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.heading = None

        # Add base linear velocity to policy observations (easier velocity tracking on flat).
        from isaaclab.envs import mdp as base_mdp
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

        self.observations.policy.base_lin_vel = ObsTerm(
            func=base_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        # Reduce history length to simplify inputs (cleaner motions, faster learning).
        self.observations.policy.history_length = 1
        self.observations.critic.history_length = 1

        # Stronger smoothness/swing penalties to reduce extra motion in the air.
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.dof_acc_l2.weight = -1.0e-6
        self.rewards.swing_feet_lateral_speed_penalty.weight = -0.15
        self.rewards.swing_hip_yaw_roll_vel_penalty.weight = -0.35
        self.rewards.ankle_action.weight = -0.01


@configclass
class TienKung2LiteFlatEnvCfg_PLAY_V1(TienKung2LiteFlatEnvCfg_V1):
    def __post_init__(self):
        super().__post_init__()

        # Small scene for play / video.
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Deterministic reset for evaluation.
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # No randomization during play.
        self.events.push_robot = None
        self.events.physics_material = None
        self.events.add_base_mass = None

        # Fixed command: walk forward.
        self.commands.base_velocity.resampling_time_range = (math.inf, math.inf)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
