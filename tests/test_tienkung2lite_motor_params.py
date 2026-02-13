from __future__ import annotations

import pytest

pytest.importorskip("isaaclab")

from tienkung_manager_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG


def test_soft_joint_pos_limit_factor_matches_baseline() -> None:
    assert TIENKUNG2LITE_CFG.soft_joint_pos_limit_factor == 0.9


def test_leg_actuator_params_match_baseline() -> None:
    legs = TIENKUNG2LITE_CFG.actuators["legs"]
    assert legs.effort_limit_sim == {
        "hip_roll_.*_joint": 180,
        "hip_pitch_.*_joint": 300,
        "hip_yaw_.*_joint": 180,
        "knee_pitch_.*_joint": 300,
    }
    assert legs.velocity_limit_sim == {
        "hip_roll_.*_joint": 15.6,
        "hip_pitch_.*_joint": 15.6,
        "hip_yaw_.*_joint": 15.6,
        "knee_pitch_.*_joint": 15.6,
    }
    assert legs.stiffness == {
        "hip_roll_.*_joint": 700,
        "hip_pitch_.*_joint": 700,
        "hip_yaw_.*_joint": 500,
        "knee_pitch_.*_joint": 700,
    }
    assert legs.damping == {
        "hip_roll_.*_joint": 10,
        "hip_pitch_.*_joint": 10,
        "hip_yaw_.*_joint": 5,
        "knee_pitch_.*_joint": 10,
    }


def test_feet_actuator_params_match_baseline() -> None:
    feet = TIENKUNG2LITE_CFG.actuators["feet"]
    assert feet.effort_limit_sim == {
        "ankle_pitch_.*_joint": 60,
        "ankle_roll_.*_joint": 30,
    }
    assert feet.velocity_limit_sim == {
        "ankle_pitch_.*_joint": 12.8,
        "ankle_roll_.*_joint": 7.8,
    }
    assert feet.stiffness == {
        "ankle_pitch_.*_joint": 30,
        "ankle_roll_.*_joint": 16.8,
    }
    assert feet.damping == {
        "ankle_pitch_.*_joint": 2.5,
        "ankle_roll_.*_joint": 1.4,
    }


def test_arm_actuator_params_match_baseline() -> None:
    arms = TIENKUNG2LITE_CFG.actuators["arms"]
    assert arms.effort_limit_sim == {
        "shoulder_pitch_.*_joint": 52.5,
        "shoulder_roll_.*_joint": 52.5,
        "shoulder_yaw_.*_joint": 52.5,
        "elbow_pitch_.*_joint": 52.5,
    }
    assert arms.velocity_limit_sim == {
        "shoulder_pitch_.*_joint": 14.1,
        "shoulder_roll_.*_joint": 14.1,
        "shoulder_yaw_.*_joint": 14.1,
        "elbow_pitch_.*_joint": 14.1,
    }
    assert arms.stiffness == {
        "shoulder_pitch_.*_joint": 60,
        "shoulder_roll_.*_joint": 20,
        "shoulder_yaw_.*_joint": 10,
        "elbow_pitch_.*_joint": 10,
    }
    assert arms.damping == {
        "shoulder_pitch_.*_joint": 3,
        "shoulder_roll_.*_joint": 1.5,
        "shoulder_yaw_.*_joint": 1,
        "elbow_pitch_.*_joint": 1,
    }
