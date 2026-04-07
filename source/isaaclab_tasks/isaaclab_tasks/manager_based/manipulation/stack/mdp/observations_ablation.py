"""
Observation functions for ablation studies.
Each function removes specific observation components to test their importance.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_obs_2cube_no_relative(
    env: ManagerBasedRLEnv,
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    ABLATION 1: Remove relative position observations (9D removed).

    Object observations WITHOUT relative positions:
        cube_2 pos (3D),
        cube_2 quat (4D),
        cube_3 pos (3D),
        cube_3 quat (4D),

    Total: 14D (vs 23D in full)
    Full obs: 58D → Ablation: 49D
    """
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    return torch.cat(
        (
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
        ),
        dim=1,
    )


def object_obs_2cube_no_eef(
    env: ManagerBasedRLEnv,
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    ABLATION 2: Remove end-effector pose observations (7D removed via config).

    This ablation keeps object_obs_2cube the same (23D) but removes eef_pos and eef_quat
    from the observation config, testing if explicit end-effector tracking is necessary.

    Full obs: 58D → Ablation: 51D
    """
    # This function is same as full - the ablation happens in config by removing eef_pos/eef_quat terms
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube_2 = cube_2_pos_w - ee_pos_w
    gripper_to_cube_3 = cube_3_pos_w - ee_pos_w

    cube_2_to_3 = cube_2_pos_w - cube_3_pos_w

    return torch.cat(
        (
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
            gripper_to_cube_2,
            gripper_to_cube_3,
            cube_2_to_3,
        ),
        dim=1,
    )


def object_obs_2cube_minimal(
    env: ManagerBasedRLEnv,
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    ABLATION 3: Minimal observations - only cube poses, no relative positions.

    Absolute positions only:
        cube_2 pos (3D),
        cube_2 quat (4D),
        cube_3 pos (3D),
        cube_3 quat (4D),

    Combined with config removing eef_pos, eef_quat, gripper_pos.
    Total object obs: 14D
    Full obs with proprioception: 8 (actions) + 9 (joint_pos) + 9 (joint_vel) + 14 (objects) = 40D
    """
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    cube_2_pos_w = cube_2.data.root_pos_w
    cube_2_quat_w = cube_2.data.root_quat_w

    cube_3_pos_w = cube_3.data.root_pos_w
    cube_3_quat_w = cube_3.data.root_quat_w

    return torch.cat(
        (
            cube_2_pos_w - env.scene.env_origins,
            cube_2_quat_w,
            cube_3_pos_w - env.scene.env_origins,
            cube_3_quat_w,
        ),
        dim=1,
    )
