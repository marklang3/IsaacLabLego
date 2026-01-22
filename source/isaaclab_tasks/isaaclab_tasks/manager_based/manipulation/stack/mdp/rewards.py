# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching a specific object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Target object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward the agent for lifting an object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Reward the agent for grasping a specific object."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    # Check if gripper is closed and near object
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32).squeeze()
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold).float()
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Reward only supports parallel gripper for now"

            grasped = torch.logical_and(
                pose_diff < diff_threshold,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            )
            grasped = torch.logical_and(
                grasped,
                torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                )
                > env.cfg.gripper_threshold,
            ).float()

    return grasped


def object_goal_distance_xy(
    env: ManagerBasedRLEnv,
    std: float,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the agent for aligning the upper object above the lower object in the xy-plane."""
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    # Calculate xy distance between objects
    pos_diff = upper_object.data.root_pos_w[:, :2] - lower_object.data.root_pos_w[:, :2]
    xy_distance = torch.norm(pos_diff, dim=1)

    return 1 - torch.tanh(xy_distance / std)


def object_above_object(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the agent for positioning an object above another object at the proper height."""
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    # Check if upper object is above lower object at minimal height
    height_diff = upper_object.data.root_pos_w[:, 2] - lower_object.data.root_pos_w[:, 2]
    return torch.where(height_diff > minimal_height, 1.0, 0.0)


def object_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Reward the agent for successfully stacking an object on another."""
    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)

    # Check if gripper is open
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32).squeeze()
        stacked = torch.logical_and(suction_cup_is_open, stacked).float()
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Reward only supports parallel gripper for now"

            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            )
            stacked = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=1e-4,
                    rtol=1e-4,
                ),
                stacked,
            ).float()

    return stacked


def object_orientation_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the agent for aligning the orientation of two objects (important for lego brick studs)."""
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    # Get quaternions
    upper_quat = upper_object.data.root_quat_w
    lower_quat = lower_object.data.root_quat_w

    # Calculate the dot product between quaternions (measure of alignment)
    # For perfectly aligned quaternions, |dot product| = 1
    quat_dot = torch.abs(torch.sum(upper_quat * lower_quat, dim=1))

    # Convert to angular difference (0 to pi)
    angular_diff = 2 * torch.acos(torch.clamp(quat_dot, -1.0, 1.0))

    return 1 - torch.tanh(angular_diff / std)


def stacking_progress(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
) -> torch.Tensor:
    """Reward the agent based on overall stacking progress (how many cubes are stacked)."""
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    # Check if cube_2 is on cube_3 (first stack)
    pos_diff_23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w
    height_dist_23 = torch.abs(pos_diff_23[:, 2] - 0.0468)
    xy_dist_23 = torch.norm(pos_diff_23[:, :2], dim=1)
    stack_23 = torch.logical_and(xy_dist_23 < 0.05, height_dist_23 < 0.01).float()

    # Check if cube_1 is on cube_2 (second stack)
    pos_diff_12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
    height_dist_12 = torch.abs(pos_diff_12[:, 2] - 0.0468)
    xy_dist_12 = torch.norm(pos_diff_12[:, :2], dim=1)
    stack_12 = torch.logical_and(xy_dist_12 < 0.05, height_dist_12 < 0.01).float()

    # Progress: 0 (nothing stacked), 1 (first stack), 2 (both stacks)
    progress = stack_23 + stack_12

    return progress


def gripper_contact_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize excessive contact forces (useful for lego bricks with studs to prevent damage)."""
    # This is a placeholder - actual implementation would require contact sensor data
    # For now, we'll use a simpler proxy: penalize if gripper is closed but not grasping properly
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # For suction grippers, no contact penalty needed
        return torch.zeros(env.num_envs, device=env.device)
    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            # Penalize if gripper is closed but moving fast (might indicate slipping/collision)
            gripper_vel = torch.abs(robot.data.joint_vel[:, gripper_joint_ids]).sum(dim=1)
            gripper_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
            gripper_closed = torch.abs(
                gripper_pos - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            ) > env.cfg.gripper_threshold

            return -torch.where(torch.logical_and(gripper_closed, gripper_vel > contact_threshold), 1.0, 0.0)
        else:
            return torch.zeros(env.num_envs, device=env.device)
