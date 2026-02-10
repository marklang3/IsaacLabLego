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


def gripper_close_when_near_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float = 0.08,
) -> torch.Tensor:
    """Reward the agent for closing gripper when near an object (encourages grasping exploration)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    # Get gripper state
    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

        # Check if gripper is closed
        gripper_closed = torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[0]]
            - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
        ) > env.cfg.gripper_threshold

        gripper_closed = torch.logical_and(
            gripper_closed,
            torch.abs(
                robot.data.joint_pos[:, gripper_joint_ids[1]]
                - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            ) > env.cfg.gripper_threshold,
        ).float()

        # Reward closing gripper when near object
        near_object = (pose_diff < distance_threshold).float()
        return near_object * gripper_closed
    else:
        return torch.zeros(env.num_envs, device=env.device)


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


# ==============================================================================
# IsaacGym-Style Reward Functions (Proven to work for cube stacking)
# ==============================================================================

def isaacgym_distance_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Distance reward following IsaacGym's approach:
    - Uses distance from hand center + left finger + right finger
    - Uses tanh for smooth, bounded gradients
    - Encourages fingertips to wrap around object
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Object position
    object_pos = object.data.root_pos_w

    # End-effector (hand center) position - frame 0
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    d_hand = torch.norm(object_pos - ee_pos, dim=-1)

    # Left fingertip position - frame 1
    left_finger_pos = ee_frame.data.target_pos_w[:, 1, :]
    d_left = torch.norm(object_pos - left_finger_pos, dim=-1)

    # Right fingertip position - frame 2
    right_finger_pos = ee_frame.data.target_pos_w[:, 2, :]
    d_right = torch.norm(object_pos - right_finger_pos, dim=-1)

    # Average distance from all three points
    d_avg = (d_hand + d_left + d_right) / 3.0

    # Tanh kernel for smooth, bounded reward (max = 1.0)
    return 1.0 - torch.tanh(10.0 * d_avg)


def isaacgym_lift_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_size: float,
    table_height: float = 1.025,
    lift_threshold: float = 0.04,
) -> torch.Tensor:
    """
    Lift reward following IsaacGym's approach:
    - Binary reward for lifting above threshold
    - Threshold is relative to object size and table height
    """
    object: RigidObject = env.scene[object_cfg.name]

    # Height of object above table
    object_height = object.data.root_pos_w[:, 2] - table_height

    # Lifted if object bottom is above threshold
    lifted = (object_height - object_size / 2.0) > lift_threshold

    return lifted.float()


def isaacgym_align_reward(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    upper_size: float,
    lower_size: float,
    table_height: float = 1.025,
    lift_threshold: float = 0.04,
) -> torch.Tensor:
    """
    Alignment reward following IsaacGym's approach:
    - Only active if upper object is lifted
    - Measures distance between objects with height offset
    - Uses tanh for smooth gradients
    """
    upper_obj: RigidObject = env.scene[upper_object_cfg.name]
    lower_obj: RigidObject = env.scene[lower_object_cfg.name]

    # Check if upper object is lifted
    upper_height = upper_obj.data.root_pos_w[:, 2] - table_height
    upper_lifted = (upper_height - upper_size / 2.0) > lift_threshold

    # Compute target position (lower cube center + offset for stacking)
    offset = torch.zeros_like(upper_obj.data.root_pos_w)
    offset[:, 2] = (upper_size + lower_size) / 2.0

    # Distance from upper cube to target position above lower cube
    d_align = torch.norm(
        upper_obj.data.root_pos_w - (lower_obj.data.root_pos_w + offset),
        dim=-1
    )

    # Alignment reward only active if lifted
    align_reward = (1.0 - torch.tanh(10.0 * d_align)) * upper_lifted.float()

    return align_reward


def isaacgym_stack_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    upper_size: float,
    lower_size: float,
    xy_threshold: float = 0.02,  # Match IsaacGym exactly (was 0.03 for larger bricks, reverting to match original)
    z_threshold: float = 0.02,    # Match IsaacGym exactly (was 0.03 for larger bricks, reverting to match original)
    gripper_away_threshold: float = 0.04,
) -> torch.Tensor:
    """
    Stack success reward following IsaacGym's approach:
    - XY alignment check
    - Height check (cube at correct stacking height)
    - Gripper must be away from cube (ensures stability)
    - Binary reward (all conditions must be true)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    upper_obj: RigidObject = env.scene[upper_object_cfg.name]
    lower_obj: RigidObject = env.scene[lower_object_cfg.name]

    # 1. XY alignment check
    xy_dist = torch.norm(
        upper_obj.data.root_pos_w[:, :2] - lower_obj.data.root_pos_w[:, :2],
        dim=-1
    )
    xy_aligned = xy_dist < xy_threshold

    # 2. Height check (upper cube should be at lower_cube_top + upper_cube_half)
    target_height = lower_obj.data.root_pos_w[:, 2] + lower_size / 2.0 + upper_size / 2.0
    height_diff = torch.abs(upper_obj.data.root_pos_w[:, 2] - target_height)
    at_correct_height = height_diff < z_threshold

    # 3. Gripper away from upper cube (ensures cube is stable, not held)
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    gripper_dist = torch.norm(upper_obj.data.root_pos_w - ee_pos, dim=-1)
    gripper_away = gripper_dist > gripper_away_threshold

    # All three conditions must be true
    stacked = xy_aligned & at_correct_height & gripper_away

    return stacked.float()


def isaacgym_combined_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    upper_size: float,
    lower_size: float,
    table_height: float = 1.025,
    lift_threshold: float = 0.04,
    r_dist_scale: float = 1.0,
    r_lift_scale: float = 1.0,
    r_align_scale: float = 1.0,
    r_stack_scale: float = 2.0,
) -> torch.Tensor:
    """
    Combined reward following IsaacGym's exact approach:
    - Compute all 4 sub-rewards
    - dist_reward = max(distance, alignment) - smooth transition
    - Final reward = stack_reward IF stacked ELSE (dist + lift + align)
    """
    # Compute sub-rewards
    reach_r = isaacgym_distance_reward(env, upper_object_cfg, ee_frame_cfg)
    lift_r = isaacgym_lift_reward(env, upper_object_cfg, upper_size, table_height, lift_threshold)
    align_r = isaacgym_align_reward(
        env, upper_object_cfg, lower_object_cfg,
        upper_size, lower_size, table_height, lift_threshold
    )
    stack_r = isaacgym_stack_reward(
        env, robot_cfg, ee_frame_cfg, upper_object_cfg, lower_object_cfg,
        upper_size, lower_size
    )

    # Either/or composition matching IsaacGym exactly:
    # R = max(R_stack, R_align + R_lift + R_reach)
    reward = torch.where(
        stack_r > 0.0,
        r_stack_scale * stack_r,
        r_dist_scale * reach_r + r_lift_scale * lift_r + r_align_scale * align_r
    )

    return reward


def nearly_stacked_reward(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    upper_size: float,
    lower_size: float,
    table_height: float = 0.0,
    lift_threshold: float = 0.01,
    xy_threshold: float = 0.03,  # Looser than full stack (0.02)
    z_threshold: float = 0.03,
) -> torch.Tensor:
    """
    Intermediate reward for being close to successful stack.
    Bridges the gap between alignment (2.0) and full stack (16.0).
    Helps policy learn "almost there" state for precision stacking.
    """
    upper_obj: RigidObject = env.scene[upper_object_cfg.name]
    lower_obj: RigidObject = env.scene[lower_object_cfg.name]

    # Check if upper object is lifted
    upper_height = upper_obj.data.root_pos_w[:, 2] - table_height
    upper_lifted = (upper_height - upper_size / 2.0) > lift_threshold

    # XY alignment check (looser threshold)
    xy_dist = torch.norm(
        upper_obj.data.root_pos_w[:, :2] - lower_obj.data.root_pos_w[:, :2],
        dim=-1
    )
    xy_close = xy_dist < xy_threshold

    # Height check (looser threshold)
    target_height = lower_obj.data.root_pos_w[:, 2] + lower_size / 2.0 + upper_size / 2.0
    height_diff = torch.abs(upper_obj.data.root_pos_w[:, 2] - target_height)
    at_close_height = height_diff < z_threshold

    # Reward for being close (but not perfectly stacked yet)
    nearly_there = xy_close & at_close_height & upper_lifted

    return nearly_there.float()


def stability_penalty(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.03,  # Only penalize when close
) -> torch.Tensor:
    """
    Penalizes excessive object velocity when objects are close to stacking position.
    Reduces hovering/shaking behavior by encouraging stillness when near target.
    Critical for precision stacking tasks.
    """
    upper_obj: RigidObject = env.scene[upper_object_cfg.name]
    lower_obj: RigidObject = env.scene[lower_object_cfg.name]

    # Check if objects are close
    xy_dist = torch.norm(
        upper_obj.data.root_pos_w[:, :2] - lower_obj.data.root_pos_w[:, :2],
        dim=-1
    )
    close_to_target = xy_dist < xy_threshold

    # Compute velocity magnitude
    velocity = torch.norm(upper_obj.data.root_lin_vel_w, dim=-1)

    # Only apply penalty when close to target
    penalty = velocity * close_to_target.float()

    return penalty


def knob_cavity_alignment_reward(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    upper_size: float,
    lower_size: float,
    table_height: float = 0.0,
    lift_threshold: float = 0.01,
    xy_threshold: float = 0.015,  # Tighter than smooth cubes (0.02) - LEGO studs require precision
    rotation_threshold: float = 0.2,  # Radians - ~11 degrees tolerance for knob alignment
) -> torch.Tensor:
    """
    LEGO-specific reward for knob-cavity alignment.
    Requires both positional AND rotational alignment for knobs to fit into cavities.
    This is stricter than smooth cube alignment due to geometric constraints.
    
    LEGO Duplo brick constraints:
    - Knobs must align with cavities (requires XY precision within ~1.5cm)
    - Rotation must be within ~11 degrees (0.2 rad) for knobs to fit
    - Height must be precise for knobs to engage with cavities
    """
    upper_obj: RigidObject = env.scene[upper_object_cfg.name]
    lower_obj: RigidObject = env.scene[lower_object_cfg.name]

    # Check if upper object is lifted
    upper_height = upper_obj.data.root_pos_w[:, 2] - table_height
    upper_lifted = (upper_height - upper_size / 2.0) > lift_threshold

    # XY alignment check (tighter for LEGO knobs)
    xy_dist = torch.norm(
        upper_obj.data.root_pos_w[:, :2] - lower_obj.data.root_pos_w[:, :2],
        dim=-1
    )
    xy_aligned = xy_dist < xy_threshold

    # Height check
    target_height = lower_obj.data.root_pos_w[:, 2] + lower_size / 2.0 + upper_size / 2.0
    height_diff = torch.abs(upper_obj.data.root_pos_w[:, 2] - target_height)
    at_correct_height = height_diff < 0.02  # 2cm tolerance

    # Rotation alignment check (CRITICAL for LEGO knobs!)
    # Extract yaw (Z-axis rotation) from quaternions
    upper_quat = upper_obj.data.root_quat_w  # (N, 4) - [w, x, y, z]
    lower_quat = lower_obj.data.root_quat_w

    # Compute relative rotation (simplified yaw difference)
    # For LEGO, we primarily care about yaw (rotation around Z)
    # Convert quaternion to yaw: yaw = atan2(2(w*z + x*y), 1 - 2(y^2 + z^2))
    upper_yaw = torch.atan2(
        2 * (upper_quat[:, 0] * upper_quat[:, 3] + upper_quat[:, 1] * upper_quat[:, 2]),
        1 - 2 * (upper_quat[:, 2]**2 + upper_quat[:, 3]**2)
    )
    lower_yaw = torch.atan2(
        2 * (lower_quat[:, 0] * lower_quat[:, 3] + lower_quat[:, 1] * lower_quat[:, 2]),
        1 - 2 * (lower_quat[:, 2]**2 + lower_quat[:, 3]**2)
    )
    
    # Compute yaw difference (wrapped to [-pi, pi])
    yaw_diff = torch.abs(upper_yaw - lower_yaw)
    yaw_diff = torch.min(yaw_diff, 2 * torch.pi - yaw_diff)  # Handle wrapping
    
    rotation_aligned = yaw_diff < rotation_threshold

    # All conditions must be true for knob-cavity alignment
    knob_aligned = xy_aligned & at_correct_height & rotation_aligned & upper_lifted

    # Provide smooth gradient as we approach alignment
    # This helps policy learn fine-tuning
    alignment_quality = (1.0 - xy_dist / xy_threshold) * (1.0 - yaw_diff / rotation_threshold)
    alignment_quality = torch.clamp(alignment_quality, 0.0, 1.0) * upper_lifted.float()

    # Return binary success + smooth gradient
    return knob_aligned.float() + 0.5 * alignment_quality
