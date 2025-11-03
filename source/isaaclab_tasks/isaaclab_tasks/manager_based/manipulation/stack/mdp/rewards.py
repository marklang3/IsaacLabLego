# SPDX-License-Identifier: BSD-3-Clause
# Improved reward helpers for stacking with better shaping

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for lifting object above minimal_height (smooth Gaussian)."""
    obj: RigidObject = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    # Smooth Gaussian reward centered at minimal_height
    return torch.exp(-((z - minimal_height) ** 2) / (2 * 0.02**2))


def object_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Shaped reward for EE reaching the object using exponential."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj_pos - ee_pos, dim=-1)
    # Exponential provides better gradient than tanh
    return torch.exp(-dist / std)


def gripper_close_reward(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for closing gripper when near object."""
    robot: Articulation = env.scene[robot_cfg.name]
    # Get gripper joint positions (last 2 joints for Franka)
    gripper_pos = robot.data.joint_pos[:, -2:]
    # Reward smaller gripper opening (encourage closing)
    return 1.0 - torch.mean(gripper_pos / 0.04, dim=-1)


def object_grasped_reward(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Combined reward for successful grasp with better shaping."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Distance between object and EE
    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj_pos - ee_pos, dim=-1)
    
    # Gripper state (closed = low value)
    gripper_pos = robot.data.joint_pos[:, -2:].mean(dim=-1)
    
    # Object height
    obj_height = obj.data.root_pos_w[:, 2]
    initial_height = 0.0203  # Initial cube height
    
    # Distance-based reaching reward (always active)
    reach_reward = torch.exp(-dist / 0.1)  # More lenient distance scaling
    
    # Height difference between gripper and object
    height_diff = torch.abs(ee_pos[:, 2] - obj_pos[:, 2])
    height_reward = torch.exp(-height_diff / 0.05)  # Encourage matching height
    
    # XY alignment reward
    xy_dist = torch.norm(obj_pos[:, :2] - ee_pos[:, :2], dim=-1)
    xy_reward = torch.exp(-xy_dist / 0.08)  # Encourage XY alignment
    
    # More lenient initial conditions for attempting grasp
    near_object = (dist < 0.08).float()  # Larger threshold for initial approach
    above_object = (height_diff < 0.05).float()  # More forgiving height alignment
    xy_aligned = (xy_dist < 0.06).float()  # Wider XY alignment window
    
    # Gripper control logic with stronger early signals
    # - Close when approximately positioned
    # - Open when far away or poorly aligned
    gripper_should_close = near_object * above_object * xy_aligned
    
    # Progressive gripper reward
    basic_grasp_reward = torch.where(
        gripper_should_close > 0,
        2.0 * (1.0 - gripper_pos / 0.04),  # Stronger reward for closing when aligned
        0.5 * (gripper_pos / 0.04)  # Weaker reward for keeping open when not aligned
    )
    
    # Additional reward for precision once near
    precision_multiplier = torch.where(
        dist < 0.04,  # When very close
        2.0,  # Double the reward for precise positioning
        1.0
    )
    
    gripper_reward = basic_grasp_reward * precision_multiplier
    
    # Lifting reward - check if actually grasped by combining position and grasp
    properly_grasped = (dist < 0.04) & (gripper_pos < 0.01) & (height_diff < 0.03)
    lift_height = (obj_height - initial_height).clamp(min=0.0)
    lift_reward = torch.where(
        properly_grasped,
        2.0 * torch.exp(lift_height / 0.05),  # Stronger exponential reward for lifting
        torch.zeros_like(lift_height)
    )
    
    # Phase-based reward combination
    reach_phase = 0.4 * reach_reward  # Base reward for reaching
    align_phase = 0.3 * (height_reward + xy_reward)  # Reward for alignment
    grasp_phase = 0.5 * gripper_reward  # Increased reward for correct gripper action
    lift_phase = 3.0 * lift_reward  # Much higher reward for successful lifting
    
    return reach_phase + align_phase + grasp_phase + lift_phase


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # Return tanh-kernel reward: (num_envs,)
    return 1 - torch.tanh(object_ee_distance / std)

def horizontal_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    src_asset_cfg: SceneEntityCfg,
    tgt_asset_cfg: SceneEntityCfg,
    minimal_height: float = 0.04,  # Only reward alignment when src is lifted
) -> torch.Tensor:
    """Gaussian reward for XY center alignment between src and tgt.
    
    Only provides reward when src object is lifted above minimal_height to avoid
    encouraging pushing objects on the table.
    """
    src: RigidObject = env.scene[src_asset_cfg.name]
    tgt: RigidObject = env.scene[tgt_asset_cfg.name]
    dxy = src.data.root_pos_w[:, :2] - tgt.data.root_pos_w[:, :2]
    dist2 = (dxy * dxy).sum(-1)
    alignment_reward = torch.exp(-dist2 / (2.0 * std * std + 1e-9))
    
    # Only give alignment reward when src is lifted
    is_lifted = src.data.root_pos_w[:, 2] > minimal_height
    return torch.where(is_lifted, alignment_reward, torch.zeros_like(alignment_reward))


def action_rate_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 regularization on joint accelerations (approximated from velocities)."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel
    # Use velocity magnitude as a proxy for action rate
    return torch.mean(joint_vel * joint_vel, dim=-1)


def joint_vel_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """L2 regularization on joint velocities."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel
    return torch.mean(joint_vel * joint_vel, dim=-1)


def object_stability(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    lin_vel_thr: float = 0.05,
    ang_vel_thr: float = 0.2,
) -> torch.Tensor:
    """Encourage low linear & angular speeds."""
    obj: RigidObject = env.scene[object_cfg.name]
    lin_speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_speed = torch.linalg.norm(obj.data.root_ang_vel_w, dim=-1)
    lin_penalty = torch.clamp(lin_speed / lin_vel_thr, max=1.0)
    ang_penalty = torch.clamp(ang_speed / ang_vel_thr, max=1.0)
    return 1.0 - 0.5 * (lin_penalty + ang_penalty)


def vertical_distance_to_target(
    env: "ManagerBasedRLEnv",
    src_asset_cfg: SceneEntityCfg,
    tgt_asset_cfg: SceneEntityCfg,
    target_height: float = 0.0406,  # One block height
    std: float = 0.02,
) -> torch.Tensor:
    """Reward for placing src at target_height above tgt."""
    src: RigidObject = env.scene[src_asset_cfg.name]
    tgt: RigidObject = env.scene[tgt_asset_cfg.name]
    
    # Calculate height difference
    z_diff = src.data.root_pos_w[:, 2] - tgt.data.root_pos_w[:, 2]
    height_error = torch.abs(z_diff - target_height)
    
    return torch.exp(-height_error / std)