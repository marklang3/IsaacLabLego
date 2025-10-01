# SPDX-License-Identifier: BSD-3-Clause
# Reward functions for stacking / lifting tasks.
# All functions return (num_envs,) torch.Tensor and follow ManagerBasedRLEnv patterns.

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------------------
# Lift-style generic terms (added here so you can import all rewards from one place)
# --------------------------------------------------------------------------------------

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward when object's Z is above minimal_height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Shaped reward (1 - tanh(dist/std)) between EE and the object."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Object position (E, 3)
    obj_pos_w = obj.data.root_pos_w
    # EE target index 0: "end_effector" frame as configured in your FrameTransformer
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (E, 3)

    dist = torch.norm(obj_pos_w - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(dist / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Shaped reward for tracking a goal pose specified in the robot root frame.

    Equivalent to the lift implementation:
      - Read command (E, 7?) but only uses [:, :3] as target position in robot frame.
      - Transform to world using robot root pose.
      - Reward = lift_gate * (1 - tanh(||x_des_w - x_obj_w|| / std))
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Command sampled in robot/root frame (only use position)
    cmd = env.command_manager.get_command(command_name)
    des_pos_b = cmd[:, :3]  # (E, 3)

    # Convert desired pos from robot base frame -> world frame
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)

    dist = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    lifted_gate = (obj.data.root_pos_w[:, 2] > minimal_height).to(dist.dtype)
    return lifted_gate * (1.0 - torch.tanh(dist / std))


# --------------------------------------------------------------------------------------
# Stacking-specific terms
# --------------------------------------------------------------------------------------

def object_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """
    Heuristic grasp reward: gripper width <= (open_val - gripper_threshold).

    Notes:
      - Expects env.gripper_joint_names (regex list) and env.gripper_open_val to be set in your env cfg.
      - If joint names aren't exposed, falls back to the last two joints (common in Franka).
    """
    # If env doesn't expose these, return zeros to remain safe.
    if not hasattr(env, "gripper_joint_names"):
        return torch.zeros(env.num_envs, device=getattr(env, "device", "cpu"))

    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos  # (E, J)

    # Try using joint names if available
    if hasattr(robot.data, "joint_names") and robot.data.joint_names is not None:
        names = list(robot.data.joint_names)
        finger_idx = [i for i, n in enumerate(names) if "finger" in n]
        if len(finger_idx) >= 2:
            width = joint_pos[:, finger_idx].sum(-1)
        else:
            # Fallback to last two joints
            width = joint_pos[:, -2:].sum(-1)
    else:
        width = joint_pos[:, -2:].sum(-1)

    open_val = getattr(env, "gripper_open_val", 0.04)
    return (width <= (open_val - gripper_threshold)).float()


def horizontal_alignment(
    env: ManagerBasedRLEnv,
    src_asset_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    tgt_asset_cfg: SceneEntityCfg = SceneEntityCfg("base_object"),
    std: float = 0.06,
) -> torch.Tensor:
    """
    Gaussian reward over XY distance between src and tgt object centers:
      R = exp( -||d_xy||^2 / (2*std^2) )
    """
    src: RigidObject = env.scene[src_asset_cfg.name]
    tgt: RigidObject = env.scene[tgt_asset_cfg.name]

    dxy = src.data.root_pos_w[:, :2] - tgt.data.root_pos_w[:, :2]
    dist2 = (dxy * dxy).sum(-1)
    return torch.exp(-dist2 / (2.0 * std * std + 1e-9))


def object_stability(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
    lin_vel_thr: float = 0.05,
    ang_vel_thr: float = 0.2,
) -> torch.Tensor:
    """
    Reward for low linear and angular velocity:
      R = exp( - max(0, ||v||-lin_thr)/lin_thr - max(0, ||w||-ang_thr)/ang_thr )
    """
    obj: RigidObject = env.scene[object_cfg.name]
    lin_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_speed = torch.norm(obj.data.root_ang_vel_w, dim=-1)

    lin_excess = torch.clamp(lin_speed - lin_vel_thr, min=0.0) / max(1e-6, lin_vel_thr)
    ang_excess = torch.clamp(ang_speed - ang_vel_thr, min=0.0) / max(1e-6, ang_vel_thr)
    return torch.exp(-(lin_excess + ang_excess))


def stack_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("stack_object"),
) -> torch.Tensor:
    """
    Stacking variant of goal tracking (identical math to object_goal_distance but defaults the object to 'stack_object').
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cmd = env.command_manager.get_command(command_name)

    des_pos_b = cmd[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    dist = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)

    lifted_gate = (obj.data.root_pos_w[:, 2] > minimal_height).to(dist.dtype)
    return lifted_gate * (1.0 - torch.tanh(dist / std))