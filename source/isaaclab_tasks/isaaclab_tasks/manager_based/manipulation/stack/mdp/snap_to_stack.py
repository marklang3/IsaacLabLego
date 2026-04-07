"""Snap-to-stack post-processing for LEGO precision stacking.

This module implements mechanical engagement simulation for LEGO bricks by detecting
when cubes are "close enough" and snapping them into proper alignment.
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def snap_to_stack_lego(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    cube_top_name: str = "cube_2",
    cube_bottom_name: str = "cube_3",
    snap_z_threshold: float = 0.06,
    snap_xy_threshold: float = 0.03,
    snap_z_target: float = 0.05,
    snap_vel_threshold: float = 0.05,
    vel_damping: float = 0.1,
    require_gripper_close: bool = True,
    gripper_distance_threshold: float = 0.15,
) -> None:
    """Apply snap-to-stack behavior when LEGO cubes are close and aligned.

    This simulates the mechanical engagement of LEGO knobs and cavities by:
    1. Detecting when top cube is close to bottom cube (Z < 6cm, XY < 3cm)
    2. **CRITICAL**: Only snap if robot gripper is actively manipulating the cube
    3. Snapping top cube to exactly 5cm above bottom cube
    4. Locking XY alignment to prevent sliding
    5. Dampening velocity to simulate friction

    Args:
        env: The environment instance.
        env_ids: Environment indices to apply snap to (unused - applies to all).
        cube_top_name: Name of the top cube object.
        cube_bottom_name: Name of the bottom cube object.
        snap_z_threshold: Maximum Z distance for snap (default: 0.06m = 6cm).
        snap_xy_threshold: Maximum XY offset for snap (default: 0.03m = 3cm).
        snap_z_target: Target Z height when snapped (default: 0.05m = 5cm).
        snap_vel_threshold: Maximum velocity for snap (default: 0.05m/s).
        vel_damping: Velocity reduction factor when snapped (default: 0.1 = 10%).
        require_gripper_close: Whether to require gripper near cube to snap (default: True).
        gripper_distance_threshold: Max distance from gripper to cube for snap (default: 0.15m).
    """

    # Get cube references
    cube_top = env.scene[cube_top_name]
    cube_bottom = env.scene[cube_bottom_name]

    # Get robot end-effector position to check if gripper is actively manipulating
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # Get end-effector position

    # Get positions and velocities
    top_pos = cube_top.data.root_pos_w
    bottom_pos = cube_bottom.data.root_pos_w
    top_vel = cube_top.data.root_lin_vel_w

    # Calculate relative position
    z_sep = top_pos[:, 2] - bottom_pos[:, 2]  # Z separation
    xy_offset = torch.norm(top_pos[:, :2] - bottom_pos[:, :2], dim=1)  # XY offset
    vel_mag = torch.norm(top_vel, dim=1)  # Velocity magnitude

    # Check if gripper is close to the top cube (actively manipulating)
    gripper_to_cube = torch.norm(ee_pos - top_pos, dim=1)
    gripper_engaging = gripper_to_cube < gripper_distance_threshold

    # Determine which environments should snap
    # CRITICAL: Only snap if gripper is actively manipulating the cube
    snap_mask = (
        (z_sep < snap_z_threshold) &
        (xy_offset < snap_xy_threshold) &
        (vel_mag < snap_vel_threshold)
    )

    if require_gripper_close:
        snap_mask = snap_mask & gripper_engaging

    if snap_mask.any():
        # Create target positions for snapped cubes
        snap_pos = bottom_pos.clone()
        snap_pos[:, 2] = bottom_pos[:, 2] + snap_z_target  # Set Z to exactly 5cm above

        # Apply snap only to environments that meet conditions
        new_pos = torch.where(snap_mask.unsqueeze(1), snap_pos, top_pos)

        # Create pose tensor (position + quaternion)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
        new_pose = torch.cat([new_pos, identity_quat], dim=1)

        # Apply snapped pose
        cube_top.write_root_pose_to_sim(new_pose)

        # Dampen velocity heavily to simulate engagement friction
        dampened_vel = top_vel * vel_damping
        dampened_vel = torch.where(snap_mask.unsqueeze(1), dampened_vel, top_vel)

        # Zero angular velocity for snapped cubes
        zero_ang_vel = torch.zeros((env.num_envs, 3), device=env.device)
        new_vel = torch.cat([dampened_vel, zero_ang_vel], dim=1)

        cube_top.write_root_velocity_to_sim(new_vel)


def snap_to_stack_lego_3cube_stage1(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    snap_z_threshold: float = 0.06,
    snap_xy_threshold: float = 0.03,
    snap_z_target: float = 0.05,
    snap_vel_threshold: float = 0.05,
    vel_damping: float = 0.1,
    require_gripper_close: bool = True,
    gripper_distance_threshold: float = 0.15,
) -> None:
    """Apply snap-to-stack for 3-cube LEGO: cube_2 on cube_3 (first stack)."""
    snap_to_stack_lego(
        env, env_ids,
        cube_top_name="cube_2",
        cube_bottom_name="cube_3",
        snap_z_threshold=snap_z_threshold,
        snap_xy_threshold=snap_xy_threshold,
        snap_z_target=snap_z_target,
        snap_vel_threshold=snap_vel_threshold,
        vel_damping=vel_damping,
        require_gripper_close=require_gripper_close,
        gripper_distance_threshold=gripper_distance_threshold,
    )


def snap_to_stack_lego_3cube_stage2(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    snap_z_threshold: float = 0.06,
    snap_xy_threshold: float = 0.03,
    snap_z_target: float = 0.05,
    snap_vel_threshold: float = 0.05,
    vel_damping: float = 0.1,
    require_gripper_close: bool = True,
    gripper_distance_threshold: float = 0.15,
) -> None:
    """Apply snap-to-stack for 3-cube LEGO: cube_1 on cube_2 (second stack)."""
    snap_to_stack_lego(
        env, env_ids,
        cube_top_name="cube_1",
        cube_bottom_name="cube_2",
        snap_z_threshold=snap_z_threshold,
        snap_xy_threshold=snap_xy_threshold,
        snap_z_target=snap_z_target,
        snap_vel_threshold=snap_vel_threshold,
        vel_damping=vel_damping,
        require_gripper_close=require_gripper_close,
        gripper_distance_threshold=gripper_distance_threshold,
    )


def snap_to_stack_lego_3cube_both(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    snap_z_threshold: float = 0.06,
    snap_xy_threshold: float = 0.03,
    snap_z_target: float = 0.05,
    snap_vel_threshold: float = 0.05,
    vel_damping: float = 0.1,
    require_gripper_close: bool = True,
    gripper_distance_threshold: float = 0.15,
) -> None:
    """Apply snap-to-stack for both stacks in 3-cube LEGO environment."""
    # First stack: cube_2 on cube_3
    snap_to_stack_lego(
        env, env_ids,
        cube_top_name="cube_2",
        cube_bottom_name="cube_3",
        snap_z_threshold=snap_z_threshold,
        snap_xy_threshold=snap_xy_threshold,
        snap_z_target=snap_z_target,
        snap_vel_threshold=snap_vel_threshold,
        vel_damping=vel_damping,
        require_gripper_close=require_gripper_close,
        gripper_distance_threshold=gripper_distance_threshold,
    )
    # Second stack: cube_1 on cube_2
    snap_to_stack_lego(
        env, env_ids,
        cube_top_name="cube_1",
        cube_bottom_name="cube_2",
        snap_z_threshold=snap_z_threshold,
        snap_xy_threshold=snap_xy_threshold,
        snap_z_target=snap_z_target,
        snap_vel_threshold=snap_vel_threshold,
        vel_damping=vel_damping,
        require_gripper_close=require_gripper_close,
        gripper_distance_threshold=gripper_distance_threshold,
    )
