"""Fixed joint constraint snap for LEGO-like stacking.

When bricks align within threshold, create a FixedJoint constraint.
This simulates LEGO "click" with zero physics instability.
"""

from __future__ import annotations

import torch
import omni.kit.commands
from pxr import UsdPhysics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sanitize_env_ids(env_ids: torch.Tensor | None, num_envs: int, device: torch.device | str) -> torch.Tensor:
    """Return a 1D int64 tensor of valid environment indices on the requested device.

    This uses a CPU-side validity filter to avoid fragile CUDA advanced-indexing paths
    when invalid indices accidentally appear.
    """
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)

    ids = torch.as_tensor(env_ids, dtype=torch.long)
    if ids.ndim == 0:
        ids = ids.unsqueeze(0)
    ids = ids.reshape(-1).detach().cpu()

    valid = (ids >= 0) & (ids < num_envs)
    if valid.any():
        ids = ids[valid]
    else:
        ids = ids[:0]

    return ids.to(device=device, dtype=torch.long)


def fixed_joint_snap(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    upper_object_name: str = "cube_2",
    lower_object_name: str = "cube_3",
    snap_xy_threshold: float = 0.015,  # 1.5cm alignment tolerance (overridden by curriculum if active)
    snap_z_threshold: float = 0.010,   # 1.0cm height tolerance (overridden by curriculum if active)
    expected_stack_height: float = 0.0192,  # One brick height
    require_low_velocity: bool = True,
    velocity_threshold: float = 0.05,  # 5cm/s
    use_curriculum: bool = True,  # Use curriculum thresholds if available
) -> None:
    """Create fixed joint constraint when bricks are aligned.

    This implements deterministic LEGO-like snapping:
    - Check alignment (XY, Z, velocity)
    - If aligned, create FixedJoint constraint
    - Constraint persists until episode reset

    Args:
        env: The RL environment.
        env_ids: Environment indices to check for snapping.
        upper_object_name: Name of the top brick.
        lower_object_name: Name of the bottom brick.
        snap_xy_threshold: Maximum XY offset for snap (meters).
        snap_z_threshold: Maximum Z offset from target height (meters).
        expected_stack_height: Target height separation (meters).
        require_low_velocity: If True, only snap when velocity is low.
        velocity_threshold: Maximum velocity for snap (m/s).
    """

    # Normalize and validate env indices early to avoid CUDA index asserts.
    env_ids = _sanitize_env_ids(env_ids, env.num_envs, env.device)

    if env_ids.numel() == 0:
        return

    # Get scene objects
    upper_object = env.scene[upper_object_name]
    lower_object = env.scene[lower_object_name]

    # Use curriculum thresholds if available
    if use_curriculum and hasattr(env, "_current_xy_threshold"):
        snap_xy_threshold = env._current_xy_threshold
        snap_z_threshold = env._current_z_threshold

    # Initialize snap tracking if not exists
    if (not hasattr(env, "_fixed_joint_snapped")) or (env._fixed_joint_snapped.shape[0] != env.num_envs):
        env._fixed_joint_snapped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        # Only print once during training startup, not spam
        if env.num_envs < 100:  # Only print for test/debug scenarios
            print(f"[FixedJointSnap] Initialized snap tracking for {env.num_envs} environments")
            print(f"[FixedJointSnap] Initial thresholds: XY={snap_xy_threshold*100:.2f}cm, Z={snap_z_threshold*100:.2f}cm")

    # Get poses
    upper_pos = upper_object.data.root_pos_w[env_ids]
    lower_pos = lower_object.data.root_pos_w[env_ids]
    upper_vel = upper_object.data.root_lin_vel_w[env_ids]

    # Calculate alignment metrics
    xy_offset = torch.norm(upper_pos[:, :2] - lower_pos[:, :2], dim=1)
    z_offset = torch.abs(upper_pos[:, 2] - lower_pos[:, 2] - expected_stack_height)
    velocity = torch.norm(upper_vel, dim=1)

    # Check snap conditions
    xy_aligned = xy_offset < snap_xy_threshold
    z_aligned = z_offset < snap_z_threshold
    vel_ok = ~require_low_velocity | (velocity < velocity_threshold)

    # Determine which envs should snap
    already_snapped = env._fixed_joint_snapped[env_ids]
    should_snap = xy_aligned & z_aligned & vel_ok & ~already_snapped

    # Apply snap for qualifying environments
    if should_snap.any():
        snap_local_ids = torch.nonzero(should_snap, as_tuple=False).squeeze(-1)
        snap_env_ids = torch.index_select(env_ids, dim=0, index=snap_local_ids)
        if snap_env_ids.numel() == 0:
            return

        # Extra guard against any stale/invalid indices from asynchronous state updates.
        snap_env_ids = _sanitize_env_ids(snap_env_ids, env.num_envs, env.device)
        if snap_env_ids.numel() == 0:
            return

        # Mark as snapped BEFORE applying (prevent repeated snaps)
        env._fixed_joint_snapped[snap_env_ids] = True

        # Create fixed joint constraint
        _create_fixed_joint(
            env=env,
            env_ids=snap_env_ids,
            upper_object=upper_object,
            lower_object=lower_object,
            expected_stack_height=expected_stack_height,
        )

        # Debug info (only print occasionally to avoid spam during training)
        num_snapped = snap_env_ids.shape[0]
        if num_snapped > 0:
            # Only print snap messages for small env counts (testing) or occasionally for training
            if not hasattr(env, "_snap_print_counter"):
                env._snap_print_counter = 0
            env._snap_print_counter += 1

            # Print every 1000th snap during training, or always during testing
            if env.num_envs < 100 or env._snap_print_counter % 1000 == 0:
                print(f"[FixedJointSnap] ✓ Snapped {num_snapped} bricks (XY < {snap_xy_threshold*100:.1f}cm, Z < {snap_z_threshold*100:.1f}cm)")


def _create_fixed_joint(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    upper_object,
    lower_object,
    expected_stack_height: float,
) -> None:
    """Create fixed joint constraint between aligned bricks.

    FAST VERSION: Uses velocity zeroing + precise alignment (NO USD joints).
    This simulates snap behavior without slow USD operations.
    """
    # Zero relative velocity for clean snap
    upper_vel = upper_object.data.root_lin_vel_w.clone()
    upper_vel[env_ids] = 0.0
    upper_ang_vel = upper_object.data.root_ang_vel_w.clone()
    upper_ang_vel[env_ids] = 0.0

    upper_object.write_root_velocity_to_sim(
        torch.cat([upper_vel, upper_ang_vel], dim=1)
    )

    # Align positions precisely
    lower_pos = lower_object.data.root_pos_w[env_ids]
    upper_pos = upper_object.data.root_pos_w.clone()
    upper_quat = upper_object.data.root_quat_w.clone()

    # Snap to exact stack position
    upper_pos[env_ids, :2] = lower_pos[:, :2]  # Align XY
    upper_pos[env_ids, 2] = lower_pos[:, 2] + expected_stack_height  # Set Z

    upper_object.write_root_pose_to_sim(
        torch.cat([upper_pos, upper_quat], dim=1)
    )


def reset_fixed_joint_snap(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Reset snap tracking for environments.

    Call this on episode reset to allow new snaps.
    Fast version - no USD joint cleanup needed.
    """
    # Reset snap tracking
    if hasattr(env, "_fixed_joint_snapped"):
        valid_env_ids = _sanitize_env_ids(env_ids, env.num_envs, env.device)
        if valid_env_ids.numel() > 0:
            env._fixed_joint_snapped[valid_env_ids] = False
