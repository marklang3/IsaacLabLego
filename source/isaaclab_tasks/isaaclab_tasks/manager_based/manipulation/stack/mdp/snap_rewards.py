"""Rewards for snap mechanism stability and tracking.

Provides bonus rewards for maintaining stable snaps and tracks snap statistics.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def held_stable_bonus(
    env: ManagerBasedRLEnv,
    upper_object_name: str = "cube_2",
    velocity_threshold: float = 0.01,  # Very low velocity (1cm/s)
    stable_steps_required: int = 10,   # Must be stable for 10 steps
) -> torch.Tensor:
    """Reward for maintaining a stable snap.

    Gives bonus when:
    - Snap is active
    - Velocities are very low
    - Maintained for N consecutive steps

    This improves robustness and prevents accidental/unstable snaps.

    Args:
        env: The RL environment.
        upper_object_name: Name of the top brick.
        velocity_threshold: Maximum velocity to count as stable (m/s).
        stable_steps_required: Number of consecutive stable steps needed.

    Returns:
        Reward tensor of shape (num_envs,).
    """

    # Initialize tracking
    if not hasattr(env, "_stable_snap_counter"):
        env._stable_snap_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # Check if snapped
    snapped = getattr(env, "_fixed_joint_snapped", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    # Check velocity
    upper_object = env.scene[upper_object_name]
    velocity = torch.norm(upper_object.data.root_lin_vel_w, dim=1)
    is_stable = velocity < velocity_threshold

    # Update counter
    is_stable_and_snapped = snapped & is_stable
    env._stable_snap_counter = torch.where(
        is_stable_and_snapped,
        env._stable_snap_counter + 1,  # Increment counter
        torch.zeros_like(env._stable_snap_counter)  # Reset if not stable
    )

    # Give reward if maintained for required steps
    reward = (env._stable_snap_counter >= stable_steps_required).float()

    return reward


def snap_success_tracker(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Track and log snap success count (for monitoring).

    This doesn't give reward, just tracks statistics for logging.

    Returns:
        Tensor indicating which envs are currently snapped (for logging).
    """

    # Initialize tracking
    if not hasattr(env, "_snap_count_total"):
        env._snap_count_total = 0
        env._snap_count_this_log = 0
        env._log_interval = 100  # Log every 100 episodes

    # Get current snap status
    snapped = getattr(env, "_fixed_joint_snapped", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))

    # Count new snaps (for logging)
    if not hasattr(env, "_previous_snap_status"):
        env._previous_snap_status = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Detect new snaps (transition from False to True)
    new_snaps = snapped & ~env._previous_snap_status
    num_new_snaps = new_snaps.sum().item()

    if num_new_snaps > 0:
        env._snap_count_total += num_new_snaps
        env._snap_count_this_log += num_new_snaps

    # Log periodically
    if hasattr(env, "_episode_count"):
        env._episode_count = getattr(env, "_episode_count", 0) + (env_ids.numel() if hasattr(env, "env_ids") else 1)

        if env._episode_count % env._log_interval == 0:
            snap_rate = env._snap_count_this_log / env._log_interval if env._log_interval > 0 else 0
            print(f"[SnapTracker] Episodes {env._episode_count}: {env._snap_count_this_log} snaps in last {env._log_interval} episodes (rate: {snap_rate:.2f} snaps/episode, total: {env._snap_count_total})")
            env._snap_count_this_log = 0

    # Update previous status
    env._previous_snap_status = snapped.clone()

    # Return current snap status (can be logged as metric)
    return snapped.float()
