"""Curriculum manager for snap thresholds.

Gradually tightens snap thresholds as training progresses to reduce early frustration.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def update_snap_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    initial_xy_threshold: float = 0.025,  # Start at 2.5cm
    final_xy_threshold: float = 0.015,    # End at 1.5cm
    initial_z_threshold: float = 0.015,   # Start at 1.5cm
    final_z_threshold: float = 0.010,     # End at 1.0cm
    curriculum_length: int = 5000,        # Transition over 5000 epochs
) -> None:
    """Update snap thresholds based on training progress.

    Args:
        env: The RL environment.
        env_ids: Environment indices (not used, but required by interface).
        initial_xy_threshold: Starting XY snap threshold (meters).
        final_xy_threshold: Final XY snap threshold (meters).
        initial_z_threshold: Starting Z snap threshold (meters).
        final_z_threshold: Final Z snap threshold (meters).
        curriculum_length: Number of epochs to transition over.
    """

    # Initialize curriculum tracking
    if not hasattr(env, "_curriculum_epoch"):
        env._curriculum_epoch = 0
        env._current_xy_threshold = initial_xy_threshold
        env._current_z_threshold = initial_z_threshold
        print(f"[SnapCurriculum] Initialized: XY={initial_xy_threshold*100:.1f}cm → {final_xy_threshold*100:.1f}cm, Z={initial_z_threshold*100:.1f}cm → {final_z_threshold*100:.1f}cm over {curriculum_length} epochs")

    # Increment epoch (called once per reset)
    env._curriculum_epoch += 1

    # Calculate curriculum progress (0.0 to 1.0)
    progress = min(1.0, env._curriculum_epoch / curriculum_length)

    # Linear interpolation
    env._current_xy_threshold = initial_xy_threshold + progress * (final_xy_threshold - initial_xy_threshold)
    env._current_z_threshold = initial_z_threshold + progress * (final_z_threshold - initial_z_threshold)

    # Log progress at milestones
    if env._curriculum_epoch in [1, 100, 500, 1000, 2000, curriculum_length]:
        print(f"[SnapCurriculum] Epoch {env._curriculum_epoch}: XY={env._current_xy_threshold*100:.2f}cm, Z={env._current_z_threshold*100:.2f}cm (progress: {progress*100:.1f}%)")


def get_current_snap_thresholds(env: ManagerBasedRLEnv) -> tuple[float, float]:
    """Get current snap thresholds from curriculum.

    Returns:
        Tuple of (xy_threshold, z_threshold) in meters.
    """
    if not hasattr(env, "_current_xy_threshold"):
        # Not initialized yet, return defaults
        return 0.025, 0.015

    return env._current_xy_threshold, env._current_z_threshold
