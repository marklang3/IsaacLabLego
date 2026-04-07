# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum learning scheduler for the stacking task."""

from __future__ import annotations


class StackingCurriculumScheduler:
    """
    Curriculum learning scheduler that progressively increases task difficulty.

    The scheduler defines three stages:
    - Stage 1 (epochs 0-499): Focus on grasping skills
    - Stage 2 (epochs 500-999): Focus on first stack (cube_2 on cube_3)
    - Stage 3 (epochs 1000+): Full task with all three cubes

    Each stage adjusts reward weights to emphasize different skills, allowing
    the policy to learn complex behaviors incrementally.
    """

    def __init__(self, total_epochs: int = 1500):
        """
        Initialize the curriculum scheduler.

        Args:
            total_epochs: Total number of training epochs
        """
        self.total_epochs = total_epochs
        self.current_stage = 1

        # Define stage boundaries
        self.stage_boundaries = {
            1: (0, 499),      # Grasp-focused stage
            2: (500, 999),    # First stack stage
            3: (1000, total_epochs),  # Full task stage
        }

    def get_stage(self, current_epoch: int) -> int:
        """
        Determine current curriculum stage based on epoch.

        Args:
            current_epoch: Current training epoch

        Returns:
            Current curriculum stage (1, 2, or 3)
        """
        for stage, (start, end) in self.stage_boundaries.items():
            if start <= current_epoch <= end:
                if stage != self.current_stage:
                    self.current_stage = stage
                    print(f"\n{'='*80}")
                    print(f"CURRICULUM: Transitioning to Stage {stage} at epoch {current_epoch}")
                    print(f"{'='*80}\n")
                return stage
        return 3  # Default to final stage

    def get_reward_weights(self, current_epoch: int) -> dict[str, float]:
        """
        Get reward weights for current curriculum stage.

        Args:
            current_epoch: Current training epoch

        Returns:
            Dictionary mapping reward term names to weights
        """
        stage = self.get_stage(current_epoch)

        if stage == 1:
            # Stage 1: Focus on grasping cube_2
            # Emphasize reaching, gripper closing, and grasping
            # Minimal emphasis on stacking or second cube
            return {
                # Cube 2 grasping (primary focus)
                'reaching_cube_2': 2.0,
                'gripper_close_near_cube_2': 15.0,  # High weight to encourage exploration
                'grasping_cube_2': 40.0,             # Very high weight for success
                'lifting_cube_2': 10.0,              # Secondary objective

                # Cube 2 stacking (minimal, just for direction)
                'aligning_cube_2_over_cube_3_xy': 1.0,
                'cube_2_above_cube_3': 1.0,
                'stacking_cube_2_on_cube_3': 5.0,

                # Cube 1 (completely disabled)
                'reaching_cube_1': 0.0,
                'gripper_close_near_cube_1': 0.0,
                'grasping_cube_1': 0.0,
                'lifting_cube_1': 0.0,
                'aligning_cube_1_over_cube_2_xy': 0.0,
                'cube_1_above_cube_2': 0.0,
                'stacking_cube_1_on_cube_2': 0.0,

                # Auxiliary rewards
                'orientation_alignment_2_3': 0.5,
                'orientation_alignment_1_2': 0.0,
                'stacking_progress': 5.0,
                'action_rate': -0.0001,
                'joint_vel': -0.0001,
            }

        elif stage == 2:
            # Stage 2: Focus on stacking cube_2 on cube_3
            # Reduce grasping emphasis, increase stacking rewards
            # Still disable cube_1 to focus on first stack
            return {
                # Cube 2 grasping (reduced emphasis, already learned)
                'reaching_cube_2': 1.0,
                'gripper_close_near_cube_2': 5.0,
                'grasping_cube_2': 10.0,
                'lifting_cube_2': 10.0,

                # Cube 2 stacking (primary focus)
                'aligning_cube_2_over_cube_3_xy': 15.0,  # High weight for alignment
                'cube_2_above_cube_3': 10.0,
                'stacking_cube_2_on_cube_3': 60.0,       # Very high reward for stacking

                # Cube 1 (still disabled)
                'reaching_cube_1': 0.0,
                'gripper_close_near_cube_1': 0.0,
                'grasping_cube_1': 0.0,
                'lifting_cube_1': 0.0,
                'aligning_cube_1_over_cube_2_xy': 0.0,
                'cube_1_above_cube_2': 0.0,
                'stacking_cube_1_on_cube_2': 0.0,

                # Auxiliary rewards
                'orientation_alignment_2_3': 2.0,
                'orientation_alignment_1_2': 0.0,
                'stacking_progress': 15.0,  # Increased progress tracking
                'action_rate': -0.0001,
                'joint_vel': -0.0001,
            }

        else:  # stage == 3
            # Stage 3: Full task with all three cubes
            # Balanced rewards across both stacking operations
            return {
                # Cube 2 operations (balanced)
                'reaching_cube_2': 1.0,
                'gripper_close_near_cube_2': 3.0,
                'grasping_cube_2': 10.0,
                'lifting_cube_2': 10.0,
                'aligning_cube_2_over_cube_3_xy': 3.0,
                'cube_2_above_cube_3': 5.0,
                'stacking_cube_2_on_cube_3': 20.0,

                # Cube 1 operations (now enabled)
                'reaching_cube_1': 1.0,
                'gripper_close_near_cube_1': 3.0,
                'grasping_cube_1': 10.0,
                'lifting_cube_1': 10.0,
                'aligning_cube_1_over_cube_2_xy': 3.0,
                'cube_1_above_cube_2': 5.0,
                'stacking_cube_1_on_cube_2': 20.0,

                # Auxiliary rewards (full emphasis)
                'orientation_alignment_2_3': 2.0,
                'orientation_alignment_1_2': 2.0,
                'stacking_progress': 15.0,
                'action_rate': -0.0001,
                'joint_vel': -0.0001,
            }

    def get_stage_description(self, stage: int) -> str:
        """
        Get human-readable description of curriculum stage.

        Args:
            stage: Curriculum stage number

        Returns:
            Description of stage objectives
        """
        descriptions = {
            1: "Stage 1: Grasp Learning - Focus on reaching and grasping cube_2",
            2: "Stage 2: First Stack - Focus on stacking cube_2 on cube_3",
            3: "Stage 3: Full Task - Complete three-cube stacking",
        }
        return descriptions.get(stage, "Unknown stage")

    def should_log_transition(self, current_epoch: int) -> bool:
        """
        Check if current epoch is a stage transition point.

        Args:
            current_epoch: Current training epoch

        Returns:
            True if this is a stage transition epoch
        """
        return current_epoch in [500, 1000]
