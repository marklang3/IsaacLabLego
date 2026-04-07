#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Wrapper script for training with curriculum learning.

This script wraps the standard RL Games training to inject curriculum weight updates.
"""

import gymnasium as gym
import torch
from omni.isaac.lab_tasks.utils import parse_env_cfg


def create_curriculum_callback(env_cfg):
    """
    Create a callback function that updates curriculum rewards each epoch.

    Args:
        env_cfg: Environment configuration with curriculum scheduler

    Returns:
        Callback function for RL Games
    """
    def on_epoch_end(runner, epoch):
        """Called at the end of each training epoch."""
        if hasattr(env_cfg, 'update_curriculum_rewards'):
            env_cfg.update_curriculum_rewards(epoch)

            # Log stage transitions
            if hasattr(env_cfg, 'curriculum_scheduler') and env_cfg.curriculum_scheduler:
                if env_cfg.curriculum_scheduler.should_log_transition(epoch):
                    stage = env_cfg.curriculum_scheduler.get_stage(epoch)
                    description = env_cfg.curriculum_scheduler.get_stage_description(stage)
                    print(f"\n{'='*80}")
                    print(f"CURRICULUM TRANSITION: {description}")
                    print(f"{'='*80}\n")

    return on_epoch_end


# Note: This is a helper module. The actual training should still use:
# ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Stack-Cube-Franka-v0
#
# The curriculum will automatically activate if use_curriculum=True in the config.
