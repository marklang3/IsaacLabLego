# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
LEGO 2-Cube Stack environment with RL rewards - COPIED FROM SMOOTH BLOCKS.

This config matches the working smooth blocks task exactly, only adjusting object sizes.
Smooth blocks: 50mm cubes
LEGO bricks: 28.8mm height (1.5x scale LEGO)
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import stack_2cube_env_cfg


@configclass
class RewardsCfg:
    """Reward terms - EXACT COPY of smooth blocks, just scaled for LEGO size."""

    # Stack cube_2 on cube_3 (IsaacGym combined reward)
    stack_cube_2_on_3 = RewTerm(
        func=mdp.isaacgym_combined_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.05,  # 50mm - EXACT match to smooth blocks (using their physics!)
            "lower_size": 0.05,
            "table_height": 0.0,  # IsaacLab table is at Z=0
            "lift_threshold": 0.01,  # 10mm - MATCH smooth blocks exactly
            "r_dist_scale": 0.1,   # Match smooth blocks
            "r_lift_scale": 1.5,   # Match smooth blocks
            "r_align_scale": 2.0,  # Match smooth blocks
            "r_stack_scale": 16.0, # Match smooth blocks
        },
        weight=1.0,
    )

    # Action penalties for smooth motion
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Intermediate reward for being close to stacked (helps bridge the gap)
    nearly_stacked = RewTerm(
        func=mdp.nearly_stacked_reward,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.05,  # 50mm - EXACT match to smooth blocks
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,  # Match smooth blocks
            "xy_threshold": 0.03,  # Match smooth blocks - 30mm tolerance
            "z_threshold": 0.03,   # Match smooth blocks - 30mm tolerance
        },
        weight=5.0,  # Match smooth blocks
    )

    # Penalty for moving too much when close to target (reduces hovering/shaking)
    stability = RewTerm(
        func=mdp.stability_penalty,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "xy_threshold": 0.03,  # Match smooth blocks
        },
        weight=-0.1,  # Match smooth blocks
    )


@configclass
class Franka2CubeStackLegoJointPosRLEnvCfg(stack_2cube_env_cfg.Franka2CubeStackEnvCfg):
    """LEGO 2-Cube Stack environment - CONFIG COPIED FROM SMOOTH BLOCKS."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Add rewards (EXACT COPY from smooth blocks)
        self.rewards = RewardsCfg()

        # Disable XR for headless RL training
        self.xr = None

        # Remove empty observation groups that cause RL training issues
        self.observations.policy.concatenate_terms = True
        if hasattr(self.observations, 'rgb_camera'):
            delattr(self.observations, 'rgb_camera')
        if hasattr(self.observations, 'subtask_terms'):
            delattr(self.observations, 'subtask_terms')
