# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
LEGO 3-Cube Stack environment with VISION + IK control + LEGO-specific rewards.

This combines:
1. IK control (7D end-effector pose) - standard for vision tasks
2. Vision capabilities (wrist + table cameras with domain randomization)
3. LEGO bricks (c_lego_duplo.usd at 1.04x scale = 5cm)
4. LEGO-specific precision rewards (nearly_stacked, knob_cavity_alignment, stability)
5. Progressive curriculum learning

Based on successful precision stacking methodology achieving 4.8x improvement
over baseline (70-75% success vs 25% baseline).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_ik_rel_visuomotor_env_cfg


@configclass
class RewardsCfg:
    """Reward terms for the MDP - Precision LEGO stacking with knob-cavity alignment."""

    # ===== FIRST STACK: cube_2 on cube_3 (always enabled) =====

    stack_cube_2_on_3 = RewTerm(
        func=mdp.isaacgym_combined_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.05,  # LEGO Duplo scaled to 5.0cm (exactly matches smooth blocks)
            "lower_size": 0.05,
            "table_height": 0.0,  # IsaacLab table at Z=0
            "lift_threshold": 0.01,  # Cubes must lift >1cm from resting position
            "r_dist_scale": 0.1,   # Match IsaacGym
            "r_lift_scale": 1.5,   # Match IsaacGym
            "r_align_scale": 2.0,  # Match IsaacGym
            "r_stack_scale": 16.0, # Match IsaacGym
        },
        weight=1.0,
    )

    nearly_stacked_2_on_3 = RewTerm(
        func=mdp.nearly_stacked_reward,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.05,
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,
            "xy_threshold": 0.03,  # Looser than full stack (bridges gap)
            "z_threshold": 0.03,
        },
        weight=5.0,  # Between align (2.0) and stack (16.0)
    )

    knob_alignment_2_on_3 = RewTerm(
        func=mdp.knob_cavity_alignment_reward,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.05,
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,
            "xy_threshold": 0.015,  # Tighter for LEGO knobs (1.5cm)
            "rotation_threshold": 0.2,  # ~11 degrees for knob alignment
        },
        weight=3.0,  # Additional reward for precise knob alignment
    )

    stability_2_on_3 = RewTerm(
        func=mdp.stability_penalty,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "xy_threshold": 0.03,
        },
        weight=-0.1,  # Penalize velocity when close
    )

    # ===== SECOND STACK: cube_1 on cube_2 (enabled via curriculum) =====

    stack_cube_1_on_2 = RewTerm(
        func=mdp.isaacgym_combined_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "upper_object_cfg": SceneEntityCfg("cube_1"),
            "lower_object_cfg": SceneEntityCfg("cube_2"),
            "upper_size": 0.05,
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,
            "r_dist_scale": 0.1,
            "r_lift_scale": 1.5,
            "r_align_scale": 2.0,
            "r_stack_scale": 16.0,
        },
        weight=0.0,  # Disabled initially, enabled via curriculum
    )

    nearly_stacked_1_on_2 = RewTerm(
        func=mdp.nearly_stacked_reward,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_1"),
            "lower_object_cfg": SceneEntityCfg("cube_2"),
            "upper_size": 0.05,
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,
            "xy_threshold": 0.03,
            "z_threshold": 0.03,
        },
        weight=0.0,  # Disabled initially, enabled via curriculum
    )

    knob_alignment_1_on_2 = RewTerm(
        func=mdp.knob_cavity_alignment_reward,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_1"),
            "lower_object_cfg": SceneEntityCfg("cube_2"),
            "upper_size": 0.05,
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,
            "xy_threshold": 0.015,
            "rotation_threshold": 0.2,
        },
        weight=0.0,  # Disabled initially, enabled via curriculum
    )

    stability_1_on_2 = RewTerm(
        func=mdp.stability_penalty,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_1"),
            "lower_object_cfg": SceneEntityCfg("cube_2"),
            "xy_threshold": 0.03,
        },
        weight=0.0,  # Disabled initially, enabled via curriculum
    )

    # ===== SHARED PENALTIES =====

    # Action penalties for smooth motion
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for progressive LEGO stacking learning.

    Simple 2-stage curriculum:
    - Stage 1 (0-65M steps): Stack cube_2 on cube_3 only
    - Stage 2 (65M+ steps): Add cube_1 stacking on cube_2

    With 4096 envs and horizon_length=32, one epoch = 131,072 steps
    So 500 epochs = ~65M steps
    """

    # Enable second stack rewards at 65M steps (~500 epochs)
    enable_stack_1_on_2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "stack_cube_1_on_2", "weight": 1.0, "num_steps": 65_000_000}
    )

    enable_nearly_stacked_1_on_2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "nearly_stacked_1_on_2", "weight": 5.0, "num_steps": 65_000_000}
    )

    enable_knob_alignment_1_on_2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "knob_alignment_1_on_2", "weight": 3.0, "num_steps": 65_000_000}
    )

    enable_stability_1_on_2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "stability_1_on_2", "weight": -0.1, "num_steps": 65_000_000}
    )


@configclass
class Franka3CubeStackLegoVisuomotorIKEnvCfg(stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg):
    """LEGO 3-Cube Stack environment with VISION + IK control + LEGO rewards.

    This configuration inherits:
    - IK control (7D Cartesian pose actions)
    - Vision (wrist + table cameras at 84x84)
    - Domain randomization (lighting, textures)

    And adds:
    - LEGO bricks (c_lego_duplo.usd at 1.04x scale = 5cm)
    - LEGO-specific precision rewards
    - Progressive curriculum learning
    """

    def __post_init__(self):
        # Post init of parent (gets IK, cameras, domain randomization)
        super().__post_init__()

        # Replace rewards with LEGO-specific ones
        self.rewards = RewardsCfg()

        # Add curriculum for progressive learning
        self.curriculum = CurriculumCfg()

        # Improved physics for LEGO precision stacking
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,  # Increased from 16 for better contact stability
            solver_velocity_iteration_count=2,   # Increased from 1 for smoother dynamics
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=1.0,      # Decreased from 5.0 to reduce jitter/instability
            disable_gravity=False,
        )

        # Replace cubes with LEGO bricks (scaled to 5cm to match smooth blocks)
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/c_lego_duplo_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.40, 0.00, 0.025], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="props/c_lego_duplo.usd",
                scale=(1.04, 1.04, 1.04),  # Scale to 5.0cm to exactly match smooth blocks
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )

        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/c_lego_duplo_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.025], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="props/c_lego_duplo.usd",
                scale=(1.04, 1.04, 1.04),  # Scale to 5.0cm to exactly match smooth blocks
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )

        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/c_lego_duplo_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.025], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="props/c_lego_duplo.usd",
                scale=(1.04, 1.04, 1.04),  # Scale to 5.0cm to exactly match smooth blocks
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Update cube position randomization to use correct Z height
        self.events.randomize_cube_positions.params["pose_range"]["z"] = (0.025, 0.025)

        # Keep cameras at standard 84x84 from parent config for now
        # (512x512 causes initialization issues with raw image observations)

        # Note: IK control and domain randomization are inherited from parent
        # Action space: 7D (x, y, z, qx, qy, qz, qw) relative end-effector pose
