# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    bin_stack_ik_rel_env_cfg,
    stack_ik_abs_env_cfg,
    stack_ik_rel_blueprint_env_cfg,
    stack_ik_rel_env_cfg,
    stack_ik_rel_env_cfg_skillgen,
    stack_ik_rel_instance_randomize_env_cfg,
    stack_ik_rel_visuomotor_cosmos_env_cfg,
    stack_ik_rel_visuomotor_env_cfg,
    stack_joint_pos_env_cfg,
    stack_joint_pos_instance_randomize_env_cfg,
    stack_2cube_env_cfg,
    stack_2cube_ik_rel_env_cfg,
    stack_2cube_lego_joint_pos_rl_env_cfg,
    stack_2cube_lego_fixed_joint_env_cfg,
    stack_2cube_smooth_fixed_joint_env_cfg,
    stack_2cube_smooth_fixed_joint_vel002_env_cfg,
    stack_2cube_smooth_fixed_joint_vel010_env_cfg,
    stack_2cube_smooth_fixed_joint_vel020_env_cfg,
    stack_2cube_rl_lego_joint_pos_rl_env_cfg,
    stack_2cube_smooth_ik_rel_env_cfg,
    stack_2cube_smooth_ik_rel_env_cfg_v2,
    stack_3cube_lego_joint_pos_rl_env_cfg,
    stack_3cube_lego_visuomotor_joint_pos_rl_env_cfg,
    stack_3cube_lego_visuomotor_ik_rel_env_cfg,
    stack_2cube_ablation_no_relative_env_cfg,
    stack_2cube_ablation_no_gripper_env_cfg,
    stack_2cube_ablation_no_eef_env_cfg,
    stack_2cube_ablation_minimal_env_cfg,
    stack_2cube_ablation_no_proprio_env_cfg,
    stack_2cube_ablation_no_absolute_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_joint_pos_instance_randomize_env_cfg.FrankaCubeStackInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_env_cfg.FrankaCubeStackEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_visuomotor_cosmos_env_cfg.FrankaCubeStackVisuomotorCosmosEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_cosmos.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_abs_env_cfg.FrankaCubeStackEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_instance_randomize_env_cfg.FrankaCubeStackInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_blueprint_env_cfg.FrankaCubeStackBlueprintEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_ik_rel_env_cfg_skillgen.FrankaCubeStackSkillgenEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": bin_stack_ik_rel_env_cfg.FrankaBinStackEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

##
# 2-Cube Stack (Matches IsaacGym Exactly)
##

gym.register(
    id="Isaac-Stack-2Cube-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_env_cfg.Franka2CubeStackEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
"""2-cube stacking with joint position control (clean LEGO brick testing)."""

gym.register(
    id="Isaac-Stack-2Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_ik_rel_env_cfg.Franka2CubeStackIKEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
    disable_env_checker=True,
)
"""2-cube stacking task matching IsaacGym's FrankaCubeStack exactly (PPO or SAC)."""

gym.register(
    id="Isaac-Stack-2Cube-Smooth-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_smooth_ik_rel_env_cfg.Franka2CubeStackSmoothIKEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
"""2-cube stacking with SMOOTH CUBES (diagnostic/testing - should learn faster than LEGO)."""

gym.register(
    id="Isaac-Stack-2Cube-Smooth-Blocks-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_smooth_ik_rel_env_cfg_v2.Franka2CubeStackSmoothIKEnvCfgV2,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
"""2-cube stacking with SMOOTH BLOCKS from original Isaac assets (for baseline comparison)."""

gym.register(
    id="Isaac-Stack-2Cube-LEGO-Franka-JointPos-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_lego_joint_pos_rl_env_cfg.Franka2CubeStackLegoJointPosRLEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""LEGO 2-cube stacking with joint position control and precision rewards (knob-cavity alignment)."""

gym.register(
    id="Isaac-Stack-2Cube-RLLego-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_rl_lego_joint_pos_rl_env_cfg.Franka2CubeStackRLLegoJointPosRLEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""RL-OPTIMIZED 2-cube stacking: Simple box geometry + high friction. No USD interlocking. Optimized for learnability!"""

gym.register(
    id="Isaac-Stack-2Cube-LEGO-FixedJoint-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_lego_fixed_joint_env_cfg.Franka2CubeStackLegoFixedJointEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""LEGO 2-cube stacking with FIXED JOINT SNAP: Deterministic constraint-based snap (zero instability). Visual: brick_rl_optimized.usd. Collision: Simple box + 4 alignment studs."""

gym.register(
    id="Isaac-Stack-2Cube-Smooth-FixedJoint-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_smooth_fixed_joint_env_cfg.Franka2CubeStackSmoothFixedJointEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""SMOOTH 2-cube stacking with FIXED JOINT SNAP: Uses proven Omniverse smooth cubes with LEGO-style fixed joint snapping. Avoids custom USD physics issues."""

gym.register(
    id="Isaac-Stack-2Cube-Smooth-FixedJoint-Vel002-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_smooth_fixed_joint_vel002_env_cfg.Franka2CubeStackSmoothFixedJointVel002EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""Hyperparameter variant: velocity_threshold=0.02 m/s (very strict)"""

gym.register(
    id="Isaac-Stack-2Cube-Smooth-FixedJoint-Vel010-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_smooth_fixed_joint_vel010_env_cfg.Franka2CubeStackSmoothFixedJointVel010EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""Hyperparameter variant: velocity_threshold=0.10 m/s (relaxed)"""

gym.register(
    id="Isaac-Stack-2Cube-Smooth-FixedJoint-Vel020-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_smooth_fixed_joint_vel020_env_cfg.Franka2CubeStackSmoothFixedJointVel020EnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""Hyperparameter variant: velocity_threshold=0.20 m/s (very relaxed)"""

gym.register(
    id="Isaac-Stack-3Cube-LEGO-Franka-JointPos-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_3cube_lego_joint_pos_rl_env_cfg.Franka3CubeStackLegoJointPosRLEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""LEGO 3-cube stacking with joint position control and precision rewards. 95-dim observation space matches smooth blocks checkpoint for transfer learning."""

gym.register(
    id="Isaac-Stack-3Cube-LEGO-Visuomotor-Franka-JointPos-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_3cube_lego_visuomotor_joint_pos_rl_env_cfg.Franka3CubeStackLegoVisuomotorJointPosRLEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
"""LEGO 3-cube stacking with VISION (wrist + table cameras), joint position control, and LEGO precision rewards. Combines ResNet18 visual features with proprioceptive state for robust manipulation."""

gym.register(
    id="Isaac-Stack-3Cube-LEGO-Visuomotor-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_3cube_lego_visuomotor_ik_rel_env_cfg.Franka3CubeStackLegoVisuomotorIKEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image_84.json"),
    },
    disable_env_checker=True,
)
"""LEGO 3-cube stacking with VISION + IK control. Uses 7D Cartesian actions, raw RGB images (84x84), LEGO precision rewards, and domain randomization. Standard approach for vision-based manipulation."""

##
# Observation Space Ablation Studies
##

gym.register(
    id="Isaac-Stack-2Cube-Ablation-NoRelative-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_ablation_no_relative_env_cfg.Franka2CubeStackAblationNoRelativeEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""ABLATION 1: Remove relative position observations (58D → 49D). Tests if explicit gripper-to-cube and cube-to-cube relative positions are necessary."""

gym.register(
    id="Isaac-Stack-2Cube-Ablation-NoGripper-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_ablation_no_gripper_env_cfg.Franka2CubeStackAblationNoGripperEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""ABLATION 2: Remove gripper state observations (58D → 56D). Tests if explicit gripper finger positions are necessary."""

gym.register(
    id="Isaac-Stack-2Cube-Ablation-NoEEF-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_ablation_no_eef_env_cfg.Franka2CubeStackAblationNoEEFEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""ABLATION 3: Remove end-effector pose observations (58D → 51D). Tests if explicit EE tracking is necessary or can be inferred from joint angles."""

gym.register(
    id="Isaac-Stack-2Cube-Ablation-Minimal-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_ablation_minimal_env_cfg.Franka2CubeStackAblationMinimalEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""ABLATION 4: Minimal observations (58D → 40D). Only proprioception + cube poses, no relative positions, EE tracking, or gripper state."""

gym.register(
    id="Isaac-Stack-2Cube-Ablation-NoProprio-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_ablation_no_proprio_env_cfg.Franka2CubeStackAblationNoProprioEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""ABLATION 5: Remove proprioception (58D → 32D). Tests if policy can learn without actions/joint positions/velocities."""

gym.register(
    id="Isaac-Stack-2Cube-Ablation-NoAbsolute-Franka-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": stack_2cube_ablation_no_absolute_env_cfg.Franka2CubeStackAblationNoAbsoluteEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_state_cfg.yaml",
    },
    disable_env_checker=True,
)
"""ABLATION 6: Remove absolute object poses (58D → 44D). Tests if policy can learn with only relative spatial relationships."""
