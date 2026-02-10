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
    stack_2cube_smooth_ik_rel_env_cfg,
    stack_2cube_smooth_ik_rel_env_cfg_v2,
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
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
"""LEGO 2-cube stacking with joint position control and precision rewards (knob-cavity alignment)."""
