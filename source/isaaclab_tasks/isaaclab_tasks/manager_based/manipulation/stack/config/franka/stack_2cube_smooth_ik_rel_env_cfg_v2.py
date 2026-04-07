# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
2-Cube Stacking with SMOOTH BLOCKS (using original Isaac USD files).

Reverts to the original smooth block assets that were in upstream IsaacLab.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import base IK config
from . import stack_2cube_ik_rel_env_cfg


@configclass
class Franka2CubeStackSmoothIKEnvCfgV2(stack_2cube_ik_rel_env_cfg.Franka2CubeStackIKEnvCfg):
    """2-Cube stacking with SMOOTH BLOCKS from original Isaac assets."""

    def __post_init__(self):
        # Call parent to set up IK actions, robot, etc.
        super().__post_init__()

        # Rigid body properties for smooth blocks (match IsaacGym)
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,  # IsaacGym uses 8, not 16 like LEGO
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Replace LEGO bricks with original smooth blocks from Isaac Nucleus
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )

        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # CRITICAL: Update reward parameters to match Isaac block size (0.05m not 0.0702m LEGO)
        # Isaac blocks are 5cm cubes, LEGO is 7cm (scaled 1.5x from 4.68cm)
        block_size = 0.05  # 5cm Isaac blocks (not 7cm LEGO)
        self.rewards.stack_cube_2_on_3.params["upper_size"] = block_size
        self.rewards.stack_cube_2_on_3.params["lower_size"] = block_size
        self.rewards.stack_cube_2_on_3.params["table_height"] = 1.025
