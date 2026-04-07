# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
2-Cube Stacking with SMOOTH CUBES (not LEGO) for testing/diagnostics.

This config is identical to stack_2cube_ik_rel_env_cfg but uses simple smooth cubes
to verify that the task/rewards/training work correctly without LEGO stud complexity.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

# Import base IK config (which has IK actions)
from . import stack_2cube_ik_rel_env_cfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.025, 0.025), "yaw": (-1.0, 1.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class Franka2CubeStackSmoothIKEnvCfg(stack_2cube_ik_rel_env_cfg.Franka2CubeStackIKEnvCfg):
    """2-Cube stacking with SMOOTH CUBES for diagnostic testing."""

    def __post_init__(self):
        # post init of parent (sets up IK actions, robot, etc.)
        super().__post_init__()

        # Override events with our version
        self.events = EventCfg()

        # Rigid body properties for smooth cubes
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,  # Match IsaacGym (less than LEGO's 16)
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Replace LEGO bricks with smooth cubes (0.05m = 5cm cubes, matching IsaacGym)
        # Note: IsaacGym uses 5cm cubes, our LEGO scaled 1.5x is ~7cm
        cube_size = 0.05  # 5cm cubes (same as IsaacGym)

        from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg

        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.025], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(cube_size, cube_size, cube_size),
                rigid_props=cube_properties,
                mass_props=MassPropertiesCfg(mass=0.057),  # ~57g mass like IsaacGym 5cm cube
                visual_material=None,  # Use default material
                physics_material=None,  # Use default physics
                semantic_tags=[("class", "cube_2")],
            ),
        )

        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.025], rot=[1, 0, 0, 0]),
            spawn=CuboidCfg(
                size=(cube_size, cube_size, cube_size),
                rigid_props=cube_properties,
                mass_props=MassPropertiesCfg(mass=0.057),  # ~57g mass like IsaacGym 5cm cube
                visual_material=None,
                physics_material=None,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Update reward parameters to match smooth cube size
        # Cube size is 0.05m (5cm), matching IsaacGym
        self.rewards.stack_cube_2_on_3.params["upper_size"] = cube_size
        self.rewards.stack_cube_2_on_3.params["lower_size"] = cube_size
        self.rewards.stack_cube_2_on_3.params["table_height"] = 1.025

        # Note: The end-effector frame is already set up by parent class
        # No need to redefine ee_frame
