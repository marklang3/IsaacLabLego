# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


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
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],  # Only 2 cubes to match IsaacGym
        },
    )


@configclass
class Franka2CubeStackEnvCfg(StackEnvCfg):
    """Configuration for 2-cube stacking task - matches IsaacGym exactly."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,  # Higher than IsaacGym (8) due to LEGO brick stud geometry - more contact points require better resolution
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Set each stacking cube deterministically (2 cubes only - matching IsaacGym)
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/c_lego_duplo_2",   
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55,  0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="props/c_lego_duplo.usd",
                scale=(1.5, 1.5, 1.5),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )

        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/c_lego_duplo_3",  
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.10, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path="props/c_lego_duplo.usd",
                scale=(1.5, 1.5, 1.5),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        # 2-Cube Configuration: Add cube_2 and cube_3 observations

        # Remove 3-cube observations that reference cube_1
        if hasattr(self.observations.policy, 'object'):
            delattr(self.observations.policy, 'object')
        if hasattr(self.observations.policy, 'cube_positions'):
            delattr(self.observations.policy, 'cube_positions')
        if hasattr(self.observations.policy, 'cube_orientations'):
            delattr(self.observations.policy, 'cube_orientations')

        # Add 2-cube specific observations (cube_2 and cube_3 only)
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        # Cube 2 position and orientation (the cube we need to pick up)
        self.observations.policy.cube_2_pos = ObsTerm(func=mdp.cube_2_position)
        self.observations.policy.cube_2_quat = ObsTerm(func=mdp.cube_2_orientation)

        # Cube 3 position and orientation (the base cube we stack on)
        self.observations.policy.cube_3_pos = ObsTerm(func=mdp.cube_3_position)
        self.observations.policy.cube_3_quat = ObsTerm(func=mdp.cube_3_orientation)

        # Disable cube_1 reward (only stack cube_2 on cube_3)
        if hasattr(self.rewards, 'stack_cube_1_on_2'):
            delattr(self.rewards, 'stack_cube_1_on_2')

        # Disable cube_1 termination checks
        if hasattr(self.terminations, 'cube_1_dropping'):
            delattr(self.terminations, 'cube_1_dropping')
        if hasattr(self.terminations, 'success'):
            delattr(self.terminations, 'success')  # cubes_stacked function expects 3 cubes

        # No curriculum needed for 2-cube task (single objective)
        if hasattr(self, 'curriculum'):
            self.curriculum = None
