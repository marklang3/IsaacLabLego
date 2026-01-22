# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task 1: Reach and grasp cube_2
    reaching_cube_2 = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube_2")},
        weight=1.0,
    )

    grasping_cube_2 = RewTerm(
        func=mdp.object_is_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube_2"),
        },
        weight=5.0,
    )

    lifting_cube_2 = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube_2")},
        weight=10.0,
    )

    # Task 2: Stack cube_2 on cube_3
    aligning_cube_2_over_cube_3_xy = RewTerm(
        func=mdp.object_goal_distance_xy,
        params={"std": 0.1, "upper_object_cfg": SceneEntityCfg("cube_2"), "lower_object_cfg": SceneEntityCfg("cube_3")},
        weight=3.0,
    )

    cube_2_above_cube_3 = RewTerm(
        func=mdp.object_above_object,
        params={
            "minimal_height": 0.04,
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
        },
        weight=5.0,
    )

    stacking_cube_2_on_cube_3 = RewTerm(
        func=mdp.object_stacked,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "xy_threshold": 0.05,
            "height_threshold": 0.01,
            "height_diff": 0.0468,
        },
        weight=20.0,
    )

    # Task 3: Reach and grasp cube_1
    reaching_cube_1 = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube_1")},
        weight=1.0,
    )

    grasping_cube_1 = RewTerm(
        func=mdp.object_is_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube_1"),
        },
        weight=5.0,
    )

    lifting_cube_1 = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube_1")},
        weight=10.0,
    )

    # Task 4: Stack cube_1 on cube_2
    aligning_cube_1_over_cube_2_xy = RewTerm(
        func=mdp.object_goal_distance_xy,
        params={"std": 0.1, "upper_object_cfg": SceneEntityCfg("cube_1"), "lower_object_cfg": SceneEntityCfg("cube_2")},
        weight=3.0,
    )

    cube_1_above_cube_2 = RewTerm(
        func=mdp.object_above_object,
        params={
            "minimal_height": 0.04,
            "upper_object_cfg": SceneEntityCfg("cube_1"),
            "lower_object_cfg": SceneEntityCfg("cube_2"),
        },
        weight=5.0,
    )

    stacking_cube_1_on_cube_2 = RewTerm(
        func=mdp.object_stacked,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "upper_object_cfg": SceneEntityCfg("cube_1"),
            "lower_object_cfg": SceneEntityCfg("cube_2"),
            "xy_threshold": 0.05,
            "height_threshold": 0.01,
            "height_diff": 0.0468,
        },
        weight=20.0,
    )

    # Orientation alignment rewards (important for lego brick studs)
    orientation_alignment_2_3 = RewTerm(
        func=mdp.object_orientation_alignment,
        params={"std": 0.5, "upper_object_cfg": SceneEntityCfg("cube_2"), "lower_object_cfg": SceneEntityCfg("cube_3")},
        weight=2.0,
    )

    orientation_alignment_1_2 = RewTerm(
        func=mdp.object_orientation_alignment,
        params={"std": 0.5, "upper_object_cfg": SceneEntityCfg("cube_1"), "lower_object_cfg": SceneEntityCfg("cube_2")},
        weight=2.0,
    )

    # Overall progress reward
    stacking_progress = RewTerm(func=mdp.stacking_progress, weight=15.0)

    # Action penalties for smooth motion
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")}
    )

    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")}
    )

    cube_3_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_3")}
    )

    success = DoneTerm(func=mdp.cubes_stacked)


@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024  # Increased for 3 cubes per env
        self.sim.physx.friction_correlation_distance = 0.00625