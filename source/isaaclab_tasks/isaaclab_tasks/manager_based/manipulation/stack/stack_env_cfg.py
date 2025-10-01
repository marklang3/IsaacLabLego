# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
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
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


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


from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from . import mdp  # already present in your file


@configclass
class RewardsCfg:
    """Reward terms for the stacking MDP (mdp-only, no command manager)."""

    # ========= Stage 1: stack cube_2 onto cube_1 =========

    # End-effector reaches cube_2
    reach_cube_2 = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.10,
            "object_cfg": SceneEntityCfg("cube_2"),
            # ee_frame_cfg is optional; mdp.object_ee_distance has a default SceneEntityCfg("ee_frame")
        },
        weight=1.0,
    )

    # Grasp cube_2 (uses your mdp.object_grasped signature with robot & ee_frame)
    grasp_cube_2 = RewTerm(
        func=mdp.object_grasped,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cube_2"),
        },
        weight=3.0,
    )

    # Lift cube_2 off the table
    lift_cube_2 = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "minimal_height": 0.03,  # adjust if your table/block geometry needs more clearance
            "object_cfg": SceneEntityCfg("cube_2"),
        },
        weight=5.0,
    )

    # Binary reward when cube_2 is stacked on cube_1 (you already use this in observations)
    stack_2_on_1 = RewTerm(
        func=mdp.object_stacked,
        params={
            "robot_cfg": SceneEntityCfg("robot"),  # if unused inside, it’s harmless
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_1"),
        },
        weight=15.0,
    )

    # # ========= Stage 2: stack cube_3 onto cube_2 (optional) =========
    # # If you start with only 2 cubes, delete or comment this block.

    # reach_cube_3 = RewTerm(
    #     func=mdp.object_ee_distance,
    #     params={"std": 0.10, "object_cfg": SceneEntityCfg("cube_3")},
    #     weight=0.75,
    # )

    # grasp_cube_3 = RewTerm(
    #     func=mdp.object_grasped,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    #         "object_cfg": SceneEntityCfg("cube_3"),
    #     },
    #     weight=2.0,
    # )

    # lift_cube_3 = RewTerm(
    #     func=mdp.object_is_lifted,
    #     params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube_3")},
    #     weight=3.0,
    # )

    # stack_3_on_2 = RewTerm(
    #     func=mdp.object_stacked,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "upper_object_cfg": SceneEntityCfg("cube_3"),
    #         "lower_object_cfg": SceneEntityCfg("cube_2"),
    #     },
    #     weight=10.0,
    # )

    # # ========= Regularization =========

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    #     weight=-1e-4,
    # )



@configclass
class StackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    rewards: RewardsCfg = RewardsCfg()

    # Unused managers
    commands = None
    # rewards = None
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
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 4 * 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
