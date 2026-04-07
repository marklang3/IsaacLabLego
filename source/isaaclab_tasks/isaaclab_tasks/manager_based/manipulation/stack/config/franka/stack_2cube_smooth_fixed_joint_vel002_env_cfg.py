"""
2-Cube Stacking with SMOOTH CUBES + Fixed Joint Snap.
HYPERPARAMETER VARIANT: velocity_threshold = 0.02 m/s (VERY STRICT)
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_2cube_lego_joint_pos_rl_env_cfg import (
    Franka2CubeStackLegoJointPosRLEnvCfg,
)


@configclass
class EventCfg:
    """Events for smooth cube stacking with fixed joint snap."""

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
            "min_separation": 0.15,
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

    update_curriculum = EventTerm(
        func=mdp.update_snap_curriculum,
        mode="interval",
        interval_range_s=(30.0, 30.0),
        params={
            "initial_xy_threshold": 0.030,
            "final_xy_threshold": 0.020,
            "initial_z_threshold": 0.020,
            "final_z_threshold": 0.015,
            "curriculum_length": 5000,
        },
    )

    # HYPERPARAMETER VARIANT: Very strict velocity requirement
    apply_snap = EventTerm(
        func=mdp.fixed_joint_snap,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "upper_object_name": "cube_2",
            "lower_object_name": "cube_3",
            "snap_xy_threshold": 0.015,
            "snap_z_threshold": 0.010,
            "expected_stack_height": 0.05,
            "require_low_velocity": True,
            "velocity_threshold": 0.02,  # VERY STRICT: 2cm/s (vs baseline 5cm/s)
            "use_curriculum": True,
        },
    )


@configclass
class RewardsCfg:
    """Rewards for smooth cube stacking with fixed joint snap."""

    stack_cube_2_on_3 = RewTerm(
        func=mdp.isaacgym_combined_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.05,
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,
            "r_dist_scale": 0.1,
            "r_lift_scale": 1.5,
            "r_align_scale": 2.0,
            "r_stack_scale": 16.0,
            "xy_threshold": 0.015,
            "z_threshold": 0.005,
        },
        weight=1.0,
    )

    rotation_alignment_2_3 = RewTerm(
        func=mdp.object_orientation_alignment,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "std": 0.5,
        },
        weight=0.1,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Franka2CubeStackSmoothFixedJointVel002EnvCfg(Franka2CubeStackLegoJointPosRLEnvCfg):
    """2-Cube stacking with velocity_threshold=0.02 (very strict)."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.events = EventCfg()
        self.rewards = RewardsCfg()
        self.sim.render_interval = self.decimation

        cube_size = 0.05
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=1.0,
            max_contact_impulse=1e32,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        )

        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.05, 0.025],
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_2")],
            ),
        )

        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.60, -0.10, 0.025],
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        self.actions.arm_action.scale = 0.5


@configclass
class Franka2CubeStackSmoothFixedJointVel002EnvCfg_PLAY(Franka2CubeStackSmoothFixedJointVel002EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
