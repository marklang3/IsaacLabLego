"""
2-Cube Stacking with SMOOTH CUBES + Fixed Joint Snap Mechanism.

Combines proven physics of smooth Omniverse cubes with deterministic LEGO-style snapping.
Avoids custom USD brick physics issues (flipping, table penetration).

Design:
- Cubes: Standard 5cm Omniverse smooth blocks (proven stable)
- Snap: Fixed joint constraint when aligned (from LEGO config)
- Curriculum: Progressive threshold tightening over 5000 epochs
- Control: Joint position control
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

    # Set arm to ready pose (must come before joint randomization)
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    # Randomize joint positions slightly (for robustness)
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Randomize cube positions with minimum separation
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.025, 0.025), "yaw": (-1.0, 1.0)},
            "min_separation": 0.15,  # Moderate separation for learning
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

    # Update curriculum thresholds (tighten over time)
    update_curriculum = EventTerm(
        func=mdp.update_snap_curriculum,
        mode="interval",
        interval_range_s=(30.0, 30.0),  # Run once per episode (thresholds change slowly)
        params={
            "initial_xy_threshold": 0.030,  # Start at 3.0cm (more forgiving for learning)
            "final_xy_threshold": 0.020,    # End at 2.0cm
            "initial_z_threshold": 0.020,   # Start at 2.0cm
            "final_z_threshold": 0.015,     # End at 1.5cm
            "curriculum_length": 5000,      # Transition over 5000 epochs
        },
    )

    # Apply fixed joint snap when aligned (FAST: velocity zeroing only, no USD joints)
    apply_snap = EventTerm(
        func=mdp.fixed_joint_snap,
        mode="interval",
        interval_range_s=(0.1, 0.1),  # Check every 0.1s (2 env steps) - still responsive but 2× less overhead
        params={
            "upper_object_name": "cube_2",
            "lower_object_name": "cube_3",
            "snap_xy_threshold": 0.015,  # Will be overridden by curriculum
            "snap_z_threshold": 0.010,   # Will be overridden by curriculum
            "expected_stack_height": 0.05,  # 5cm cube height
            "require_low_velocity": True,
            "velocity_threshold": 0.05,
            "use_curriculum": True,
        },
    )


@configclass
class RewardsCfg:
    # IsaacGym hierarchical reward (reach → lift → align → stack)
    stack_cube_2_on_3 = RewTerm(
        func=mdp.isaacgym_combined_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.05,  # 5cm cubes
            "lower_size": 0.05,
            "table_height": 0.0,
            "lift_threshold": 0.01,  # 10mm lift (matches baseline)
            "r_dist_scale": 0.1,     # Match baseline
            "r_lift_scale": 1.5,     # Match baseline
            "r_align_scale": 2.0,    # Match baseline
            "r_stack_scale": 16.0,   # Match baseline
            "xy_threshold": 0.015,   # 1.5cm XY alignment (tighter to prevent hovering exploit)
            "z_threshold": 0.005,    # 5mm vertical tolerance (force actual contact, not hovering)
        },
        weight=1.0,
    )

    # Rotation alignment reward for LEGO transfer preparation
    # Small weight for fine-tuning on top of pre-trained baseline (was 1.5, reduced to avoid destabilisation)
    rotation_alignment_2_3 = RewTerm(
        func=mdp.object_orientation_alignment,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "std": 0.5,  # Tanh smoothing parameter
        },
        weight=0.1,
    )

    # Action smoothness penalties (match baseline)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Franka2CubeStackSmoothFixedJointEnvCfg(Franka2CubeStackLegoJointPosRLEnvCfg):
    """2-Cube stacking with smooth cubes and fixed joint snap mechanism."""

    def __post_init__(self):
        # Post init of parent (sets up robot, actions, etc.)
        super().__post_init__()

        # Set episode length to 30s (match successful baseline run)
        self.episode_length_s = 30.0

        # Override events with our version
        self.events = EventCfg()

        # Override rewards with our version
        self.rewards = RewardsCfg()

        # Fix render interval for performance (match decimation to avoid wasteful rendering)
        self.sim.render_interval = self.decimation  # 5 (was 2, causing 2.5 renders per env step!)

        # Cube size
        cube_size = 0.05  # 5cm cubes (same as IsaacGym)

        # Rigid body properties - 32/2 for precision stacking (now with proper speed!)
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,  # Precision stacking (replicate_physics=False fixed the speed!)
            solver_velocity_iteration_count=2,   # Smooth dynamics
            max_depenetration_velocity=1.0,
            max_contact_impulse=1e32,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        )

        # Replace LEGO bricks with smooth cubes (using USD files for 16× speedup!)
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/cube_2",  # lowercase to match rewards
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.05, 0.025],  # Z = half height (5cm / 2 = 2.5cm)
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
            prim_path="{ENV_REGEX_NS}/cube_3",  # lowercase to match rewards
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.60, -0.10, 0.025],  # Z = half height
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Match baseline action scale for smooth control
        self.actions.arm_action.scale = 0.5  # Baseline value - enables precise manipulation


@configclass
class Franka2CubeStackSmoothFixedJointEnvCfg_PLAY(Franka2CubeStackSmoothFixedJointEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
