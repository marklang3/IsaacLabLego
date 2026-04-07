"""
LEGO 2-Cube Stack with Fixed Joint Constraint Snap.

Design Philosophy (RL-Optimized):
- Visual: brick_rl_optimized.usd (realistic appearance)
- Collision: Simple box + 4 small alignment studs
- Snap Mechanism: Fixed joint constraint (deterministic, zero instability)
- Physics: Stable defaults (CCD, high solver iterations)

When bricks align (XY < 1.5cm, Z correct, low velocity):
→ Create fixed joint constraint
→ Deterministic "LEGO click"
→ Zero sliding after snap
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_2cube_lego_joint_pos_rl_env_cfg import (
    Franka2CubeStackLegoJointPosRLEnvCfg,
)


@configclass
class EventCfg:
    """Events for LEGO stacking with fixed joint snap."""

    # Set arm to ready pose (must come before joint randomization)
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    # Apply small random joint offset for diversity
    reset_robot_joints = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Randomize both brick positions with min separation (z = half brick height = 14.4mm)
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0144, 0.0144), "yaw": (-1.0, 1.0)},
            "min_separation": 0.2,  # INCREASED from 0.1 to 0.2 - force robot to reach, can't just nudge
            "asset_cfgs": [SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

    # Reset snap tracking on episode reset
    reset_snap_tracking = EventTerm(
        func=mdp.reset_fixed_joint_snap,
        mode="reset",
    )

    # Update curriculum thresholds (tighten over time)
    update_curriculum = EventTerm(
        func=mdp.update_snap_curriculum,
        mode="reset",
        params={
            "initial_xy_threshold": 0.025,  # Start at 2.5cm (easier early)
            "final_xy_threshold": 0.015,    # End at 1.5cm (precise)
            "initial_z_threshold": 0.015,   # Start at 1.5cm
            "final_z_threshold": 0.010,     # End at 1.0cm
            "curriculum_length": 5000,      # Transition over 5000 epochs
        },
    )

    # Fixed joint snap - runs every step
    check_snap_alignment = EventTerm(
        func=mdp.fixed_joint_snap,
        mode="interval",
        interval_range_s=(0.05, 0.05),  # Check every 50ms
        params={
            "upper_object_name": "cube_2",
            "lower_object_name": "cube_3",
            "snap_xy_threshold": 0.015,   # Final target: 1.5cm (curriculum starts at 2.5cm)
            "snap_z_threshold": 0.010,    # Final target: 1.0cm (curriculum starts at 1.5cm)
            "expected_stack_height": 0.0288,  # One brick height (28.8mm - scaled up 1.5x)
            "require_low_velocity": True,  # ENABLE velocity check for gentle placement
            "velocity_threshold": 0.05,   # 5cm/s (gentle placement required)
            "use_curriculum": True,       # ENABLE curriculum (thresholds will tighten over time)
        },
    )


@configclass
class RewardsCfg:
    """Reward terms emphasizing lifting BEFORE alignment."""

    # IsaacGym hierarchical reward (reach → lift → align → stack)
    stack_cube_2_on_3 = RewTerm(
        func=mdp.isaacgym_combined_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.0288,  # 28.8mm HEIGHT
            "lower_size": 0.0288,
            "table_height": 0.0,
            "lift_threshold": 0.004,  # REDUCED from 0.006 to 0.004 (4mm) - easier to trigger
            "r_dist_scale": 0.1,
            "r_lift_scale": 5.0,  # INCREASED from 1.5 to 5.0 - make lifting very valuable
            "r_align_scale": 2.0,
            "r_stack_scale": 16.0,
        },
        weight=1.0,
    )

    # Strong alignment bonus BUT ONLY IF LIFTED (prevents nudging exploit)
    nearly_stacked = RewTerm(
        func=mdp.nearly_stacked_reward,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "upper_size": 0.0288,
            "lower_size": 0.0288,
            "table_height": 0.0,
            "lift_threshold": 0.004,  # REDUCED - must be lifted to get this bonus
            "xy_threshold": 0.03,
            "z_threshold": 0.015,
        },
        weight=2.0,  # REDUCED from 5.0 to 2.0 - less exploitable
    )

    # Stability bonus (low velocity when close)
    stability = RewTerm(
        func=mdp.stability_penalty,
        params={
            "upper_object_cfg": SceneEntityCfg("cube_2"),
            "lower_object_cfg": SceneEntityCfg("cube_3"),
            "xy_threshold": 0.03,
        },
        weight=-0.1,
    )

    # Held stable bonus (maintain snap with low velocity)
    held_stable = RewTerm(
        func=mdp.held_stable_bonus,
        params={
            "upper_object_name": "cube_2",
            "velocity_threshold": 0.01,     # 1cm/s (very stable)
            "stable_steps_required": 10,    # 10 consecutive steps
        },
        weight=2.0,  # Significant bonus for robustness
    )

    # Action penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-1e-4,
    )


@configclass
class Franka2CubeStackLegoFixedJointEnvCfg(Franka2CubeStackLegoJointPosRLEnvCfg):
    """LEGO stacking with fixed joint constraint snap."""

    def __post_init__(self):
        super().__post_init__()

        # Override events
        self.events = EventCfg()

        # Override rewards
        self.rewards = RewardsCfg()
        
        # CRITICAL FIX: Increase action scale for better responsiveness
        from isaaclab_tasks.manager_based.manipulation.stack import mdp
        self.actions.arm_action.scale = 1.0  # Increased from 0.5

        # ========================================
        # PHYSICS PARAMETERS (User Specified)
        # ========================================

        # Rigid body properties - VERY AGGRESSIVE for stability (stop jitter)
        brick_rigid_props = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,  # DOUBLED - much more stable
            solver_velocity_iteration_count=16,  # DOUBLED - stops micro-vibrations
            max_depenetration_velocity=0.5,     # REDUCED - gentler contact resolution
            max_contact_impulse=1e32,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        )

        # Collision properties - optimized to prevent jitter
        brick_collision = CollisionPropertiesCfg(
            contact_offset=0.01,   # 10mm - larger for earlier contact detection
            rest_offset=-0.001,    # -1mm - slight penetration reduces jitter
        )

        # ========================================
        # RL-OPTIMIZED BRICK USD
        # ========================================

        # Replace cubes with RL-optimized brick
        # Visual: brick.usdc (realistic)
        # Collision: Simple box only (no stud collisions)
        # Friction: 1.2/1.0 (built into USD)
        # Mass: 0.111kg (111g - built into USD, 1.5x scale)

        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.05, 0.0144],  # Z = half height (28.8mm / 2 = 14.4mm)
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path="/home/mlangtry/IsaacLabLego/props/brick_rl_optimized.usd",
                rigid_props=brick_rigid_props,
                collision_props=brick_collision,
                semantic_tags=[("class", "cube_2")],
            ),
        )

        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.60, -0.10, 0.0144],  # Z = half height
                rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path="/home/mlangtry/IsaacLabLego/props/brick_rl_optimized.usd",
                rigid_props=brick_rigid_props,
                collision_props=brick_collision,
                semantic_tags=[("class", "cube_3")],
            ),
        )
