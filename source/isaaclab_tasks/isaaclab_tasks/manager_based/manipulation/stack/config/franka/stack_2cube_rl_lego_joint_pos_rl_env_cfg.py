"""RL-Optimized LEGO brick stacking with procedural geometry.

Design Philosophy:
- Simple box collider (no underside cavities)
- High friction for stability
- Let rewards simulate "snap", not geometry
- Optimized for learnability, not physical realism
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_2cube_lego_joint_pos_rl_env_cfg import (
    Franka2CubeStackLegoJointPosRLEnvCfg,
    RewardsCfg,
)


@configclass
class Franka2CubeStackRLLegoJointPosRLEnvCfg(Franka2CubeStackLegoJointPosRLEnvCfg):
    """RL-Optimized LEGO stacking environment.

    Uses simple box geometry with high friction instead of complex USD with interlocking.
    Optimized for RL training stability.
    """

    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()

        # ========================================
        # RL-OPTIMIZED PHYSICS (Critical!)
        # ========================================

        # Rigid body properties - stable solver settings
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,  # Standard (not over-tuned)
            solver_velocity_iteration_count=8,   # Increased for stability
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        )

        # Collision properties - clean contact generation
        cube_collision = CollisionPropertiesCfg(
            contact_offset=0.002,  # Small contact offset (2mm)
            rest_offset=0.0,       # Objects can touch
        )

        # HIGH FRICTION - this is what makes stacking work!
        cube_friction = RigidBodyMaterialCfg(
            static_friction=1.2,   # Very high - prevents sliding
            dynamic_friction=1.0,  # High - stable contacts
            restitution=0.0,       # No bounce - dead stop
            friction_combine_mode="multiply",  # Strongest friction
        )

        # ========================================
        # SIMPLE BOX GEOMETRY (No USD needed!)
        # ========================================

        # 2x2 LEGO brick dimensions (32mm x 32mm x 19.2mm)
        brick_size = (0.032, 0.032, 0.0192)

        # Replace USD-based cubes with procedural boxes
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.05, 0.0096],  # Z = half height
                rot=[1, 0, 0, 0]
            ),
            spawn=CuboidCfg(
                size=brick_size,
                rigid_props=cube_properties,
                collision_props=cube_collision,
                physics_material=cube_friction,
                visual_material=None,  # Default material
                semantic_tags=[("class", "cube_2")],
            ),
        )

        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.60, -0.10, 0.0096],  # Z = half height
                rot=[1, 0, 0, 0]
            ),
            spawn=CuboidCfg(
                size=brick_size,
                rigid_props=cube_properties,
                collision_props=cube_collision,
                physics_material=cube_friction,
                visual_material=None,
                semantic_tags=[("class", "cube_3")],
            ),
        )

        # Update randomization to match new Z height
        self.events.randomize_cube_positions.params["pose_range"]["z"] = (0.0096, 0.0096)

        # DISABLE snap-to-stack - let pure physics + friction work
        # The high friction should be sufficient for stacking
        if hasattr(self.events, 'snap_to_stack'):
            delattr(self.events, 'snap_to_stack')


@configclass
class Franka2CubeStackRLLegoJointPosRLEnvCfg_Curriculum(Franka2CubeStackRLLegoJointPosRLEnvCfg):
    """Phase 2: Add curriculum for tighter precision (optional).

    Can introduce:
    - Tighter alignment rewards
    - Stability bonuses
    - Snap rewards (reward-based, not geometry!)
    """
    pass
