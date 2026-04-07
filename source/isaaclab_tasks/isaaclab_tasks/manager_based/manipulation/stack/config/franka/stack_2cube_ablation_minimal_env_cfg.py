"""
ABLATION 4: Minimal observation space.

Tests the absolute minimum observations needed: proprioception + cube poses only.
Removes all derived observations (relative positions, EE tracking, gripper state).

Observation space: 58D → 40D
- Removes: relative positions (9D), eef_pos (3D), eef_quat (4D), gripper_pos (2D)
- Keeps: actions (8D), joint_pos (9D), joint_vel (9D), cube poses (14D)
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_2cube_smooth_fixed_joint_env_cfg import (
    Franka2CubeStackSmoothFixedJointEnvCfg,
)


@configclass
class ObservationsCfg:
    """Observation specifications for ablation study: MINIMAL."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Minimal observations: proprioception + cube poses only."""

        # Proprioception (26D total)
        actions = ObsTerm(func=mdp.last_action)  # 8D
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 9D
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 9D

        # Object observations - cube poses only (14D)
        object = ObsTerm(func=mdp.object_obs_2cube_minimal)

        # ALL REMOVED:
        # - eef_pos (3D)
        # - eef_quat (4D)
        # - gripper_pos (2D)
        # - relative positions (9D, removed in object_obs_2cube_minimal)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Franka2CubeStackAblationMinimalEnvCfg(Franka2CubeStackSmoothFixedJointEnvCfg):
    """Environment configuration for ablation study: minimal observation space.

    Expected observation dimension: 40D (26D proprioception + 14D cube poses)
    """

    def __post_init__(self):
        super().__post_init__()
        # Replace observations with ablation config
        self.observations = ObservationsCfg()
        self.observations.policy.enable_corruption = False
