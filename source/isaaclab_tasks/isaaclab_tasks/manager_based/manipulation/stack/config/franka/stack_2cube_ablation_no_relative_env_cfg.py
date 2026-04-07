"""
ABLATION 1: Remove relative position observations (9D).

Tests if explicit relative positions (gripper-to-cube, cube-to-cube) are necessary
or if the policy can infer them from absolute positions.

Observation space: 58D → 49D
- Removes: gripper_to_cube_2 (3D), gripper_to_cube_3 (3D), cube_2_to_3 (3D)
- Keeps: All proprioception, cube poses, eef pose, gripper state
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
    """Observation specifications for ablation study: NO RELATIVE POSITIONS."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - without relative positions."""

        # Proprioception (26D total)
        actions = ObsTerm(func=mdp.last_action)  # 8D
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 9D
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 9D

        # Object observations WITHOUT relative positions (14D)
        object = ObsTerm(func=mdp.object_obs_2cube_no_relative)

        # End-effector observations (7D)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)  # 3D
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)  # 4D

        # Gripper state (2D)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Franka2CubeStackAblationNoRelativeEnvCfg(Franka2CubeStackSmoothFixedJointEnvCfg):
    """Environment configuration for ablation study: remove relative position observations.

    Expected observation dimension: 49D
    """

    def __post_init__(self):
        super().__post_init__()
        # Replace observations with ablation config
        self.observations = ObservationsCfg()
        self.observations.policy.enable_corruption = False
