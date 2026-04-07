"""
ABLATION: Remove absolute object pose observations (14D).

Tests if the policy can learn with only relative spatial relationships,
without absolute position/orientation of objects in world frame.

Observation space: 58D → 44D
- Removes: cube_2 absolute pos (3D), cube_2 absolute quat (4D),
           cube_3 absolute pos (3D), cube_3 absolute quat (4D)
- Keeps: relative distances (gripper-to-cubes, cube-to-cube: 9D),
         proprioception (26D), end-effector (7D), gripper (2D)
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
    """Observation specifications for ablation study: NO ABSOLUTE OBJECT POSES."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - without absolute object poses."""

        # Proprioception (26D)
        actions = ObsTerm(func=mdp.last_action)  # 8D
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 9D
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 9D

        # Object observations - ONLY relative positions (9D)
        # Uses custom function that excludes absolute poses
        object = ObsTerm(func=mdp.object_obs_2cube_relative_only)

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
class Franka2CubeStackAblationNoAbsoluteEnvCfg(Franka2CubeStackSmoothFixedJointEnvCfg):
    """Environment configuration for ablation study: remove absolute object pose observations.

    Expected observation dimension: 44D (26D proprio + 9D relative + 7D eef + 2D gripper)
    """

    def __post_init__(self):
        super().__post_init__()
        # Replace observations with ablation config
        self.observations = ObservationsCfg()
        self.observations.policy.enable_corruption = False
