"""
ABLATION 3: Remove end-effector pose observations (7D).

Tests if explicit end-effector tracking is necessary
or if the policy can infer EE pose from joint positions.

Observation space: 58D → 51D
- Removes: eef_pos (3D), eef_quat (4D)
- Keeps: All other observations including relative positions
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
    """Observation specifications for ablation study: NO END-EFFECTOR POSE."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - without end-effector pose."""

        # Proprioception (26D total)
        actions = ObsTerm(func=mdp.last_action)  # 8D
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 9D
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 9D
        gripper_pos = ObsTerm(func=mdp.gripper_pos)  # 2D

        # Object observations with relative positions (23D)
        object = ObsTerm(func=mdp.object_obs_2cube_no_eef)

        # eef_pos REMOVED
        # eef_quat REMOVED

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Franka2CubeStackAblationNoEEFEnvCfg(Franka2CubeStackSmoothFixedJointEnvCfg):
    """Environment configuration for ablation study: remove end-effector pose observations.

    Expected observation dimension: 51D
    """

    def __post_init__(self):
        super().__post_init__()
        # Replace observations with ablation config
        self.observations = ObservationsCfg()
        self.observations.policy.enable_corruption = False
