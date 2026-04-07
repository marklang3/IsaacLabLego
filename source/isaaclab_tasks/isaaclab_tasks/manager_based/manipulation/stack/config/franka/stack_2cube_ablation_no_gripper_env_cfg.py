"""
ABLATION 2: Remove gripper state observations (2D).

Tests if explicit gripper finger positions are necessary
or if the policy can infer gripper state from other observations.

Observation space: 58D → 56D
- Removes: gripper_pos (2D)
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
    """Observation specifications for ablation study: NO GRIPPER STATE."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - without gripper state."""

        # Proprioception (17D total, was 26D)
        actions = ObsTerm(func=mdp.last_action)  # 8D
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 9D
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 9D
        # gripper_pos REMOVED

        # Object observations with relative positions (23D)
        object = ObsTerm(func=mdp.object_obs_2cube)

        # End-effector observations (7D)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)  # 3D
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)  # 4D

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Franka2CubeStackAblationNoGripperEnvCfg(Franka2CubeStackSmoothFixedJointEnvCfg):
    """Environment configuration for ablation study: remove gripper state observations.

    Expected observation dimension: 56D
    """

    def __post_init__(self):
        super().__post_init__()
        # Replace observations with ablation config
        self.observations = ObservationsCfg()
        self.observations.policy.enable_corruption = False
