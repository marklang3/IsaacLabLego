"""
ABLATION: Remove proprioception observations (26D).

Tests if the policy can learn without internal state information
(actions, joint positions/velocities), relying only on external
observations (objects, end-effector, gripper).

Observation space: 58D → 32D
- Removes: actions (8D), joint_pos (9D), joint_vel (9D)
- Keeps: object state (23D), end-effector (7D), gripper (2D)
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
    """Observation specifications for ablation study: NO PROPRIOCEPTION."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - without proprioception."""

        # Proprioception REMOVED (was 26D)
        # actions REMOVED (8D)
        # joint_pos REMOVED (9D)
        # joint_vel REMOVED (9D)

        # Object observations with relative positions (23D)
        object = ObsTerm(func=mdp.object_obs_2cube)

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
class Franka2CubeStackAblationNoProprioEnvCfg(Franka2CubeStackSmoothFixedJointEnvCfg):
    """Environment configuration for ablation study: remove proprioception observations.

    Expected observation dimension: 32D (23D object + 7D eef + 2D gripper)
    """

    def __post_init__(self):
        super().__post_init__()
        # Replace observations with ablation config
        self.observations = ObservationsCfg()
        self.observations.policy.enable_corruption = False
