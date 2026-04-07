# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
LEGO 3-Cube Stack environment with VISION + RL rewards using JOINT POSITION control.

This combines:
1. Vision capabilities (wrist + table cameras with domain randomization)
2. LEGO-specific precision rewards (nearly_stacked, knob_cavity_alignment, stability)
3. 95-dimensional observation space (compatible with smooth blocks checkpoint)
4. Progressive curriculum learning

Based on successful precision stacking methodology achieving 4.8x improvement
over baseline (70-75% success vs 25% baseline).
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from . import stack_3cube_lego_joint_pos_rl_env_cfg


@configclass
class EventCfg:
    """Configuration for events with visual domain randomization."""

    # Cube position randomization
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.025, 0.025), "yaw": (-1.0, 1.0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

    # Visual domain randomization for sim-to-real transfer
    randomize_light = EventTerm(
        func=franka_stack_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1500.0, 10000.0),
            "color_variation": 0.4,
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Cloudy/lakeside_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hospital_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/hotel_room_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/old_bus_depot_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/small_empty_house_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Indoor/surgery_4k.hdr",
                f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr",
            ],
            "default_intensity": 3000.0,
            "default_color": (0.75, 0.75, 0.75),
            "default_texture": "",
        },
    )

    randomize_table_visual_material = EventTerm(
        func=franka_stack_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash/Ash_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Bamboo_Planks/Bamboo_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Birch/Birch_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Cherry/Cherry_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Mahogany_Planks/Mahogany_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak/Oak_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber/Timber_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Timber_Cladding/Timber_Cladding_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Walnut_Planks/Walnut_Planks_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Stone/Marble/Marble_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ],
            "default_texture": (
                f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/Materials/Textures/DemoTable_TableBase_BaseColor.png"
            ),
        },
    )

    randomize_robot_arm_visual_texture = EventTerm(
        func=franka_stack_events.randomize_visual_texture_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "textures": [
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Cast/Aluminum_Cast_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Polished/Aluminum_Polished_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brass/Brass_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Bronze/Bronze_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Brushed_Antique_Copper/Brushed_Antique_Copper_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Cast_Metal_Silver_Vein/Cast_Metal_Silver_Vein_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Copper/Copper_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Gold/Gold_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Iron/Iron_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/RustedMetal/RustedMetal_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Silver/Silver_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Carbon/Steel_Carbon_BaseColor.png",
                f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Steel_Stainless/Steel_Stainless_BaseColor.png",
            ],
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP with vision."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values and vision."""

        # Proprioceptive observations (same as non-vision config)
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # Vision observations - ResNet18 features (512 dims each)
        # Using image_features for CNN encoding instead of raw pixels
        table_cam_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "rgb",
                "model_name": "resnet18",  # 512-dim feature vector
            },
        )
        wrist_cam_features = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "model_name": "resnet18",  # 512-dim feature vector
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class Franka3CubeStackLegoVisuomotorJointPosRLEnvCfg(
    stack_3cube_lego_joint_pos_rl_env_cfg.Franka3CubeStackLegoJointPosRLEnvCfg
):
    """LEGO 3-Cube Stack environment with VISION + RL rewards using JOINT POSITION control.

    This configuration combines:
    - Vision: Wrist + table cameras with ResNet18 feature extraction
    - LEGO rewards: Precision stacking with knob-cavity alignment
    - Observation space: ~1,119 dims (95 proprioceptive + 512 + 512 visual features)
    - Domain randomization: Lighting, textures for sim-to-real transfer
    """

    def __post_init__(self):
        # post init of parent (gets LEGO rewards, physics, cubes)
        super().__post_init__()

        # Override observations to include vision
        self.observations = ObservationsCfg()

        # Set events with domain randomization
        self.events = EventCfg()

        # Add cameras to scene
        # Wrist camera (mounted on gripper) - high res for LEGO knob detail
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,  # Higher resolution for LEGO details
            width=200,
            data_types=["rgb", "distance_to_image_plane"],  # RGB + depth
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
            ),
        )

        # Table camera (overhead view) - for spatial reasoning
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,  # Higher resolution for LEGO details
            width=200,
            data_types=["rgb", "distance_to_image_plane"],  # RGB + depth
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )

        # Set camera rendering settings
        self.rerender_on_reset = True  # Re-render cameras on environment reset
        self.sim.render.antialiasing_mode = "OFF"  # Disable DLSS for determinism

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam_features", "wrist_cam_features"]

        # Note: Rewards are inherited from Franka3CubeStackLegoJointPosRLEnvCfg
        # This includes:
        # - IsaacGym hierarchical rewards (reach, lift, align, stack)
        # - Nearly stacked reward (bridges precision gap)
        # - Knob-cavity alignment reward (LEGO-specific)
        # - Stability penalty (reduces hovering)
        # - Progressive curriculum (2-stage)
