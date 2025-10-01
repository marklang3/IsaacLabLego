# SPDX-License-Identifier: BSD-3-Clause
# Minimal reward helpers for stacking (keeps cfg unchanged)

from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary reward if object z is above minimal_height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return (obj.data.root_pos_w[:, 2] > minimal_height).to(torch.float32)


def object_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Shaped reward for EE reaching the object: 1 - tanh(||p_obj - p_ee|| / std)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos = obj.data.root_pos_w                        # (E, 3)
    ee_pos = ee.data.target_pos_w[..., 0, :]             # (E, 3) first target frame
    dist = torch.norm(obj_pos - ee_pos, dim=-1)          # (E,)
    return 1.0 - torch.tanh(dist / std)


def horizontal_alignment(
    env: "ManagerBasedRLEnv",
    src_asset_cfg: SceneEntityCfg,
    tgt_asset_cfg: SceneEntityCfg,
    std: float = 0.06,
) -> torch.Tensor:
    """Gaussian reward for XY center alignment between src and tgt."""
    src: RigidObject = env.scene[src_asset_cfg.name]
    tgt: RigidObject = env.scene[tgt_asset_cfg.name]
    dxy = src.data.root_pos_w[:, :2] - tgt.data.root_pos_w[:, :2]
    dist2 = (dxy * dxy).sum(-1)
    return torch.exp(-dist2 / (2.0 * std * std + 1e-9))


def object_stability(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    lin_vel_thr: float = 0.05,
    ang_vel_thr: float = 0.2,
) -> torch.Tensor:
    """Encourage low linear & angular speeds after placement (smooth, in (0,1])."""
    obj: RigidObject = env.scene[object_cfg.name]
    lin_speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=-1)
    ang_speed = torch.linalg.norm(obj.data.root_ang_vel_w, dim=-1)
    lin_excess = torch.clamp(lin_speed - lin_vel_thr, min=0.0) / (lin_vel_thr + 1e-9)
    ang_excess = torch.clamp(ang_speed - ang_vel_thr, min=0.0) / (ang_vel_thr + 1e-9)
    return torch.exp(-(lin_excess + ang_excess))
