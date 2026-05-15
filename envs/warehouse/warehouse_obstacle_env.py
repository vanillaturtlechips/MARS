"""Phase 1.5 — 창고 맵 (선반 장애물) + 네비게이션."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, sample_uniform

from .warehouse_env import WarehouseNavEnv, WarehouseNavEnvCfg

# ---------------------------------------------------------------------------
# 창고 선반 레이아웃
# ---------------------------------------------------------------------------
# 선반 4개: 2행 × 2열, 중앙에 십자 통로 gap
#
#   [=Shelf 0=]   [=Shelf 1=]   ← y = +2.5
#
#   ←── 메인 통로 (y ∈ -2 ~ +2) ──→
#
#   [=Shelf 2=]   [=Shelf 3=]   ← y = -2.5
#        ↑                 ↑
#    x=-2.0             x=+2.0
#    gap at x ∈ (-0.5, 0.5) → 십자 통로
# ---------------------------------------------------------------------------

# (center_x, center_y, center_z)
SHELF_CENTERS = [
    (-2.0,  2.5, 0.75),
    ( 2.0,  2.5, 0.75),
    (-2.0, -2.5, 0.75),
    ( 2.0, -2.5, 0.75),
]
SHELF_SIZE = (3.0, 0.5, 1.5)   # (x길이, y폭, z높이)
SHELF_HALF = tuple(s / 2 for s in SHELF_SIZE)

# 로봇이 선반에 너무 가까울 때 패널티 적용 거리
PROX_WARN  = 0.8   # 경고 거리 [m]
PROX_CRIT  = 0.4   # 충돌 판정 거리 [m]


def _shelf_aabb_dist(pos_xy: torch.Tensor) -> torch.Tensor:
    """각 env의 로봇 위치 → 가장 가까운 선반까지의 AABB 거리 (N,)."""
    device = pos_xy.device
    n = pos_xy.shape[0]
    min_dist = torch.full((n,), 1e6, device=device)

    for cx, cy, _ in SHELF_CENTERS:
        # AABB: diff = |robot - center| - half_size, clamp ≥ 0, then norm
        diff = torch.stack([
            (pos_xy[:, 0] - cx).abs() - SHELF_HALF[0],
            (pos_xy[:, 1] - cy).abs() - SHELF_HALF[1],
        ], dim=1).clamp(min=0.0)
        dist = diff.norm(dim=1)
        min_dist = torch.minimum(min_dist, dist)

    return min_dist


def _goal_in_shelf(goal_xy: torch.Tensor) -> torch.Tensor:
    """goal_xy (N,2) 중 선반 안에 있는 것을 True로 반환."""
    device = goal_xy.device
    inside = torch.zeros(goal_xy.shape[0], dtype=torch.bool, device=device)
    margin = 0.3  # 로봇 반폭 고려
    for cx, cy, _ in SHELF_CENTERS:
        in_x = (goal_xy[:, 0] - cx).abs() < SHELF_HALF[0] + margin
        in_y = (goal_xy[:, 1] - cy).abs() < SHELF_HALF[1] + margin
        inside |= (in_x & in_y)
    return inside


@configclass
class WarehouseObstacleNavEnvCfg(WarehouseNavEnvCfg):
    observation_space = 7   # +1: min_obstacle_dist

    # 장애물 패널티 (VF 붕괴 방지를 위해 작게)
    rew_prox_warn: float  = -0.1
    rew_prox_crit: float  = -1.0


class WarehouseObstacleNavEnv(WarehouseNavEnv):
    cfg: WarehouseObstacleNavEnvCfg

    # ------------------------------------------------------------------
    # Scene: 기존 로봇 + 지면 위에 선반 추가
    # ------------------------------------------------------------------
    def _setup_scene(self):
        from isaaclab.assets import RigidObject
        self.robot = RigidObject(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # 선반 4개 → env_0 아래 스폰, clone_environments 가 복제함
        shelf_cfg = sim_utils.CuboidCfg(
            size=SHELF_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.38, 0.18), metallic=0.0
            ),
        )
        for i, (cx, cy, cz) in enumerate(SHELF_CENTERS):
            shelf_cfg.func(
                f"/World/envs/env_0/Shelf_{i}",
                shelf_cfg,
                translation=(cx, cy, cz),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Observations: 기존 6개 + min_obstacle_dist
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        base_obs = super()._get_observations()["policy"]  # (N, 6)

        pos_w   = self.robot.data.root_pos_w[:, :2]
        local_pos = pos_w - self.scene.env_origins[:, :2]
        min_dist  = _shelf_aabb_dist(local_pos).unsqueeze(1).clamp(max=5.0)  # (N,1)

        obs = torch.cat([base_obs, min_dist], dim=1)  # (N, 7)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards: 기존 + 근접 패널티
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        base_rew = super()._get_rewards()

        pos_w     = self.robot.data.root_pos_w[:, :2]
        local_pos = pos_w - self.scene.env_origins[:, :2]
        dist      = _shelf_aabb_dist(local_pos)

        warn = (dist < PROX_WARN) & (dist >= PROX_CRIT)
        crit = dist < PROX_CRIT

        prox_rew = (
            self.cfg.rew_prox_warn * warn.float()
            + self.cfg.rew_prox_crit * crit.float()
        )
        return base_rew + prox_rew

    # ------------------------------------------------------------------
    # Reset: 선반 안에 골 놓이지 않도록 rejection sampling
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # 부모 reset (로봇 자세 + 초기 골 샘플링)
        super()._reset_idx(env_ids)

        # env_ids를 long 텐서로 통일
        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.long()

        # 부모가 이미 set한 골 중 선반과 겹치는 것을 재샘플
        goals = self._goal_pos_w[env_ids_t].clone()
        local = goals - self.scene.env_origins[env_ids_t, :2]
        remaining = _goal_in_shelf(local)          # (N,) bool

        for _ in range(20):
            if not remaining.any():
                break
            rem = remaining.nonzero(as_tuple=True)[0]   # remaining 인덱스
            n_rem = rem.shape[0]
            angle = sample_uniform(-math.pi, math.pi, (n_rem,), device=self.device)
            dist  = sample_uniform(self.cfg.goal_min_dist, self.cfg.goal_range,
                                   (n_rem,), device=self.device)
            origins   = self.scene.env_origins[env_ids_t[rem], :2]
            candidate = origins + torch.stack(
                [dist * torch.cos(angle), dist * torch.sin(angle)], dim=1
            )
            bad = _goal_in_shelf(candidate - origins)
            good = ~bad
            goals[rem[good]]      = candidate[good]
            remaining[rem[good]]  = False

        self._goal_pos_w[env_ids_t] = goals
