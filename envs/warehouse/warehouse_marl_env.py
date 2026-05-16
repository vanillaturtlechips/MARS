"""Phase 3 — 멀티 로봇 창고 환경 (로봇 N대 동시 운용).

Phase 1.5 WarehouseObstacleNavEnv 를 확장:
  - 동일 env 안에 로봇 N대 스폰
  - 각 로봇 독립 관측 (자기 goal + 선반 거리 + 다른 로봇까지 거리)
  - MPG 보상 (potential_reward.py)
  - 로봇 간 충돌 감지 → episode terminated
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, sample_uniform

from .warehouse_obstacle_env import SHELF_CENTERS, SHELF_HALF, _shelf_aabb_dist, _goal_in_shelf

# 관측 구성 (로봇 1대 기준)
#   goal_x_body, goal_y_body, goal_dist   : 3
#   vx_body, vy_body, omega_z             : 3
#   min_shelf_dist                        : 1
#   other_robot_dist × (N-1)              : N-1
# 합계: 7 + (N-1)

N_ROBOTS = 3
OBS_PER_ROBOT = 7 + (N_ROBOTS - 1)   # 9

# 로봇 간 충돌 판정 거리
ROBOT_COLLISION_DIST = 0.55   # m (로봇 폭 0.5m + 여유)

# 스폰 오프셋: 로봇 3대를 env 원점 기준으로 분산 배치
SPAWN_OFFSETS = [
    (-1.5, -1.5),
    ( 1.5, -1.5),
    ( 0.0,  1.5),
]


@configclass
class WarehouseMARLEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 3 * N_ROBOTS          # 각 로봇 [vx, vy, omega]
    observation_space = OBS_PER_ROBOT * N_ROBOTS
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=14.0, replicate_physics=True
    )

    max_vx: float = 1.5
    max_vy: float = 1.0
    max_omega: float = 2.0

    goal_radius: float = 0.35
    goal_range: float = 4.0
    goal_min_dist: float = 1.0

    # MPG 가중치 (beta 낮춰 self reward가 dominant하도록)
    alpha: float = 1.0
    beta: float = 0.3

    rew_collision: float = -5.0   # 로봇 간 충돌 패널티 (IPPO 단계에서 사용)


class WarehouseMARLEnv(DirectRLEnv):
    cfg: WarehouseMARLEnvCfg

    def __init__(self, cfg: WarehouseMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._goal_pos_w = torch.zeros(self.num_envs, N_ROBOTS, 2, device=self.device)
        self._actions = torch.zeros(self.num_envs, N_ROBOTS, 3, device=self.device)

    # ------------------------------------------------------------------
    # Scene: 로봇 N대 + 선반 4개
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # 로봇 N대 등록
        self.robots: list[RigidObject] = []
        for i in range(N_ROBOTS):
            robot_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Robot_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.4, 0.3),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        linear_damping=2.0,
                        angular_damping=5.0,
                        max_linear_velocity=5.0,
                        max_angular_velocity=10.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            (0.2, 0.4, 0.8),   # 파랑
                            (0.8, 0.3, 0.2),   # 빨강
                            (0.2, 0.7, 0.3),   # 초록
                        )[i],
                        metallic=0.1,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(SPAWN_OFFSETS[i][0], SPAWN_OFFSETS[i][1], 0.15)
                ),
            )
            robot = RigidObject(robot_cfg)
            self.robots.append(robot)

        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        shelf_cfg_base = sim_utils.CuboidCfg(
            size=(3.0, 0.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.38, 0.18), metallic=0.0
            ),
        )
        for s_i, (cx, cy, cz) in enumerate(SHELF_CENTERS):
            shelf_cfg_base.func(
                f"/World/envs/env_0/Shelf_{s_i}",
                shelf_cfg_base,
                translation=(cx, cy, cz),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, robot in enumerate(self.robots):
            self.scene.rigid_objects[f"robot_{i}"] = robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: (N, N_ROBOTS * 3) → (N, N_ROBOTS, 3)
        self._actions = actions.clone().clamp(-1.0, 1.0).view(self.num_envs, N_ROBOTS, 3)

    def _apply_action(self) -> None:
        for i, robot in enumerate(self.robots):
            quat = robot.data.root_quat_w
            _, _, yaw = euler_xyz_from_quat(quat)
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)

            vx_b = self._actions[:, i, 0] * self.cfg.max_vx
            vy_b = self._actions[:, i, 1] * self.cfg.max_vy
            omega = self._actions[:, i, 2] * self.cfg.max_omega

            vel_cmd = torch.zeros(self.num_envs, 6, device=self.device)
            vel_cmd[:, 0] = cos_yaw * vx_b - sin_yaw * vy_b
            vel_cmd[:, 1] = sin_yaw * vx_b + cos_yaw * vy_b
            vel_cmd[:, 5] = omega
            robot.write_root_velocity_to_sim(vel_cmd)

    # ------------------------------------------------------------------
    # Observations: 각 로봇 OBS_PER_ROBOT 차원 → concat
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        obs_list = []
        for i, robot in enumerate(self.robots):
            pos_w = robot.data.root_pos_w[:, :2]
            quat  = robot.data.root_quat_w
            lin_vel_w = robot.data.root_lin_vel_w
            ang_vel_w = robot.data.root_ang_vel_w

            goal_vec_w = self._goal_pos_w[:, i] - pos_w
            goal_dist  = goal_vec_w.norm(dim=1, keepdim=True).clamp(max=10.0)
            goal_3d    = torch.cat([goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
            goal_body  = quat_apply_inverse(quat, goal_3d)[:, :2]
            vel_body   = quat_apply_inverse(quat, lin_vel_w)[:, :2]
            omega_z    = ang_vel_w[:, 2:3]

            local_pos  = pos_w - self.scene.env_origins[:, :2]
            shelf_dist = _shelf_aabb_dist(local_pos).unsqueeze(1).clamp(max=5.0)

            # 다른 로봇까지 거리
            other_dists = []
            for j, other in enumerate(self.robots):
                if j == i:
                    continue
                other_pos = other.data.root_pos_w[:, :2]
                d = (pos_w - other_pos).norm(dim=1, keepdim=True).clamp(max=6.0)
                other_dists.append(d)

            obs_i = torch.cat([goal_body, goal_dist, vel_body, omega_z, shelf_dist] + other_dists, dim=1)
            obs_list.append(obs_i)

        obs = torch.cat(obs_list, dim=1)   # (N, N_ROBOTS * OBS_PER_ROBOT)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards: MPG (potential_reward.py) + IPPO 단계는 단순 충돌 패널티
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        from training.multi_robot.potential_reward import all_robots_mpg_reward

        positions = torch.stack([r.data.root_pos_w[:, :2] for r in self.robots], dim=1)
        rewards_per_robot = all_robots_mpg_reward(
            positions, self._goal_pos_w,
            alpha=self.cfg.alpha, beta=self.cfg.beta,
            goal_radius=self.cfg.goal_radius,
            time_step=self.episode_length_buf,
        )   # (N, N_ROBOTS)

        return rewards_per_robot.mean(dim=1)   # (N,) — 로봇 수로 나눠 스케일 정규화

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        any_oob     = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        collision   = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        positions = []
        for i, robot in enumerate(self.robots):
            pos_w = robot.data.root_pos_w[:, :2]
            positions.append(pos_w)
            dist = (pos_w - self._goal_pos_w[:, i]).norm(dim=1)
            all_reached &= (dist < self.cfg.goal_radius)
            local = pos_w - self.scene.env_origins[:, :2]
            any_oob |= local.abs().max(dim=1).values > 8.0

        # 로봇 간 충돌 감지
        for i in range(N_ROBOTS):
            for j in range(i + 1, N_ROBOTS):
                d = (positions[i] - positions[j]).norm(dim=1)
                collision |= (d < ROBOT_COLLISION_DIST)

        terminated = all_reached | any_oob | collision
        timed_out  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robots[0]._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.long()
        n = env_ids_t.shape[0]

        for i, robot in enumerate(self.robots):
            default_state = robot.data.default_root_state[env_ids_t].clone()
            default_state[:, :3] += self.scene.env_origins[env_ids_t]
            default_state[:, 0] += SPAWN_OFFSETS[i][0]
            default_state[:, 1] += SPAWN_OFFSETS[i][1]
            default_state[:, :2] += sample_uniform(-0.3, 0.3, (n, 2), device=self.device)

            yaw = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
            zeros = torch.zeros(n, device=self.device)
            default_state[:, 3] = torch.cos(yaw / 2)
            default_state[:, 4] = zeros
            default_state[:, 5] = zeros
            default_state[:, 6] = torch.sin(yaw / 2)
            default_state[:, 7:] = 0.0
            robot.write_root_state_to_sim(default_state, env_ids_t)

        # 로봇별 목표 샘플링 (선반과 겹치지 않도록)
        for i in range(N_ROBOTS):
            angle = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
            dist  = sample_uniform(self.cfg.goal_min_dist, self.cfg.goal_range, (n,), device=self.device)
            origins = self.scene.env_origins[env_ids_t, :2]
            candidates = origins + torch.stack(
                [dist * torch.cos(angle), dist * torch.sin(angle)], dim=1
            )
            local = candidates - origins
            bad   = _goal_in_shelf(local)
            for _ in range(20):
                if not bad.any():
                    break
                rem = bad.nonzero(as_tuple=True)[0]
                n_rem = rem.shape[0]
                a2 = sample_uniform(-math.pi, math.pi, (n_rem,), device=self.device)
                d2 = sample_uniform(self.cfg.goal_min_dist, self.cfg.goal_range, (n_rem,), device=self.device)
                cand2 = origins[rem] + torch.stack([d2 * torch.cos(a2), d2 * torch.sin(a2)], dim=1)
                b2 = _goal_in_shelf(cand2 - origins[rem])
                candidates[rem[~b2]] = cand2[~b2]
                bad[rem[~b2]] = False

            self._goal_pos_w[env_ids_t, i] = candidates
