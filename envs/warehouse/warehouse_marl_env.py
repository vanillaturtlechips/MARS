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
#   goal_x_body, goal_y_body, goal_dist              : 3
#   vx_body, vy_body, omega_z                        : 3
#   min_shelf_dist                                   : 1
#   other_robot_dx, dy, dist, vx_rel, vy_rel × (N-1): 5*(N-1)
# 합계: 7 + 5*(N-1)

N_ROBOTS = 3
OBS_PER_ROBOT = 7 + 5 * (N_ROBOTS - 1)   # 17
OBS_PER_ROBOT_BATTERY = OBS_PER_ROBOT + 3  # 20: + battery_level(1) + 충전소 상대위치(2)


def obs_per_robot(enable_battery: bool) -> int:
    return OBS_PER_ROBOT_BATTERY if enable_battery else OBS_PER_ROBOT

# 로봇 간 충돌 판정 거리
ROBOT_COLLISION_DIST = 0.55   # m (로봇 폭 0.5m + 여유)

# 스폰 오프셋: 로봇 3대를 env 원점 기준으로 분산 배치
SPAWN_OFFSETS = [
    (-1.5, -1.5),
    ( 1.5, -1.5),
    ( 0.0,  1.5),
]

# 충전소 위치 (env-local) — y=0 중앙 통로(선반 ±2.5와 안 겹침), 로봇 경로 근처
# 코너(±5)는 random 탐색이 도달 못 해 charging 경험 0 → 학습 불가였음. 접근성 개선.
CHARGER_POSITIONS = [(-3.0, 0.0), (3.0, 0.0)]


@configclass
class WarehouseMARLEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 3 * N_ROBOTS          # 각 로봇 [vx, vy, omega]
    observation_space = OBS_PER_ROBOT * N_ROBOTS
    state_space = OBS_PER_ROBOT * N_ROBOTS  # 51 — centralized critic global state (17*3)

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

    # MPG 가중치 (SAFE_DIST 마스킹과 조합)
    # alpha↑: goal tracking 강화 / beta: SAFE_DIST 내 반발력만 작동
    # model_9999.pt는 beta=0.5로 훈련됨 — 재훈련 시 1.5 사용
    alpha: float = 1.0
    beta: float = 1.5

    rew_collision: float  = -200.0  # 충돌 = 즉사 (교착 -90과 확실한 차이)
    rew_goal: float       =    6.0  # 목표 도달 보상
    rew_stationary: float =   -0.6  # S6 장애물회피 98% 달성값. -0.45 절충은 S6/S1 박살(에바)로 폐기.
                                     # S3/S4 좁은통로·동일목표 교착은 orchestrator 영역(레이어 분리)

    # 동적 장애물 (갑자기 출현하는 정지 장애물). False면 기존 동작 100% 보존
    enable_dynamic_obstacles: bool = False
    n_dynamic_obstacles: int = 2
    obstacle_size: tuple = (0.7, 0.7, 1.0)
    obstacle_spawn_step_min: int = 50   # 등장 시점(step) — 로봇이 진행한 뒤 출현(교착 완화)
    obstacle_spawn_step_max: int = 130
    obstacle_z: float = 0.5             # 등장 시 z (지하 -5.0에서 올라옴)
    # per-step 페널티 (로봇-로봇 충돌 -200은 done 1회, 장애물은 done 안 하므로 작게)
    # 과하면 끼였을 때 누적 폭발(VF loss) → -30/step: 회피 유인 유지하며 안정
    rew_obstacle_collision: float = -30.0

    # 배터리 + 충전소 (enable_battery=False면 기존 동작·obs 차원 100% 보존)
    enable_battery: bool = False
    n_chargers: int = 2
    battery_drain: float = 0.003        # step당 소모 (0.005는 학습 전 전부 방전→악순환)
    battery_charge: float = 0.02        # 충전소 근처 step당 회복
    charge_radius: float = 1.0          # 충전 판정 거리 (0.6→1.0: 도달·머물기 쉽게)
    battery_init_min: float = 0.4       # 초기 여유 (충전 학습 시간 확보, 학습 전 즉시방전 방지)
    battery_init_max: float = 1.0
    battery_low_thresh: float = 0.5     # 조기 경보 (충전소 더 일찍 인지)
    rew_charging: float = 2.0           # 충전소 위 충전 중 보상 (urgency 가중) — 양의 가치 앵커
    rew_charger_progress: float = 5.0   # 충전소 접근 progress (potential-based, goal 방해 안 함) — 강화
    rew_depleted: float = -0.3          # 방전 페널티 완화 (-5→-0.3, 음수 지배 제거)
    # 방전 시 속도 급감 — 충전을 생존 필수로 만듦 (방전되면 goal 못 감 → 충전 학습 강제)
    battery_speed_thresh: float = 0.2   # 이 이상이면 풀속도
    min_speed_scale: float = 0.3        # 방전 시 0.3배(느리지만 충전소 도달 가능) — 0.0은 영구정지 악순환


class WarehouseMARLEnv(DirectRLEnv):
    cfg: WarehouseMARLEnvCfg

    def __init__(self, cfg: WarehouseMARLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._goal_pos_w = torch.zeros(self.num_envs, N_ROBOTS, 2, device=self.device)
        self._actions = torch.zeros(self.num_envs, N_ROBOTS, 3, device=self.device)
        self._prev_positions: torch.Tensor | None = None   # delta repulsion 버퍼

        # 동적 장애물 상태 버퍼
        if self.cfg.enable_dynamic_obstacles:
            no = self.cfg.n_dynamic_obstacles
            d = self.device
            self._obst_pos_local  = torch.zeros(self.num_envs, no, 2, device=d)  # env-local 등장 위치
            self._obst_spawn_step = torch.zeros(self.num_envs, no, device=d)     # 등장 시점(step)
            self._obst_active     = torch.zeros(self.num_envs, no, dtype=torch.bool, device=d)

        # 배터리 버퍼 (로봇별 0~1) + 충전소 거리 progress 버퍼
        if self.cfg.enable_battery:
            self._battery = torch.ones(self.num_envs, N_ROBOTS, device=self.device)
            self._prev_charger_dist = torch.zeros(self.num_envs, N_ROBOTS, device=self.device)

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

        # 동적 장애물: kinematic cuboid, 초기 지하 -5m (등장 전)
        self.obstacles: list[RigidObject] = []
        if self.cfg.enable_dynamic_obstacles:
            for o in range(self.cfg.n_dynamic_obstacles):
                obst_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/DynObstacle_{o}",
                    spawn=sim_utils.CuboidCfg(
                        size=self.cfg.obstacle_size,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.9, 0.5, 0.1), metallic=0.0
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
                )
                self.obstacles.append(RigidObject(obst_cfg))

        # 충전소: 낮은 패드(시각 표식, collision 없이 로봇이 위로 지나감)
        if self.cfg.enable_battery:
            charger_cfg = sim_utils.CuboidCfg(
                size=(1.0, 1.0, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.9, 0.3), metallic=0.0
                ),
            )
            for c, (cx, cy) in enumerate(CHARGER_POSITIONS[:self.cfg.n_chargers]):
                charger_cfg.func(
                    f"/World/envs/env_0/Charger_{c}", charger_cfg,
                    translation=(cx, cy, 0.025),
                    orientation=(1.0, 0.0, 0.0, 0.0),
                )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, robot in enumerate(self.robots):
            self.scene.rigid_objects[f"robot_{i}"] = robot
        for o, obst in enumerate(self.obstacles):
            self.scene.rigid_objects[f"obstacle_{o}"] = obst

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions: (N, N_ROBOTS * 3) → (N, N_ROBOTS, 3)
        self._actions = actions.clone().clamp(-1.0, 1.0).view(self.num_envs, N_ROBOTS, 3)
        self._update_obstacles()
        self._update_battery()

    def _update_battery(self) -> None:
        """step마다 배터리 소모, 충전소 charge_radius 이내면 회복."""
        if not self.cfg.enable_battery:
            return
        self._battery -= self.cfg.battery_drain
        origins = self.scene.env_origins[:, :2]
        for i, robot in enumerate(self.robots):
            local = robot.data.root_pos_w[:, :2] - origins   # (N, 2)
            on_charger = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for cx, cy in CHARGER_POSITIONS[:self.cfg.n_chargers]:
                d = ((local[:, 0] - cx) ** 2 + (local[:, 1] - cy) ** 2).sqrt()
                on_charger |= (d < self.cfg.charge_radius)
            self._battery[:, i] = torch.where(
                on_charger,
                self._battery[:, i] + self.cfg.battery_charge,
                self._battery[:, i],
            )
        self._battery.clamp_(0.0, 1.0)

    def _update_obstacles(self) -> None:
        """episode_length_buf가 등장 시점 도달 시 장애물을 지하→지상으로 올림 (kinematic)."""
        if not self.cfg.enable_dynamic_obstacles:
            return
        origins = self.scene.env_origins  # (N, 3)
        for o, obst in enumerate(self.obstacles):
            active = self.episode_length_buf >= self._obst_spawn_step[:, o]   # (N,)
            self._obst_active[:, o] = active
            state = obst.data.default_root_state.clone()                      # (N, 13)
            state[:, 0] = origins[:, 0] + self._obst_pos_local[:, o, 0]
            state[:, 1] = origins[:, 1] + self._obst_pos_local[:, o, 1]
            state[:, 2] = torch.where(
                active,
                torch.full_like(state[:, 2], self.cfg.obstacle_z),
                torch.full_like(state[:, 2], -5.0),
            )
            state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            state[:, 7:]  = 0.0
            obst.write_root_state_to_sim(state)

    def _dynamic_obstacle_dist(self, local_pos: torch.Tensor) -> torch.Tensor:
        """로봇 local 위치 → 가장 가까운 '등장한' 동적 장애물까지 AABB 거리 (N,).
        비활성(지하)·비활성화 시 무한대 → min 연산에 영향 없음 (obs 분포 보존)."""
        n = local_pos.shape[0]
        min_d = torch.full((n,), 1e6, device=self.device)
        if not self.cfg.enable_dynamic_obstacles:
            return min_d
        half_x = self.cfg.obstacle_size[0] / 2.0
        half_y = self.cfg.obstacle_size[1] / 2.0
        for o in range(self.cfg.n_dynamic_obstacles):
            ox = self._obst_pos_local[:, o, 0]
            oy = self._obst_pos_local[:, o, 1]
            diff = torch.stack([
                (local_pos[:, 0] - ox).abs() - half_x,
                (local_pos[:, 1] - oy).abs() - half_y,
            ], dim=1).clamp(min=0.0)
            d = diff.norm(dim=1)
            d = torch.where(self._obst_active[:, o], d, torch.full_like(d, 1e6))
            min_d = torch.minimum(min_d, d)
        return min_d

    def _apply_action(self) -> None:
        for i, robot in enumerate(self.robots):
            quat = robot.data.root_quat_w
            _, _, yaw = euler_xyz_from_quat(quat)
            cos_yaw = torch.cos(yaw)
            sin_yaw = torch.sin(yaw)

            vx_b = self._actions[:, i, 0] * self.cfg.max_vx
            vy_b = self._actions[:, i, 1] * self.cfg.max_vy
            omega = self._actions[:, i, 2] * self.cfg.max_omega

            # 방전 시 속도 급감 — battery 낮으면 못 움직임 → 충전이 goal 도달의 전제
            if self.cfg.enable_battery:
                scale = (self._battery[:, i] / self.cfg.battery_speed_thresh).clamp(
                    self.cfg.min_speed_scale, 1.0)
                vx_b  = vx_b * scale
                vy_b  = vy_b * scale
                omega = omega * scale

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
            # 선반 + 등장한 동적 장애물 중 가장 가까운 거리 (obs 차원 유지: 1채널)
            nearest = torch.minimum(_shelf_aabb_dist(local_pos),
                                    self._dynamic_obstacle_dist(local_pos))
            shelf_dist = nearest.unsqueeze(1).clamp(max=5.0)

            # 다른 로봇까지 상대 위치(body frame) + 거리 + 상대 속도 — 5-dim per robot
            other_dists = []
            for j, other in enumerate(self.robots):
                if j == i:
                    continue
                other_pos   = other.data.root_pos_w[:, :2]
                other_vel_w = other.data.root_lin_vel_w[:, :2]

                rel_w  = other_pos - pos_w
                rel_3d = torch.cat([rel_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
                rel_body_xy = quat_apply_inverse(quat, rel_3d)[:, :2]
                d = rel_w.norm(dim=1, keepdim=True).clamp(max=6.0)

                vel_rel_w  = other_vel_w - lin_vel_w[:, :2]
                vel_rel_3d = torch.cat([vel_rel_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
                vel_rel_body = quat_apply_inverse(quat, vel_rel_3d)[:, :2]

                other_dists.append(torch.cat([rel_body_xy, d, vel_rel_body], dim=1))

            obs_i = torch.cat([goal_body, goal_dist, vel_body, omega_z, shelf_dist] + other_dists, dim=1)

            # 배터리 + 가장 가까운 충전소 방향(body frame) — enable_battery 시 +3D
            if self.cfg.enable_battery:
                chargers = torch.tensor(
                    CHARGER_POSITIONS[:self.cfg.n_chargers],
                    device=self.device, dtype=torch.float32,
                )  # (C, 2)
                local_pos = pos_w - self.scene.env_origins[:, :2]   # (N, 2)
                dists = (local_pos.unsqueeze(1) - chargers.unsqueeze(0)).norm(dim=2)  # (N, C)
                nidx = dists.argmin(dim=1)                          # (N,)
                charger_w = self.scene.env_origins[:, :2] + chargers[nidx]  # world (N,2)
                cvec_3d = torch.cat([
                    charger_w - pos_w,
                    torch.zeros(self.num_envs, 1, device=self.device),
                ], dim=1)
                charger_body = quat_apply_inverse(quat, cvec_3d)[:, :2]
                obs_i = torch.cat([obs_i, self._battery[:, i:i+1], charger_body], dim=1)

            obs_list.append(obs_i)

        obs = torch.cat(obs_list, dim=1)   # (N, N_ROBOTS * OBS_PER_ROBOT) = (N, 51)
        # "policy": per-robot 분리 → wrapper가 (E*N, 17) 로 split
        # "critic": 그대로 51-dim global state → centralized critic 입력
        return {"policy": obs, "critic": obs}

    # ------------------------------------------------------------------
    # Rewards: 진짜 CTDE — per-robot 개별 보상 (credit assignment)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        from training.multi_robot.potential_reward import all_robots_mpg_reward

        positions = torch.stack([r.data.root_pos_w[:, :2] for r in self.robots], dim=1)

        # 첫 스텝 또는 reset 직후에는 이전 위치가 현재와 동일 → delta = 0
        if self._prev_positions is None:
            self._prev_positions = positions.clone()

        # MPG: 로봇별 goal-approaching + delta repulsion 보상 (N, N_ROBOTS)
        per_robot = all_robots_mpg_reward(
            positions, self._goal_pos_w,
            prev_positions=self._prev_positions,
            alpha=self.cfg.alpha, beta=self.cfg.beta,
            goal_radius=self.cfg.goal_radius,
            time_step=self.episode_length_buf,
            rew_goal=self.cfg.rew_goal,
        )

        self._prev_positions = positions.clone()

        # per-robot 충돌 패널티 — 충돌한 로봇 양쪽 각자 부담
        for i in range(N_ROBOTS):
            for j in range(i + 1, N_ROBOTS):
                d = (positions[:, i] - positions[:, j]).norm(dim=1)
                col = (d < ROBOT_COLLISION_DIST).float() * self.cfg.rew_collision
                per_robot[:, i] += col
                per_robot[:, j] += col

        # 동적 장애물 충돌 패널티 — 등장한 장애물에 0.1m 이내 접근 시
        if self.cfg.enable_dynamic_obstacles:
            for i, robot in enumerate(self.robots):
                local_pos = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
                obst_d = self._dynamic_obstacle_dist(local_pos)   # (N,) AABB 거리
                hit = (obst_d < 0.1).float() * self.cfg.rew_obstacle_collision
                per_robot[:, i] += hit

        # 배터리: 저전력(urgency>0)일 때만 작동, battery≥thresh면 0 → goal 추구 방해 안 함
        #   ① 충전 중(충전소 위) 양의 보상 = 가치 앵커  ② 접근 progress(potential)  ③ 방전 페널티
        if self.cfg.enable_battery:
            origins2 = self.scene.env_origins[:, :2]
            chargers = torch.tensor(
                CHARGER_POSITIONS[:self.cfg.n_chargers],
                device=self.device, dtype=torch.float32,
            )
            for i, robot in enumerate(self.robots):
                bat = self._battery[:, i]
                urgency = ((self.cfg.battery_low_thresh - bat)
                           / self.cfg.battery_low_thresh).clamp(min=0.0)   # 0~1
                local = robot.data.root_pos_w[:, :2] - origins2
                cdist = (local.unsqueeze(1) - chargers.unsqueeze(0)).norm(dim=2).min(dim=1).values
                # ① 충전 중 보상 (충전소 위 + 저전력) — 다 차면 urgency→0이라 자동 종료(죽치기 방지)
                on_charger = (cdist < self.cfg.charge_radius).float()
                per_robot[:, i] += on_charger * urgency * self.cfg.rew_charging
                # ② potential-based 접근 progress (가까워진 만큼 +, goal 이동 방해 안 함)
                progress = (self._prev_charger_dist[:, i] - cdist).clamp(-0.5, 0.5)
                per_robot[:, i] += urgency * progress * self.cfg.rew_charger_progress
                self._prev_charger_dist[:, i] = cdist
                # ③ 방전 페널티 (완화)
                per_robot[:, i] += (bat <= 1e-3).float() * self.cfg.rew_depleted

        # per-robot 정지 패널티 — 목표 미도달 시에만 (이미 도달/충전 중인 로봇 제외)
        # 충전소 위 머물기는 면제: 안 그러면 "충전하려 멈추면 벌받는" 모순 → 충전 학습 방해
        _chargers = None
        if self.cfg.enable_battery:
            _chargers = torch.tensor(CHARGER_POSITIONS[:self.cfg.n_chargers],
                                     device=self.device, dtype=torch.float32)
        for i, robot in enumerate(self.robots):
            speed = robot.data.root_lin_vel_w[:, :2].norm(dim=1)
            dist_i = (robot.data.root_pos_w[:, :2] - self._goal_pos_w[:, i]).norm(dim=1)
            not_at_goal = (dist_i > self.cfg.goal_radius).float()
            if self.cfg.enable_battery:
                local = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
                cdist = (local.unsqueeze(1) - _chargers.unsqueeze(0)).norm(dim=2).min(dim=1).values
                not_at_goal = not_at_goal * (cdist >= self.cfg.charge_radius).float()  # 충전소 위면 면제
            per_robot[:, i] += (speed < 0.1).float() * not_at_goal * self.cfg.rew_stationary

        # 배터리 진단 로그 — 충전 행동이 일어나는지 측정
        # mean_battery 유지/상승 = 충전함(학습됨), 하락+depleted↑ = 충전 안 함(탐색/보상 실패)
        if self.cfg.enable_battery:
            log = self.extras.setdefault("log", {})
            log["mean_battery"]  = self._battery.mean().item()
            log["depleted_pct"]  = (self._battery <= 1e-3).float().mean().item() * 100.0
            log["min_battery"]   = self._battery.min().item()

        # wrapper가 per-robot 보상을 각 actor에 분배
        self.extras["per_robot_rewards"] = per_robot  # (N, N_ROBOTS)

        return per_robot.mean(dim=1)  # (N,) — env 인터페이스 유지

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

        # reset된 env의 prev_positions를 현재 위치로 동기화 (delta = 0 보장)
        if self._prev_positions is not None:
            for i, robot in enumerate(self.robots):
                self._prev_positions[env_ids_t, i] = robot.data.root_pos_w[env_ids_t, :2]

        # 동적 장애물: 등장 위치(env 중앙 영역)·시점 랜덤화, 비활성으로 초기화
        if self.cfg.enable_dynamic_obstacles:
            no = self.cfg.n_dynamic_obstacles
            self._obst_pos_local[env_ids_t] = sample_uniform(
                -3.0, 3.0, (n, no, 2), device=self.device
            )
            self._obst_spawn_step[env_ids_t] = sample_uniform(
                float(self.cfg.obstacle_spawn_step_min),
                float(self.cfg.obstacle_spawn_step_max),
                (n, no), device=self.device,
            )
            self._obst_active[env_ids_t] = False

        # 배터리 초기화 + 충전소 거리 progress 버퍼 동기화 (delta=0 보장)
        if self.cfg.enable_battery:
            self._battery[env_ids_t] = sample_uniform(
                self.cfg.battery_init_min, self.cfg.battery_init_max,
                (n, N_ROBOTS), device=self.device,
            )
            chargers = torch.tensor(
                CHARGER_POSITIONS[:self.cfg.n_chargers],
                device=self.device, dtype=torch.float32,
            )
            origins_r = self.scene.env_origins[env_ids_t, :2]
            for i, robot in enumerate(self.robots):
                local = robot.data.root_pos_w[env_ids_t, :2] - origins_r
                cd = (local.unsqueeze(1) - chargers.unsqueeze(0)).norm(dim=2).min(dim=1).values
                self._prev_charger_dist[env_ids_t, i] = cd
