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
from isaaclab.utils.math import euler_xyz_from_quat, sample_uniform
# quat_apply_inverse는 신규 Isaac Lab(RunPod) 이름. 구버전(Paperspace v2.0.0)은 quat_rotate_inverse(동일 기능)로 폴백
try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

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
OBS_PER_ROBOT_EXT_OBS = OBS_PER_ROBOT + 2 # 19: + 가장 가까운 장애물 body-frame xy(2) — 회피 방향 학습용


def obs_per_robot(enable_battery: bool, extended_obstacle_obs: bool = False) -> int:
    if enable_battery:
        return OBS_PER_ROBOT_BATTERY + (2 if extended_obstacle_obs else 0)
    return OBS_PER_ROBOT_EXT_OBS if extended_obstacle_obs else OBS_PER_ROBOT

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
    # diff-drive(레벨1): True면 vy(옆걸음) 차단 → 방향 틀려면 돌아야 함(실제 AMR처럼 커브).
    #   obs/액션 차원 불변 → model_10998 그대로 로드해 fine-tune 가능. 기본 False=기존 동작 보존.
    disable_strafe: bool = False
    # diff-drive 커리큘럼: max_vy를 1.0→0으로 '성공률 게이트' 적응형 감소(하드컷 절벽 회피).
    #   strafe_curriculum=True면 _cur_max_vy가 cfg.max_vy에서 시작해, 최근 윈도우 성공률이
    #   thresh 이상이면 step만큼 내림. 붕괴(collapse_thresh 미만) 시 한 단계 되돌림(revert).
    strafe_curriculum: bool = False
    curriculum_success_thresh: float = 0.8   # ≥이면 max_vy 한 단계 내림(locomotion 커리큘럼 표준 0.8~0.9)
    curriculum_collapse_thresh: float = 0.5  # <이면 한 단계 되돌림(revert) — value loss 폭발 보험
    curriculum_window: int = 200             # 성공률 측정 윈도우(완료 에피소드 수)
    curriculum_step: float = 0.1             # max_vy 감소 폭(0 근처는 더 작게 권장 — 문헌)
    rew_heading: float = 0.0                 # heading→goal 정렬 보상(목표 향해 '돌아서 가기' 학습). 0=무효
    # diff-drive 컨트롤러(계층/운전기사): 정책 desired-velocity를 [v, omega]로 변환해 물리에 적용.
    #   재학습 불필요(model_10998 그대로 재생). 옆걸음 의도 → 그쪽으로 돌아서 전진(실제 AMR 커브).
    #   물리 자체가 nonholonomic → 외형은 실제 pose 추종(어긋남/폭주 없음). 데모/eval 재생용.
    diff_drive_controller: bool = False
    diff_drive_turn_gain: float = 3.0        # 방향오차→omega 게인(클수록 빨리 돌아 향함)

    # 시각만 diff-drive (yaw 자동 정렬) — 회피 능력 손상 0, 외형만 진행방향 바라봄.
    #   vx/vy는 정책 그대로(strafe 보존, S5/S6 유지) → 물리 큐브가 진짜로 옆이동
    #   omega만 자동으로 속도 방향 정렬 → 몸이 가는 방향 바라봄 = diff-drive 시각
    #   Amazon Kiva 방식. (D) 컨트롤러와 차이: vy 안 죽임, omega만 정렬
    visual_yaw_align: bool = False
    yaw_align_gain: float = 4.0              # yaw 오차→omega 게인(클수록 빨리 정렬)
    yaw_align_min_speed: float = 0.15        # 이 속도 미만이면 yaw 안 돌림(정지 시 빙빙 방지)

    # 확장 obs: 가장 가까운 장애물 body-frame xy(2D) 추가 — diff-drive 회피 학습용.
    #   17D(기본)에서 19D로 확장. 정책이 "왼쪽 장애물 → 오른쪽 turn" 식의 spatial 판단 가능해짐.
    #   model_10998(17D) 호환 깨짐 → from-scratch 재학습 필요. (False=기존 호환 보존)
    extended_obstacle_obs: bool = False
    # 확장 보상: diff-drive에서 strafe 없이 omega로 회피 학습시키는 명시 신호.
    #   기본 0 = 비활성. extended_obstacle_obs=True일 때만 작동(필요한 obs 채널 보장).
    rew_evasive_turn: float = 0.0            # 가까운 장애물 반대방향 omega에 +
                                              # 권장 0.1 (충돌 -200 대비 회피 시도 sparse 신호 보강)
    rew_clearance: float    = 0.0            # 정면 콘에 장애물 없을 때 +/step
                                              # 권장 0.05 (회피 성공 결과에 매 step 양의 신호)
    # 정적 선반 근접 페널티 — 표면 가까이 갈수록 거리 비례 벌점(선반 회피의 직접 신호).
    #   기존엔 선반에 박아도 벌점 0(교착 시간낭비뿐)이라 회피를 안 배웠음. 기본 0=비활성.
    rew_shelf_prox: float = 0.0              # 권장 -2.0 (표면 접촉 시 -2/step, 거리 비례 감쇠)
    shelf_prox_dist: float = 0.6             # 이 거리(m) 이내부터 페널티 시작

    goal_radius: float = 0.35
    goal_range: float = 4.0
    goal_min_dist: float = 1.0
    # 목표를 선반에서 얼마나 떼어 샘플할지(작을수록 선반 근처/뒤에 생겨 '돌아가기'를 강제).
    #   기본 0.8=기존(트인 곳만). 회피 학습엔 0.15 권장(선반 옆/뒤 목표 → 우회 필수).
    goal_shelf_margin: float = 0.8

    # ── 시나리오형 학습 (docs §6 정공법): eval이 시험하는 상황을 학습 분포에 직접 주입 ──
    #   랜덤 목표만으론 정면교행·3way·선반관통을 안 겪어 영영 못 배운다는 진단의 해법.
    #   reset마다 env별로 시나리오 종류를 뽑아 구조화 스폰/목표를 부여(좌표는 랜덤화→일반화).
    scenario_train: bool = False
    # 혼합 가중치 [D 랜덤, A 정면교행, B 3way삼각, C 선반우회]. 합은 내부에서 정규화.
    #   D를 섞어 model_10998 일반 내비 능력 보존(warm-start). C가 선반우회(19D 방향obs 필요).
    scenario_weights: tuple = (0.30, 0.25, 0.20, 0.25)
    # True면 시나리오 스폰 yaw를 목표 방향±지터로(교행 정렬 phase 단축). 기본 False=랜덤 yaw(학습 정합).
    scenario_face_goal: bool = False
    # 데모 연출: 도달 즉시 새 골 부여(끊임없이 일하게) + all_reached 종료 안 함. 학습은 False 유지.
    continuous_goals: bool = False
    # 데모 연출: 선반↔스테이션 운반 사이클(데모 env가 _goal_pos_w를 상태기계로 구동). 학습은 False.
    task_cycle: bool = False

    # MPG 가중치 (SAFE_DIST 마스킹과 조합)
    # alpha↑: goal tracking 강화 / beta: SAFE_DIST 내 반발력만 작동
    # model_9999.pt는 beta=0.5로 훈련됨 — 재훈련 시 1.5 사용
    alpha: float = 1.0
    beta: float = 1.5

    rew_collision: float  = -200.0  # 충돌 = 즉사 (교착 -90과 확실한 차이)
    rew_goal: float       =    6.0  # 목표 도달 보상
    rew_stationary: float =   -0.6  # S6 장애물회피 98% 달성값. -0.45 절충은 S6/S1 박살(에바)로 폐기.
                                     # S3/S4 좁은통로·동일목표 교착은 orchestrator 영역(레이어 분리)
    rew_spin: float       =    0.0  # |omega| 페널티 계수(diff-drive에서 제자리 휙휙 회전 억제). 기본 0=무효.
                                     # diff_drive fine-tune 시 -0.05~-0.15 권장(과하면 회피 위해 못 돎)

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

        # diff-drive 커리큘럼 상태 — env가 스스로 성공률 측정해 max_vy 적응 감소
        self._cur_max_vy = float(self.cfg.max_vy)   # 현재 유효 strafe 상한(1.0에서 시작)
        self._curr_ep_count = 0                     # 윈도우 내 완료 에피소드 수
        self._curr_success_count = 0.0              # 윈도우 내 로봇별 도달비율 누적(분수)
        # 에피소드 중 각 로봇이 목표 도달했는지(도달률 게이트용). reset마다 초기화.
        self._reached_ever = torch.zeros(self.num_envs, N_ROBOTS, dtype=torch.bool, device=self.device)

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

    def _nearest_obstacle_body(self, pos_w: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """가장 가까운 장애물(선반 중심 + 동적장애물)의 body-frame xy + 거리.
        반환: (body_xy (N, 2), dist (N,))
        선반은 중심점, 동적장애물은 등장한 것만. 비활성 시 매우 멀리(1e6) 처리.
        """
        n = pos_w.shape[0]
        env_origins_xy = self.scene.env_origins[:, :2]  # (N, 2)

        # 선반은 3m 긴 박스 → '중심'이 아니라 'AABB 가장 가까운 표면점'을 써야 회피에 유효.
        #   중심을 쓰면 선반 끝에 있는 로봇이 엉뚱한 옆방향을 받아 벽 위치를 모름(회피 학습 불가).
        shelf_c = torch.tensor(
            [[c[0], c[1]] for c in SHELF_CENTERS],
            device=self.device, dtype=torch.float32,
        )  # (S, 2)
        shelf_world_c = env_origins_xy.unsqueeze(1) + shelf_c.unsqueeze(0)   # (N, S, 2) 중심(월드)
        _half = torch.tensor([SHELF_HALF[0], SHELF_HALF[1]],
                             device=self.device, dtype=torch.float32)         # (2,)
        lo = shelf_world_c - _half
        hi = shelf_world_c + _half
        # 로봇 위치를 박스에 clamp → 박스 표면(밖이면)·로봇점(안이면) = 가장 가까운 점
        nearest_pt = torch.minimum(torch.maximum(pos_w.unsqueeze(1), lo), hi)  # (N, S, 2)
        rel_shelf  = nearest_pt - pos_w.unsqueeze(1)                           # (N, S, 2)
        dist_shelf = rel_shelf.norm(dim=2)                                     # (N, S)

        # 동적 장애물 (등장한 것만)
        if self.cfg.enable_dynamic_obstacles:
            obst_world = env_origins_xy.unsqueeze(1) + self._obst_pos_local  # (N, O, 2)
            rel_obst = obst_world - pos_w.unsqueeze(1)                       # (N, O, 2)
            dist_obst = rel_obst.norm(dim=2)                                 # (N, O)
            dist_obst = torch.where(self._obst_active, dist_obst,
                                    torch.full_like(dist_obst, 1e6))
            all_rel = torch.cat([rel_shelf, rel_obst], dim=1)
            all_dist = torch.cat([dist_shelf, dist_obst], dim=1)
        else:
            all_rel = rel_shelf
            all_dist = dist_shelf

        nearest_idx = all_dist.argmin(dim=1)                                  # (N,)
        batch_idx = torch.arange(n, device=self.device)
        nearest_rel_w = all_rel[batch_idx, nearest_idx]                       # (N, 2)
        nearest_dist  = all_dist[batch_idx, nearest_idx]                      # (N,)

        # body frame 회전
        rel_3d = torch.cat([nearest_rel_w, torch.zeros(n, 1, device=self.device)], dim=1)
        body_xy = quat_apply_inverse(quat, rel_3d)[:, :2]                     # (N, 2)
        return body_xy, nearest_dist

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

            # 유효 strafe 상한: 커리큘럼(적응형) > 하드차단 > 기본. 셋 다 obs/act 차원 불변.
            if self.cfg.strafe_curriculum:
                eff_max_vy = self._cur_max_vy        # 성공률 게이트로 1.0→0 점진 감소
            elif self.cfg.disable_strafe:
                eff_max_vy = 0.0                     # 하드 차단(절벽 — 실패 확인됨)
            else:
                eff_max_vy = self.cfg.max_vy         # 기존 홀로노믹
            vx_b = self._actions[:, i, 0] * self.cfg.max_vx
            vy_b = self._actions[:, i, 1] * eff_max_vy
            omega = self._actions[:, i, 2] * self.cfg.max_omega

            if self.cfg.diff_drive_controller:
                # ── 계층 컨트롤러: 정책 desired-velocity(vx_b,vy_b)를 diff-drive [v,omega]로 변환 ──
                #   strafe 상한 무시하고 정책의 full velocity 의도를 '조향신호'로 사용(eff_max_vy 미적용).
                vx_des = self._actions[:, i, 0] * self.cfg.max_vx
                vy_des = self._actions[:, i, 1] * self.cfg.max_vy   # 옆걸음 의도 = 그쪽으로 '돌아라'
                heading_err = torch.atan2(vy_des, vx_des)           # 현재 헤딩 기준 목표 속도 방향
                desired_speed = torch.sqrt(vx_des * vx_des + vy_des * vy_des)
                omega = torch.clamp(self.cfg.diff_drive_turn_gain * heading_err,
                                    -self.cfg.max_omega, self.cfg.max_omega)
                # 정렬된 만큼만 전진(향하는 중엔 감속, 다 돌면 풀속도) → 옆걸음 0
                vx_b = desired_speed * torch.relu(torch.cos(heading_err))
                vy_b = torch.zeros_like(vx_b)

            # NOTE: visual_yaw_align cfg는 demo_play.py의 _apply_action override에서 처리.
            # env 레벨에서 omega를 건드리면 body frame이 바뀌어 정책 action 해석이 흔들리는
            # control feedback loop가 발생 — 떨림 발생. 시각만 손대고 물리는 건드리지 않음.

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

            # 확장 obs: 가장 가까운 장애물 body-frame xy (회피 방향 학습 신호)
            #   왼쪽(y>0) ⇒ 오른쪽으로 turn, 오른쪽(y<0) ⇒ 왼쪽으로 turn — spatial 판단 가능
            if self.cfg.extended_obstacle_obs:
                obs_body_xy, _ = self._nearest_obstacle_body(pos_w, quat)
                obs_i = torch.cat([obs_i, obs_body_xy.clamp(-5.0, 5.0)], dim=1)

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
            # 정적 선반 근접 페널티 — 표면 < shelf_prox_dist 안으로 갈수록 거리 비례 벌점.
            #   ramming/hugging을 직접 처벌 → "선반 돌아가기" 학습. rew_shelf_prox=0이면 무효.
            if self.cfg.rew_shelf_prox != 0.0:
                _local_i = robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
                _sd = _shelf_aabb_dist(_local_i)
                _prox = ((self.cfg.shelf_prox_dist - _sd) / self.cfg.shelf_prox_dist).clamp(min=0.0)
                per_robot[:, i] += self.cfg.rew_shelf_prox * _prox
            # diff-drive: 제자리 휙휙 회전 억제(|omega| 비례 페널티). rew_spin=0이면 무효.
            if self.cfg.rew_spin != 0.0:
                per_robot[:, i] += self.cfg.rew_spin * self._actions[:, i, 2].abs()
            # diff-drive: heading→goal 보상 — '속도투영형'(순수 cos는 멈춤 리워드해킹 함정).
            #   r = v_forward × cos(bearing): 목표 보고 '굴러갈 때만' +. 멈춤/스핀/strafe=0 → 해킹 방지.
            #   목표 미도달 시에만(도착 후 미세정렬 방해 안 함). rew_heading=0이면 무효.
            if self.cfg.rew_heading != 0.0:
                _, _, yaw_i = euler_xyz_from_quat(robot.data.root_quat_w)
                hx, hy = torch.cos(yaw_i), torch.sin(yaw_i)              # heading 단위벡터
                gvec = self._goal_pos_w[:, i] - robot.data.root_pos_w[:, :2]
                gnorm = gvec.norm(dim=1).clamp(min=1e-6)
                cos_bearing = (hx * gvec[:, 0] + hy * gvec[:, 1]) / gnorm   # heading·목표방향
                v = robot.data.root_lin_vel_w[:, :2]
                v_forward = v[:, 0] * hx + v[:, 1] * hy                  # 전진 속도(헤딩 성분)
                per_robot[:, i] += self.cfg.rew_heading * v_forward * cos_bearing * not_at_goal

            # 확장 회피 보상 — extended_obstacle_obs True이고 가중치 > 0일 때만 작동.
            #   ① rew_evasive_turn: 가까운(< 1.5m) 장애물 있을 때, 그 반대 방향으로 omega하면 +
            #     장애물이 body y>0(왼쪽)이면 omega<0(오른쪽 회전)에 +, y<0이면 omega>0에 +.
            #     수식: signal = -omega * sign(obs_y) * proximity. PPO sparse 충돌(-200) 보강.
            #   ② rew_clearance: 정면 45° 콘에 가까운 장애물 없을 때 +/step (회피 결과 양의 신호)
            if self.cfg.extended_obstacle_obs and (
                self.cfg.rew_evasive_turn != 0.0 or self.cfg.rew_clearance != 0.0
            ):
                obs_body_xy, obs_dist = self._nearest_obstacle_body(
                    robot.data.root_pos_w[:, :2], robot.data.root_quat_w
                )
                if self.cfg.rew_evasive_turn != 0.0:
                    proximity = ((1.5 - obs_dist) / 1.5).clamp(min=0.0, max=1.0)   # 0 멀리~1 매우가깝
                    omega_i = self._actions[:, i, 2] * self.cfg.max_omega
                    # bearing y 부호: >0 왼쪽 / <0 오른쪽. 반대로 돌면 +
                    evasive = -omega_i * obs_body_xy[:, 1].sign() * proximity
                    per_robot[:, i] += self.cfg.rew_evasive_turn * evasive
                if self.cfg.rew_clearance != 0.0:
                    # 정면 45° 콘(|y| < x AND x > 0) 안에 1.5m 이내 장애물 있으면 막힘
                    in_front_cone = (obs_body_xy[:, 0] > 0) & (obs_body_xy[:, 1].abs() < obs_body_xy[:, 0])
                    blocked = in_front_cone & (obs_dist < 1.5)
                    clear = (~blocked).float()
                    per_robot[:, i] += self.cfg.rew_clearance * clear * not_at_goal

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
            reached_i = dist < self.cfg.goal_radius
            all_reached &= reached_i
            self._reached_ever[:, i] |= reached_i   # 에피소드 중 한 번이라도 도달하면 True
            local = pos_w - self.scene.env_origins[:, :2]
            any_oob |= local.abs().max(dim=1).values > 8.0

        # 로봇 간 충돌 감지
        for i in range(N_ROBOTS):
            for j in range(i + 1, N_ROBOTS):
                d = (positions[i] - positions[j]).norm(dim=1)
                collision |= (d < ROBOT_COLLISION_DIST)

        # ── 데모 연속 골: 도달한 로봇에 즉시 새(랜덤) 골 부여 → 끊임없이 일함 ──
        if self.cfg.continuous_goals:
            for i in range(N_ROBOTS):
                reached_i = (positions[i] - self._goal_pos_w[:, i]).norm(dim=1) < self.cfg.goal_radius
                idx = reached_i.nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    self._goal_pos_w[idx, i] = self._sample_goal(idx)
        # 연속 골/운반 사이클 모두 도달로는 에피소드 안 끝냄(충돌/이탈만 리셋).
        # 운반 사이클의 골 갱신은 데모 env._get_dones 오버라이드(_advance_task)가 담당.
        if self.cfg.continuous_goals or self.cfg.task_cycle:
            all_reached = torch.zeros_like(all_reached)

        terminated = all_reached | any_oob | collision
        timed_out  = self.episode_length_buf >= self.max_episode_length - 1

        # ── diff-drive 커리큘럼: env가 스스로 성공률 측정 → max_vy 적응 감소 ──
        #   하드컷(절벽) 대신, 정책이 현재 strafe 상한에서 '잘 하고 있을 때만' 한 단계 더 조임.
        if self.cfg.strafe_curriculum:
            done = terminated | timed_out
            if done.any():
                # 성공 = 로봇별 도달률(에피소드 중 목표 도달한 로봇 비율) — '3대 동시'보다 견고, "98% 도달"과 동일 의미
                frac = self._reached_ever[done].float().mean(dim=1)     # 완료 에피소드별 도달비율
                self._curr_success_count += float(frac.sum().item())
                self._curr_ep_count     += int(done.sum().item())
            if self._curr_ep_count >= self.cfg.curriculum_window:
                rate = self._curr_success_count / max(self._curr_ep_count, 1)
                cmax = float(self.cfg.max_vy)
                if rate >= self.cfg.curriculum_success_thresh and self._cur_max_vy > 0.0:
                    self._cur_max_vy = max(0.0, self._cur_max_vy - self.cfg.curriculum_step)
                    print(f"[Curriculum] 성공률 {rate:.2f} ≥ {self.cfg.curriculum_success_thresh} "
                          f"→ max_vy ↓ {self._cur_max_vy:.2f}")
                elif (rate < self.cfg.curriculum_collapse_thresh and self._cur_max_vy < cmax):
                    self._cur_max_vy = min(cmax, self._cur_max_vy + self.cfg.curriculum_step)
                    print(f"[Curriculum] 성공률 {rate:.2f} 붕괴 → max_vy ↑ {self._cur_max_vy:.2f} (revert)")
                self._curr_ep_count = 0
                self._curr_success_count = 0.0
                log = self.extras.setdefault("log", {})
                log["curriculum_reach_rate"] = rate    # 직전 윈도우 로봇별 도달률
            log = self.extras.setdefault("log", {})
            log["curriculum_max_vy"] = self._cur_max_vy

        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _sample_goal(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        """주어진 env들에 대해 선반과 안 겹치는 목표 1개씩 샘플(rejection). (n,2) world 좌표."""
        n = env_ids_t.shape[0]
        origins = self.scene.env_origins[env_ids_t, :2]
        angle = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
        dist  = sample_uniform(self.cfg.goal_min_dist, self.cfg.goal_range, (n,), device=self.device)
        candidates = origins + torch.stack([dist * torch.cos(angle), dist * torch.sin(angle)], dim=1)
        _m = self.cfg.goal_shelf_margin   # 작을수록 선반 근처/뒤 목표 허용 → 우회 강제(회피 학습)
        bad = _goal_in_shelf(candidates - origins, margin=_m)
        for _ in range(20):
            if not bad.any():
                break
            rem = bad.nonzero(as_tuple=True)[0]
            n_rem = rem.shape[0]
            a2 = sample_uniform(-math.pi, math.pi, (n_rem,), device=self.device)
            d2 = sample_uniform(self.cfg.goal_min_dist, self.cfg.goal_range, (n_rem,), device=self.device)
            cand2 = origins[rem] + torch.stack([d2 * torch.cos(a2), d2 * torch.sin(a2)], dim=1)
            b2 = _goal_in_shelf(cand2 - origins[rem], margin=_m)
            candidates[rem[~b2]] = cand2[~b2]
            bad[rem[~b2]] = False
        return candidates

    def _sample_scenario_layout(
        self, env_ids_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """시나리오형 학습용 구조화 스폰/목표 생성 (env-local 좌표).

        env마다 종류를 뽑아(D/A/B/C) 패밀리별 파라메트릭 레이아웃을 채운다. 좌표는
        매 reset 랜덤화 → eval 인스턴스가 학습 분포의 한 샘플이 되게(암기 아닌 일반화).

        반환:
          spawn_local (n, N_ROBOTS, 2) — 전 env(랜덤 envs는 기본 오프셋+노이즈)
          goal_local  (n, N_ROBOTS, 2) — 시나리오 envs만 유효(랜덤 envs는 placeholder)
          rand_mask   (n,) bool        — True면 _reset_idx가 _sample_goal로 목표 덮어씀
        선반 배치: 중심 (±2,±2.5), x∈[-3.5,-0.5]/[0.5,3.5], y∈[±2.25,±2.75].
          중앙 수평통로 |y|<2.25 개방, x≈0 세로틈(폭1m)으로 남북 통과.
        """
        n = env_ids_t.shape[0]
        d = self.device
        R = N_ROBOTS

        def U(lo, hi, *shape):
            return sample_uniform(lo, hi, tuple(shape), device=d)

        def push_clear(pts: torch.Tensor, margin: float = 0.35) -> torch.Tensor:
            """선반(+margin) 안에 든 점을 y축으로 '가장 가까운 통로 면' 밖으로 이동.
            선반은 y로 얇아(half 0.25) y만 살짝 밀면 도달 가능·트인 곳이 됨. 원래 side 보존."""
            out = pts.clone()
            for cx, cy, *_ in SHELF_CENTERS:
                in_x = (out[..., 0] - cx).abs() < SHELF_HALF[0] + margin
                in_y = (out[..., 1] - cy).abs() < SHELF_HALF[1] + margin
                hit = in_x & in_y
                below = cy - (SHELF_HALF[1] + margin)
                above = cy + (SHELF_HALF[1] + margin)
                pick_below = (out[..., 1] - below).abs() < (out[..., 1] - above).abs()
                new_y = torch.where(pick_below, torch.full_like(out[..., 1], below),
                                    torch.full_like(out[..., 1], above))
                out[..., 1] = torch.where(hit, new_y, out[..., 1])
            return out

        spawn = torch.zeros(n, R, 2, device=d)
        goal  = torch.zeros(n, R, 2, device=d)

        w = torch.tensor(self.cfg.scenario_weights, device=d, dtype=torch.float32)
        sid = torch.multinomial(w / w.sum(), n, replacement=True)   # (n,) ∈ {0:D,1:A,2:B,3:C}
        rand_mask = (sid == 0)

        # ── D 랜덤(id 0): 기존 고정 오프셋+노이즈 스폰. 목표는 _reset_idx가 _sample_goal로 채움 ──
        off = torch.tensor(SPAWN_OFFSETS, device=d, dtype=torch.float32)   # (R, 2)
        spawn[:] = off.unsqueeze(0) + U(-0.3, 0.3, n, R, 2)

        # ── A 정면교행(id 1): 원점 통과 교행 2대 + 수직 횡단 1대(S1/S5형) ──
        a = (sid == 1)
        if a.any():
            na = int(a.sum())
            th = U(-math.pi, math.pi, na)
            ux, uy = torch.cos(th), torch.sin(th)
            u = torch.stack([ux, uy], dim=1) * U(2.5, 4.0, na).unsqueeze(1)   # 교행축
            v = torch.stack([-uy, ux], dim=1) * U(2.5, 3.5, na).unsqueeze(1)  # 수직축
            sp = torch.zeros(na, R, 2, device=d)
            gl = torch.zeros(na, R, 2, device=d)
            sp[:, 0] =  u; gl[:, 0] = -u    # 로봇0 ↔ 로봇1 정면교행(목표 교환)
            sp[:, 1] = -u; gl[:, 1] =  u
            sp[:, 2] =  v; gl[:, 2] = -v    # 로봇2 수직 횡단
            sp += U(-0.3, 0.3, na, R, 2)
            spawn[a] = push_clear(sp); goal[a] = push_clear(gl)   # 선반 박힌 끝점 통로로

        # ── B 3way 삼각(id 2): 삼각 정점 → 각자 반대 정점(중심 통과, S2형) ──
        b = (sid == 2)
        if b.any():
            nb = int(b.sum())
            r   = U(2.0, 3.5, nb)
            phi = U(-math.pi, math.pi, nb)
            sp = torch.zeros(nb, R, 2, device=d)
            for k in range(R):
                ang = phi + k * (2.0 * math.pi / R)
                sp[:, k, 0] = r * torch.cos(ang)
                sp[:, k, 1] = r * torch.sin(ang)
            sp += U(-0.25, 0.25, nb, R, 2)
            spawn[b] = push_clear(sp)
            goal[b]  = push_clear(-sp)      # 반대 정점 = 중심 통과 대치(선반 박힌 끝점 통로로)

        # ── C 선반우회(id 3): 중앙통로 출발 → 선반 너머 목표(직선경로가 선반 관통→우회 강제) ──
        #   x열을 셔플([-2,0,2])해 로봇 간 ≥1.6m x-분리 보장. 같은 부호=세로로 선반 한 줄 가로지름.
        #   x≈0 로봇은 세로틈 통과(트임), x±2 로봇은 우회 필수. 19D 방향obs가 출구를 알려줘야 학습됨.
        c = (sid == 3)
        if c.any():
            nc = int(c.sum())
            base = torch.tensor([-2.0, 0.0, 2.0], device=d)
            perm = torch.argsort(torch.rand(nc, R, device=d), dim=1)   # env별 열 셔플
            cols = base[perm]                                          # (nc, R)
            sgn = torch.where(U(0.0, 1.0, nc, R) > 0.5, 1.0, -1.0)     # 위/아래 면 선택
            sp = torch.zeros(nc, R, 2, device=d)
            gl = torch.zeros(nc, R, 2, device=d)
            xk = cols + U(-0.4, 0.4, nc, R)
            sp[..., 0] = xk
            sp[..., 1] = sgn * U(0.8, 1.5, nc, R)    # 중앙통로(|y|<2.25) 출발
            gl[..., 0] = xk
            gl[..., 1] = sgn * U(3.3, 4.2, nc, R)    # 선반 너머(|y|>2.75) 목표 — 경로만 선반 관통
            spawn[c] = sp; goal[c] = gl

        return spawn, goal, rand_mask

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robots[0]._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.long()
        n = env_ids_t.shape[0]

        # 커리큘럼 도달 플래그 초기화(새 에피소드 시작)
        if self.cfg.strafe_curriculum:
            self._reached_ever[env_ids_t] = False

        # 시나리오형 학습: env별 구조화 스폰/목표(교행·3way·선반우회). 비활성 시 기존 랜덤 경로.
        scenario = self.cfg.scenario_train
        if scenario:
            spawn_local, goal_local, rand_mask = self._sample_scenario_layout(env_ids_t)

        for i, robot in enumerate(self.robots):
            default_state = robot.data.default_root_state[env_ids_t].clone()
            default_state[:, :3] += self.scene.env_origins[env_ids_t]
            if scenario:
                # 레이아웃이 이미 지터 포함 → 추가 노이즈 없이 그대로 사용
                default_state[:, 0] += spawn_local[:, i, 0]
                default_state[:, 1] += spawn_local[:, i, 1]
            else:
                default_state[:, 0] += SPAWN_OFFSETS[i][0]
                default_state[:, 1] += SPAWN_OFFSETS[i][1]
                default_state[:, :2] += sample_uniform(-0.3, 0.3, (n, 2), device=self.device)

            if scenario and self.cfg.scenario_face_goal:
                gvec = goal_local[:, i] - spawn_local[:, i]
                yaw = torch.atan2(gvec[:, 1], gvec[:, 0]) + sample_uniform(-0.4, 0.4, (n,), device=self.device)
            else:
                yaw = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
            zeros = torch.zeros(n, device=self.device)
            default_state[:, 3] = torch.cos(yaw / 2)
            default_state[:, 4] = zeros
            default_state[:, 5] = zeros
            default_state[:, 6] = torch.sin(yaw / 2)
            default_state[:, 7:] = 0.0
            robot.write_root_state_to_sim(default_state, env_ids_t)

        # 로봇별 목표 샘플링 (선반과 겹치지 않도록)
        if scenario:
            origins_xy = self.scene.env_origins[env_ids_t, :2]
            for i in range(N_ROBOTS):
                self._goal_pos_w[env_ids_t, i] = goal_local[:, i] + origins_xy
            # 랜덤(D) envs는 일반 내비 보존을 위해 _sample_goal로 목표 덮어쓰기
            if rand_mask.any():
                ridx = env_ids_t[rand_mask]
                for i in range(N_ROBOTS):
                    self._goal_pos_w[ridx, i] = self._sample_goal(ridx)
        else:
            for i in range(N_ROBOTS):
                self._goal_pos_w[env_ids_t, i] = self._sample_goal(env_ids_t)

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
