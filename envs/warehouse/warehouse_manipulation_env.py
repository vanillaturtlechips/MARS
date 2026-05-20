"""Phase 2 — 창고 Pick & Place 환경.

로봇: Franka Panda (7-DOF 암 + 평행 그리퍼)
임무: 박스를 집어 지정 선반 위치에 내려놓기

Teacher 관측 (특권 정보, 훈련 전용):
  box_pos(3) + box_quat(4) + box_mass(1) +
  ee_pos(3) + gripper_width(1) + goal_pos(3) +
  joint_pos(9) + joint_vel(9) = 33차원

Student 관측 (배포용, 실제 센서):
  ee_pos(3) + gripper_width(1) + goal_pos_approx(3) +
  noisy_box_rel(3, σ=0.03m, 카메라 감지 시뮬레이션) +
  joint_pos(9) + joint_vel(9) = 28차원

Teacher-Student 증류 순서:
  1. Teacher PPO 훈련 → place_success_rate > 90%
  2. Teacher 궤적 수집 (10만 에피소드)
  3. Student 모방학습
  4. Student 단독 평가 (unseen 박스 크기)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms

try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
except ImportError:
    # isaaclab_assets 패키지 경로가 다를 경우 대비
    from isaaclab_assets import FRANKA_PANDA_CFG  # type: ignore

# 목표 선반 위치 4곳 (world frame, 로봇 베이스 기준)
# box spawn: x=[0.45,0.55], y=[-0.15,0.15] (정면)
# goals:     y=±0.32-0.35 (측면) → 측방 운반 필수, trivial place 없음
# min box-goal dist: box(0.55,0.15)↔goal(0.50,0.32) = 0.177m > 0.12m ✓
PLACE_GOALS = [
    (0.48,  0.35, 0.53),
    (0.48, -0.35, 0.53),
    (0.50,  0.32, 0.53),
    (0.50, -0.32, 0.53),
]

TEACHER_OBS_DIM = 30   # box_rel(3)+quat(4)+mass(1)+gripper(1)+goal_rel(3)+jpos(9)+jvel(9)
STUDENT_OBS_DIM = 28   # ee_pos(3)+gripper(1)+goal_rel(3)+noisy_box_rel(3)+jpos(9)+jvel(9)


@configclass
class WarehouseManipulationEnvCfg(DirectRLEnvCfg):
    decimation = 2               # 120Hz sim / 2 = 60Hz policy
    episode_length_s = 15.0
    action_space = 4             # [dx, dy, dz, gripper] Cartesian delta control
    observation_space = TEACHER_OBS_DIM   # Teacher 모드 기본
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=3.0, replicate_physics=True
    )

    # 보상 가중치 — 하이브리드 설계
    # Approach: Exp 금속탐지기 (박스 찾기), decay=5.0으로 중거리 hover 억제
    # Transport: Progress Delta (성과급) — 월급 루팡 방지
    #   이동한 만큼만 보상, 가만히 서있으면 0, 뒤로 가면 음수
    #   V(hover@goal) = transport(이동분) + 0(정지) << V(place) = transport + 800
    rew_approach:  float =  0.5    # Exp decay=5.0 적용
    rew_grasp:     float = 30.0    # 단발 grasp 유인
    rew_transport: float = 10.0    # delta * 100 배율 → 3cm 이동 시 +30/step
    rew_place:     float = 800.0   # 대형 터미널 보상 유지
    rew_drop:      float =   0.0   # 낙하 패널티 제거 (박스 회피 전략 방지)
    rew_time:      float = -0.02   # 스텝 패널티 축소 (탐색 장려)

    # 박스 Domain Randomization
    box_size_range: tuple[float, float] = (0.04, 0.08)   # m (정육면체 한 변)
    box_mass_range: tuple[float, float] = (0.3, 2.0)     # kg

    # Franka max reach @z=0.53: x_max=0.671m → box x≤0.55 (89% reach, IK 안정)
    # EE x≈0.307 ~ box x=0.45: dist=0.155m < threshold → 일부 즉시 grasp (OK, transport 학습)
    # EE x≈0.307 ~ box x=0.55,y=0.15: dist=0.292m > threshold → approach 학습 필요
    grasp_dist_threshold: float = 0.25   # ee ~ box 거리 [m]
    place_dist_threshold: float = 0.12   # box ~ goal 거리 [m]

    camera_noise_min: float = 0.005      # 카메라 노이즈 DR 하한 [m]
    camera_noise_max: float = 0.02       # 카메라 노이즈 DR 상한 [m]

    student_mode: bool = False    # True면 Student 관측 반환


@configclass
class WarehouseManipulationStudentEnvCfg(WarehouseManipulationEnvCfg):
    """Student 훈련용 — 특권 정보 없음."""
    observation_space = STUDENT_OBS_DIM
    student_mode = True
    grasp_dist_threshold: float = 0.25   # Teacher와 동일 (grasp은 진짜 박스 위치 기준)
    camera_noise_min: float = 0.005      # 0.5cm
    camera_noise_max: float = 0.02       # 2cm (RealSense 수준)


class WarehouseManipulationEnv(DirectRLEnv):
    cfg: WarehouseManipulationEnvCfg

    def __init__(self, cfg: WarehouseManipulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        d = self.device

        body_names = list(self.robot.data.body_names)
        self._ee_body_idx  = body_names.index("panda_hand")
        self._jac_body_idx = self._ee_body_idx - 1  # get_jacobians: base 제외 offset

        self._goal_pos_w  = torch.zeros(n, 3, device=d)
        self._box_mass    = torch.ones(n, device=d)
        self._grasped     = torch.zeros(n, dtype=torch.bool, device=d)
        self._actions     = torch.zeros(n, 4, device=d)
        self._prev_dist_ee_box   = torch.full((n,), 999.0, device=d)
        self._prev_dist_box_goal = torch.full((n,), 999.0, device=d)
        self._frozen_box_state   = torch.zeros(n, 13, device=d)
        self._camera_noise_std   = torch.full((n,), 0.03, device=d)  # 에피소드당 카메라 노이즈 레벨
        # grasp 시 EE→박스 offset (박스가 EE를 따라 이동하도록)
        self._grasp_ee_offset    = torch.zeros(n, 3, device=d)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # Franka Panda
        franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        # 기본 stiffness=80은 너무 낮아 관절이 명령의 12%만 추종 → 400으로 상향
        for actuator in franka_cfg.actuators.values():
            if hasattr(actuator, 'stiffness') and actuator.stiffness < 400:
                actuator.stiffness = 400.0
                actuator.damping   = 40.0
        self.robot = Articulation(franka_cfg)

        # 박스 (크기는 reset에서 DR 적용)
        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.06, 0.06),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.6, 0.1), metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.03)),
        )
        self.box = RigidObject(box_cfg)

        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # 테이블: 상면 z=0.5m, 박스와 EE가 같은 높이에서 상호작용
        table_cfg = sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.15), metallic=0.0),
        )
        table_cfg.func("/World/envs/env_0/Table", table_cfg,
                       translation=(0.45, 0.0, 0.25), orientation=(1.0, 0.0, 0.0, 0.0))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["box"]   = self.box

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Actions: 관절 위치 목표 전달
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        n = self.num_envs
        # Cartesian delta: actions[:, :3] = [dx, dy, dz] world frame
        delta_pos = self._actions[:, :3] * 0.03  # max 3cm/step

        # DLS IK: Δq = J^T (J J^T + λI)^{-1} Δx  (λ=0.01: 원래 0.05보다 덜 보수적)
        jac = self.robot.root_physx_view.get_jacobians()
        J   = jac[:, self._jac_body_idx, :3, :7]   # [N, 3, 7]
        lam = 0.01
        JT      = J.transpose(-2, -1)
        JJT_reg = torch.bmm(J, JT) + lam * torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        J_dls   = torch.bmm(JT, torch.linalg.inv(JJT_reg))
        delta_q = torch.bmm(J_dls, delta_pos.unsqueeze(-1)).squeeze(-1)

        joint_target = self.robot.data.joint_pos[:, :7] + delta_q
        self.robot.set_joint_position_target(joint_target, joint_ids=list(range(7)))

        # Gripper: action[:, 3] ∈ [-1, 1] → [0, 0.04]m
        gripper_pos = ((self._actions[:, 3:4] + 1.0) / 2.0) * 0.04
        self.robot.set_joint_position_target(
            gripper_pos.expand(-1, 2).clone(), joint_ids=[7, 8]
        )

        # Proximity grasp: 박스를 EE + offset 위치로 이동 (EE를 따라 운반)
        if self._grasped.any():
            grasped_ids = self._grasped.nonzero(as_tuple=True)[0]
            ee_pos, _ = self._get_ee_pose()
            frozen = self._frozen_box_state[grasped_ids].clone()
            frozen[:, :3] = ee_pos[grasped_ids] + self._grasp_ee_offset[grasped_ids]
            frozen[:, 7:13] = 0.0
            self.box.write_root_state_to_sim(frozen, grasped_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        ee_pos, _  = self._get_ee_pose()
        joint_pos  = self.robot.data.joint_pos        # (N, 9)
        joint_vel  = self.robot.data.joint_vel        # (N, 9)
        gripper_w  = (joint_pos[:, 7:8] + joint_pos[:, 8:9])  # 그리퍼 폭

        if self.cfg.student_mode:
            # Student: box 위치는 카메라 감지 시뮬레이션 (Domain Randomization)
            # 노이즈 레벨은 에피소드 시작 시 [1cm, 6cm] 균일 샘플 — per-step 샘플 금지
            # (per-step이면 3cm 이동 vs 최대 6cm 노이즈 → SNR<1 → gradient 파괴)
            # 실제 배포 시 RGB-D 카메라 출력으로 대체
            box_pos = self.box.data.root_pos_w             # (N, 3)
            noise   = torch.randn_like(box_pos) * self._camera_noise_std.unsqueeze(1)
            box_rel_noisy = (box_pos - ee_pos) + noise
            ee_pos_local = ee_pos - self.robot.data.root_pos_w  # 로봇 베이스 기준 상대좌표
            obs = torch.cat([
                ee_pos_local,                    # (N, 3) EE 위치 (베이스 프레임)
                gripper_w,                       # (N, 1)
                self._goal_pos_w - ee_pos,       # (N, 3) goal 상대좌표
                box_rel_noisy,                   # (N, 3) 카메라 기반 box 위치 (노이즈 포함)
                joint_pos[:, :9],                # (N, 9)
                joint_vel[:, :9],                # (N, 9)
            ], dim=1)   # (N, 28)
        else:
            # Teacher: 특권 정보 포함 (box/goal 모두 EE 기준 상대좌표)
            box_pos  = self.box.data.root_pos_w            # (N, 3)
            box_quat = self.box.data.root_quat_w           # (N, 4)
            obs = torch.cat([
                box_pos - ee_pos,                # (N, 3) box 상대좌표 ← 핵심
                box_quat,                        # (N, 4)
                self._box_mass.unsqueeze(1),     # (N, 1)
                gripper_w,                       # (N, 1)
                self._goal_pos_w - ee_pos,       # (N, 3) goal 상대좌표
                joint_pos[:, :9],                # (N, 9)
                joint_vel[:, :9],                # (N, 9)
            ], dim=1)   # (N, 30)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        dist_ee_box  = (ee_pos - box_pos).norm(dim=1)

        dist_ee_goal = (ee_pos - self._goal_pos_w).norm(dim=1)

        # 파지 판정
        newly_grasped = (~self._grasped) & (dist_ee_box < self.cfg.grasp_dist_threshold)
        self._grasped |= newly_grasped
        # grasp 발동 순간: 박스 state 저장 + EE→박스 offset 계산
        if newly_grasped.any():
            new_ids = newly_grasped.nonzero(as_tuple=True)[0]
            self._frozen_box_state[new_ids] = self.box.data.root_state_w[new_ids].clone()
            self._grasp_ee_offset[new_ids]  = box_pos[new_ids] - ee_pos[new_ids]

        # 박스 현재 위치 (EE + offset으로 운반 중)
        box_pos_carried = ee_pos + self._grasp_ee_offset  # grasped 아닌 env는 의미없음
        dist_box_goal = (box_pos_carried - self._goal_pos_w).norm(dim=1)

        # 낙하 판정 (proximity grasp에서는 박스가 날아가지 않으므로 거의 발생 안 함)
        dropped = self._grasped & (box_pos[:, 2] < 0.30)

        # 거치 성공: 박스 위치가 goal에 도달 (EE가 박스를 실제로 운반해야 함)
        placed = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)

        not_grasped = (~self._grasped).float()

        # Approach: Exp 금속탐지기 — decay=5.0으로 근거리에만 집중
        approach = self.cfg.rew_approach * torch.exp(-dist_ee_box * 5.0) * not_grasped

        # Transport: Progress Delta — "어제보다 오늘 더 다가간 만큼만" 보상
        # clamp(-0.1, 0.1): 첫 스텝 prev=999 튀는 현상 방지, 최대 이동 제한
        delta_goal = (self._prev_dist_box_goal - dist_box_goal).clamp(-0.1, 0.1)
        transport  = self.cfg.rew_transport * delta_goal * 100.0 * self._grasped.float()

        self._prev_dist_ee_box   = dist_ee_box.detach()
        self._prev_dist_box_goal = dist_box_goal.detach()

        rew = (
            approach
            + transport
            + self.cfg.rew_grasp * newly_grasped.float()
            + self.cfg.rew_place * placed.float()
            + self.cfg.rew_drop  * dropped.float()
            + self.cfg.rew_time  # 매 스텝 고정 패널티
        )
        return rew

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        box_pos_carried = ee_pos + self._grasp_ee_offset
        dist_box_goal   = (box_pos_carried - self._goal_pos_w).norm(dim=1)
        placed  = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)
        dropped = self._grasped & (box_pos[:, 2] < 0.30)

        # dropped는 termination 조건에서 제외 — placed만 종료
        # (dropped penalty로 policy가 박스 회피 전략을 학습하는 것 방지)
        terminated = placed
        timed_out  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.long()
        n = env_ids_t.shape[0]

        # Franka "ready" 자세: ee가 테이블 앞 ~40cm 높이 (default 수직 자세 대비 박스에 훨씬 가까움)
        reach_pose = torch.tensor(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            device=self.device
        ).unsqueeze(0).expand(n, -1)
        self.robot.set_joint_position_target(reach_pose, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(reach_pose, torch.zeros_like(reach_pose), env_ids=env_ids_t)

        # 박스 위치 랜덤화 — 절대 좌표로 설정 (default_root_state.x=0.5 누적 버그 방지)
        box_state = self.box.data.default_root_state[env_ids_t].clone()
        box_state[:, 0] = self.scene.env_origins[env_ids_t, 0] + sample_uniform(0.45, 0.55, (n,), device=self.device)
        box_state[:, 1] = self.scene.env_origins[env_ids_t, 1] + sample_uniform(-0.15, 0.15, (n,), device=self.device)
        box_state[:, 2] = self.scene.env_origins[env_ids_t, 2] + 0.53  # 테이블 위 (상면 0.5m + 박스 반경 0.03m)
        self.box.write_root_state_to_sim(box_state, env_ids_t)

        # 박스 질량 DR
        self._box_mass[env_ids_t] = sample_uniform(
            self.cfg.box_mass_range[0], self.cfg.box_mass_range[1], (n,), device=self.device
        )

        # 목표 위치 선택 (4개 중 랜덤)
        goal_idx = torch.randint(len(PLACE_GOALS), (n,), device=self.device)
        goals    = torch.tensor(PLACE_GOALS, device=self.device)[goal_idx]
        self._goal_pos_w[env_ids_t] = goals + self.scene.env_origins[env_ids_t]

        # 상태 리셋
        self._grasped[env_ids_t] = False
        self._prev_dist_ee_box[env_ids_t]   = 999.0
        self._prev_dist_box_goal[env_ids_t] = 999.0
        self._frozen_box_state[env_ids_t]   = 0.0
        self._grasp_ee_offset[env_ids_t]    = 0.0
        # 카메라 노이즈 DR: 에피소드마다 [noise_min, noise_max] 균일 샘플
        noise_range = self.cfg.camera_noise_max - self.cfg.camera_noise_min
        self._camera_noise_std[env_ids_t] = (
            torch.rand(n, device=self.device) * noise_range + self.cfg.camera_noise_min
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """End-effector 위치와 방향 반환."""
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_pos, ee_quat
