"""Phase 2 — 창고 Pick & Place 환경.

로봇: Franka Panda (7-DOF 암 + 평행 그리퍼)
임무: 박스를 집어 지정 선반 위치에 내려놓기

Teacher 관측 (특권 정보, 훈련 전용):
  box_pos(3) + box_quat(4) + box_mass(1) +
  ee_pos(3) + gripper_width(1) + goal_pos(3) +
  joint_pos(9) + joint_vel(9) = 33차원

Student 관측 (배포용, 실제 센서):
  ee_pos(3) + gripper_width(1) + goal_pos_approx(3) +
  joint_pos(9) + joint_vel(9) = 25차원
  (RGB-D CNN feature는 추후 추가)

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
PLACE_GOALS = [
    (0.4,  0.2, 0.53),
    (0.4, -0.2, 0.53),
    (0.5,  0.1, 0.53),
    (0.5, -0.1, 0.53),
]

TEACHER_OBS_DIM = 30   # box_rel(3)+quat(4)+mass(1)+gripper(1)+goal_rel(3)+jpos(9)+jvel(9)
STUDENT_OBS_DIM = 25


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

    # 보상 가중치
    rew_approach:  float =  3.0    # exp(-dist*10): 0.5m이상에서 time_penalty보다 낮아 강제 접근
    rew_grasp:     float = 10.0    # 파지 성공
    rew_transport: float =  5.0    # potential-based delta 스케일 (time_penalty의 5배 이상)
    rew_place:     float = 20.0    # 거치 성공
    rew_drop:      float =   0.0   # 낙하 패널티 제거 (박스 회피 전략 방지)
    rew_time:      float = -0.1    # 스텝 패널티 → timeout(-90) vs place(+20) 차이 120점으로 확대

    # 박스 Domain Randomization
    box_size_range: tuple[float, float] = (0.04, 0.08)   # m (정육면체 한 변)
    box_mass_range: tuple[float, float] = (0.3, 2.0)     # kg

    # EE(x≈0.4) ~ box(x≥0.60) 최소거리 0.202m > 0.15m → trivial success 없음
    grasp_dist_threshold: float = 0.20   # ee ~ box 거리 [m]
    place_dist_threshold: float = 0.12   # ee ~ goal 거리 [m] (goal 최소거리 0.145m > 0.12m ✓)

    student_mode: bool = False    # True면 Student 관측 반환


@configclass
class WarehouseManipulationStudentEnvCfg(WarehouseManipulationEnvCfg):
    """Student 훈련용 — 특권 정보 없음."""
    observation_space = STUDENT_OBS_DIM
    student_mode = True


class WarehouseManipulationEnv(DirectRLEnv):
    cfg: WarehouseManipulationEnvCfg

    def __init__(self, cfg: WarehouseManipulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        d = self.device

        body_names = list(self.robot.data.body_names)
        self._ee_body_idx = body_names.index("panda_hand")  # body_pos_w 용 (base 포함)
        # get_jacobians()는 base body(index 0)를 제외하므로 offset -1
        self._jac_body_idx = self._ee_body_idx - 1
        print(f"[DEBUG] body_names ({len(body_names)}): {body_names}")
        print(f"[DEBUG] panda_hand → body_pos_w idx={self._ee_body_idx}, jacobian idx={self._jac_body_idx}")

        self._goal_pos_w  = torch.zeros(n, 3, device=d)
        self._box_mass    = torch.ones(n, device=d)
        self._grasped     = torch.zeros(n, dtype=torch.bool, device=d)
        self._actions     = torch.zeros(n, 4, device=d)
        self._prev_dist_ee_box   = torch.full((n,), 999.0, device=d)
        self._prev_dist_box_goal = torch.full((n,), 999.0, device=d)
        # grasp 시점의 박스 위치 저장 (proximity grasp: 팔이 박스를 밀지 못하도록 freeze)
        self._frozen_box_state   = torch.zeros(n, 13, device=d)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # Franka Panda
        franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.robot = Articulation(franka_cfg)

        # 박스 (크기는 reset에서 DR 적용)
        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.06, 0.06),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
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

        # Cartesian delta: actions[:, :3] = [dx, dy, dz] world frame, actions[:, 3] = gripper
        delta_pos = self._actions[:, :3] * 0.03  # max 3cm/step

        # Translational Jacobian for panda_hand (body 8), arm joints (DOF 0-6)
        # jac shape: [N, num_bodies, 6, num_dofs] — rows 0:3 = translational
        jac = self.robot.root_physx_view.get_jacobians()
        if not hasattr(self, '_jac_debug_count'):
            self._jac_debug_count = 0
        J = jac[:, self._jac_body_idx, :3, :7]   # [N, 3, 7]  (get_jacobians: base 제외)

        # Damped Least Squares IK: Δq = J^T (J J^T + λI)^{-1} Δx
        lam = 0.05
        JT = J.transpose(-2, -1)                                          # [N, 7, 3]
        JJT = torch.bmm(J, JT)                                            # [N, 3, 3]
        JJT_reg = JJT + lam * torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        J_dls = torch.bmm(JT, torch.linalg.inv(JJT_reg))                  # [N, 7, 3]
        delta_q = torch.bmm(J_dls, delta_pos.unsqueeze(-1)).squeeze(-1)   # [N, 7]

        if self._jac_debug_count < 3:
            self._jac_debug_count += 1
            print(f"[DEBUG IK] J[0] norm={J[0].norm():.4f}")
            print(f"[DEBUG IK] delta_pos[0]={delta_pos[0].detach().cpu().tolist()}")
            print(f"[DEBUG IK] delta_q[0]={delta_q[0].detach().cpu().tolist()}")
            print(f"[DEBUG IK] joint_pos[0,:7]={self.robot.data.joint_pos[0,:7].detach().cpu().tolist()}")

        joint_target = self.robot.data.joint_pos[:, :7] + delta_q
        self.robot.set_joint_position_target(joint_target, joint_ids=list(range(7)))

        # Gripper: action[:, 3] ∈ [-1, 1] → [0, 0.04]m
        gripper_pos = ((self._actions[:, 3:4] + 1.0) / 2.0) * 0.04
        self.robot.set_joint_position_target(
            gripper_pos.expand(-1, 2).clone(), joint_ids=[7, 8]
        )

        # Proximity grasp freeze
        if self._grasped.any():
            grasped_ids = self._grasped.nonzero(as_tuple=True)[0]
            frozen = self._frozen_box_state[grasped_ids].clone()
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
            # Student: 특권 정보 없음 (goal 상대좌표만 제공)
            obs = torch.cat([
                ee_pos,                          # (N, 3)
                gripper_w,                       # (N, 1)
                self._goal_pos_w - ee_pos,       # (N, 3) goal 상대좌표
                joint_pos[:, :9],                # (N, 9)
                joint_vel[:, :9],                # (N, 9)
            ], dim=1)   # (N, 25)
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

        if not hasattr(self, '_pos_debug_done'):
            self._pos_debug_done = True
            print(f"[DEBUG POS] ee_pos[0]={ee_pos[0].detach().cpu().tolist()}")
            print(f"[DEBUG POS] box_pos[0]={box_pos[0].detach().cpu().tolist()}")
            print(f"[DEBUG POS] dist mean={dist_ee_box.mean():.3f} min={dist_ee_box.min():.3f} max={dist_ee_box.max():.3f}")
        dist_ee_goal = (ee_pos - self._goal_pos_w).norm(dim=1)

        # 파지 판정
        newly_grasped = (~self._grasped) & (dist_ee_box < self.cfg.grasp_dist_threshold)
        self._grasped |= newly_grasped
        # grasp 발동 순간 박스 상태 저장 (이후 freeze에 사용)
        if newly_grasped.any():
            new_ids = newly_grasped.nonzero(as_tuple=True)[0]
            self._frozen_box_state[new_ids] = self.box.data.root_state_w[new_ids].clone()

        # 낙하 판정: 테이블(z=0.5m)에서 떨어진 경우
        dropped = self._grasped & (box_pos[:, 2] < 0.45)

        # 거치 성공: EE가 목표 위치에 도달 (박스는 비물리적 proximity 파지이므로 EE 기준)
        placed = self._grasped & (dist_ee_goal < self.cfg.place_dist_threshold)

        # exp(-dist) 절대거리 기반: 랜덤 정책도 박스 방향 gradient 받음
        # delta 기반은 랜덤 정책에서 기댓값=0 → bootstrap 불가
        not_grasped = (~self._grasped).float()

        approach  = self.cfg.rew_approach  * torch.exp(-dist_ee_box  * 10.0) * not_grasped
        transport = self.cfg.rew_transport * torch.exp(-dist_ee_goal *  5.0) * self._grasped.float()

        self._prev_dist_ee_box   = dist_ee_box.detach()
        self._prev_dist_box_goal = dist_ee_goal.detach()

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

        dist_ee_goal = (ee_pos - self._goal_pos_w).norm(dim=1)
        placed  = self._grasped & (dist_ee_goal < self.cfg.place_dist_threshold)
        dropped = self._grasped & (box_pos[:, 2] < 0.45)

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
        box_state[:, 0] = self.scene.env_origins[env_ids_t, 0] + sample_uniform(0.60, 0.75, (n,), device=self.device)
        box_state[:, 1] = self.scene.env_origins[env_ids_t, 1] + sample_uniform(-0.2, 0.2, (n,), device=self.device)
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

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """End-effector 위치와 방향 반환."""
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_pos, ee_quat
