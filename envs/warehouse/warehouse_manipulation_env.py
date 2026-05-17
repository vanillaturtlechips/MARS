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
    (0.4,  0.2, 0.05),
    (0.4, -0.2, 0.05),
    (0.5,  0.1, 0.05),
    (0.5, -0.1, 0.05),
]

TEACHER_OBS_DIM = 33
STUDENT_OBS_DIM = 25


@configclass
class WarehouseManipulationEnvCfg(DirectRLEnvCfg):
    decimation = 2               # 120Hz sim / 2 = 60Hz policy
    episode_length_s = 15.0
    action_space = 9             # 7 관절 위치 + 2 그리퍼 (Franka)
    observation_space = TEACHER_OBS_DIM   # Teacher 모드 기본
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=3.0, replicate_physics=True
    )

    # 보상 가중치
    rew_approach:  float =  0.05   # 박스 접근 (호버링 꼼수 억제용으로 0.1→0.05)
    rew_approach_cap: float = 1.0  # 스텝당 approach reward 상한 (호버링 cap)
    rew_grasp:     float =  5.0    # 파지 성공
    rew_transport: float =  0.1    # 목표로 이송
    rew_place:     float = 20.0    # 거치 성공
    rew_drop:      float = -10.0   # 낙하 패널티

    # 박스 Domain Randomization
    box_size_range: tuple[float, float] = (0.04, 0.08)   # m (정육면체 한 변)
    box_mass_range: tuple[float, float] = (0.3, 2.0)     # kg

    # 파지 판정 (커리큘럼: 초기 8cm → 수렴 후 타이트하게 조임)
    grasp_dist_threshold: float = 0.08   # ee ~ box 거리 [m] (0.03→0.08)
    place_dist_threshold: float = 0.05   # box ~ goal 거리 [m]

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

        self._goal_pos_w  = torch.zeros(n, 3, device=d)
        self._box_mass    = torch.ones(n, device=d)
        self._grasped     = torch.zeros(n, dtype=torch.bool, device=d)
        self._actions     = torch.zeros(n, 9, device=d)

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
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.6, 0.1), metallic=0.0
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.03)),
        )
        self.box = RigidObject(box_cfg)

        spawn_ground_plane("/World/ground", GroundPlaneCfg())
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
        # 관절 위치 범위로 스케일링
        joint_pos_target = self._actions * torch.pi   # [-π, π]
        self.robot.set_joint_position_target(joint_pos_target)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        ee_pos, _  = self._get_ee_pose()
        joint_pos  = self.robot.data.joint_pos        # (N, 9)
        joint_vel  = self.robot.data.joint_vel        # (N, 9)
        gripper_w  = (joint_pos[:, 7:8] + joint_pos[:, 8:9])  # 그리퍼 폭

        if self.cfg.student_mode:
            # Student: 특권 정보 없음
            obs = torch.cat([
                ee_pos,                  # (N, 3)
                gripper_w,               # (N, 1)
                self._goal_pos_w,        # (N, 3)  — 근사 (사전 제공)
                joint_pos[:, :9],        # (N, 9)
                joint_vel[:, :9],        # (N, 9)
            ], dim=1)   # (N, 25)
        else:
            # Teacher: 특권 정보 포함
            box_pos  = self.box.data.root_pos_w            # (N, 3)
            box_quat = self.box.data.root_quat_w           # (N, 4)
            obs = torch.cat([
                box_pos,                 # (N, 3)
                box_quat,                # (N, 4)
                self._box_mass.unsqueeze(1),  # (N, 1)
                ee_pos,                  # (N, 3)
                gripper_w,               # (N, 1)
                self._goal_pos_w,        # (N, 3)
                joint_pos[:, :9],        # (N, 9)
                joint_vel[:, :9],        # (N, 9)
            ], dim=1)   # (N, 33)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        dist_ee_box  = (ee_pos - box_pos).norm(dim=1)
        dist_box_goal = (box_pos - self._goal_pos_w).norm(dim=1)

        # 파지 판정: ee가 박스에 충분히 가까우면 grasped
        newly_grasped = (~self._grasped) & (dist_ee_box < self.cfg.grasp_dist_threshold)
        self._grasped |= newly_grasped

        # 낙하 판정: 파지 후 박스가 바닥으로 떨어진 경우
        dropped = self._grasped & (box_pos[:, 2] < 0.01)

        # 거치 성공
        placed = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)

        approach = (self.cfg.rew_approach * (1.0 / (dist_ee_box + 0.01))).clamp(max=self.cfg.rew_approach_cap)
        rew = (
            approach
            + self.cfg.rew_grasp   * newly_grasped.float()
            + self.cfg.rew_transport * self._grasped.float() * (1.0 / (dist_box_goal + 0.01))
            + self.cfg.rew_place   * placed.float()
            + self.cfg.rew_drop    * dropped.float()
        )
        return rew

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        box_pos = self.box.data.root_pos_w
        dist_box_goal = (box_pos - self._goal_pos_w).norm(dim=1)

        placed  = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)
        dropped = self._grasped & (box_pos[:, 2] < 0.01)

        terminated = placed | dropped
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

        # 로봇 초기 관절 자세 리셋
        default_joint = self.robot.data.default_joint_pos[env_ids_t]
        self.robot.set_joint_position_target(default_joint, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(default_joint, torch.zeros_like(default_joint), env_ids=env_ids_t)

        # 박스 위치 랜덤화 (Domain Randomization)
        box_state = self.box.data.default_root_state[env_ids_t].clone()
        box_state[:, :3] += self.scene.env_origins[env_ids_t]
        box_state[:, 0] += sample_uniform(0.3, 0.6, (n,), device=self.device)
        box_state[:, 1] += sample_uniform(-0.2, 0.2, (n,), device=self.device)
        box_state[:, 2]  = self.scene.env_origins[env_ids_t, 2] + 0.03
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

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """End-effector 위치와 방향 반환."""
        ee_pos  = self.robot.data.body_pos_w[:, -3]    # panda_hand 링크
        ee_quat = self.robot.data.body_quat_w[:, -3]
        return ee_pos, ee_quat
