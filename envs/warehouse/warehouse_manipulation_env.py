"""Phase 2 — 창고 Pick & Place 환경.

로봇: Franka Panda (7-DOF 암 + 평행 그리퍼)
임무: 박스를 집어 지정 선반 위치에 내려놓기

관측 (31-dim):
  box_or_goal_rel(3) + box_quat(4) + box_mass(1) +
  gripper(1) + goal_rel(3) + jpos(9) + jvel(9) + grasped(1)

  box_or_goal_rel:
    not grasped → box_pos - ee_pos  (approach 방향)
    grasped     → box_pos_carried - goal_pos  (transport 방향, goal까지 벡터)
  grasped(1): 파지 여부 플래그 — 정책이 phase를 명시적으로 구분

Jetson 배포 시: box_or_goal_rel → RGB-D 추정값, box_quat → 카메라 추정, box_mass → 1.0
"""

from __future__ import annotations

import glob
import importlib.util
import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane

# NVIDIA 클라우드 에셋 베이스 URL (Isaac Sim 5.1)
_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

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

OBS_DIM = 31  # box_or_goal_rel(3)+box_quat(4)+box_mass(1)+gripper(1)+goal_rel(3)+jpos(9)+jvel(9)+grasped(1)
TEACHER_OBS_DIM = OBS_DIM
STUDENT_OBS_DIM = OBS_DIM


@configclass
class WarehouseManipulationEnvCfg(DirectRLEnvCfg):
    decimation = 2               # 120Hz sim / 2 = 60Hz policy
    episode_length_s = 2.0      # 120steps: 업데이트당 에피소드 5× 증가, VF 수렴 가속
    action_space = 4             # [dx, dy, dz, gripper] Cartesian delta control
    observation_space = OBS_DIM  # 31-dim
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2 ** 27,  # 128MB — 대규모 env 충돌 스택 오버플로 방지
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=3.0, replicate_physics=True
    )

    # 보상 가중치
    # Approach      : Exp(-dist_ee_box*5)    — EE가 박스 가까울수록 연속 유인 (비파지 시)
    # Transport     : Potential-based shaping — 박스가 goal 방향으로 이동한 만큼
    #                 shaping = rew_transport * (prev_dist - cur_dist)
    #                 3cm 직선 이동 시 +1.5/step (scale=50), random walk → E=0
    # Goal Prox     : Exp(-dist_box_goal*3)  — grasped 시 goal 근처일수록 지속 보상
    #                 scale=3 (5에서 축소): dist=0.35m에서 exp(-1.05)=0.35 (학습 가능 범위)
    # Transport Dst : -dist_box_goal * 3.0   — grasped 시 거리 패널티, VF 수렴 가속
    #                 scale×6: dist=0.35m → -1.05/step (기존 -0.175의 6배, gradient 강화)
    #   hover exploit 검증: V(hover@0.12m,600step) = -252 + 208 - 12 = -56 << V(place) = 800 ✓
    rew_approach:      float =  1.0    # Exp(-dist_ee_box*5) 배율
    rew_grasp:         float = 30.0    # 단발 grasp 유인
    rew_transport:     float = 50.0    # potential shaping 배율 — 3cm 이동 시 +1.5/step
    rew_goal_prox:     float =  5.0    # Exp(-dist_box_goal*3) 배율 — scale 증가로 먼 거리서도 gradient
    rew_transport_dst: float =  1.0    # dist_box_goal 패널티 — 리턴 분산 감소
    rew_place:         float = 800.0   # 대형 터미널 보상 유지
    rew_drop:          float =   0.0   # 낙하 패널티 제거 (박스 회피 전략 방지)
    rew_time:          float = -0.02   # 스텝 패널티

    # 박스 Domain Randomization
    box_size_range: tuple[float, float] = (0.04, 0.08)   # m (정육면체 한 변)
    box_mass_range: tuple[float, float] = (0.3, 2.0)     # kg

    # PackingTable 상면 z≈0.5m, box spawn z=0.5m, goal z=0.53m
    # reach_pose → EE z≈0.487m, dist_to_box≈0.013~0.159m (<0.35m)
    # → 에피소드 시작 즉시 grasp 발동, 학습은 순수 transport에 집중
    grasp_dist_threshold: float = 0.35   # ee ~ box 거리 [m] — reset 후 max 초기거리 0.159m → 0.35m으로 전 env 즉시 grasp 보장
    place_dist_threshold: float = 0.12   # box ~ goal 거리 [m]

    camera_noise_min: float = 0.0   # 카메라 노이즈 DR 하한 [m] — 0: 노이즈 없음 (Teacher 동등)
    camera_noise_max: float = 0.0   # 카메라 노이즈 DR 상한 [m] — grasp 반경 25cm >> 카메라 오차 2cm
    enable_background: bool = False   # True: 창고 배경+조명 로드 (GUI 시각화용, 훈련 시 False)


@configclass
class WarehouseManipulationStudentEnvCfg(WarehouseManipulationEnvCfg):
    """하위 호환용 alias — Teacher와 동일."""
    pass


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
        self._prev_dist_box_goal = torch.full((n,), 999.0, device=d)
        self._frozen_box_state   = torch.zeros(n, 13, device=d)
        self._camera_noise_std   = torch.full((n,), 0.03, device=d)  # 에피소드당 카메라 노이즈 레벨
        # grasp 시 EE→박스 offset (박스가 EE를 따라 이동하도록)
        self._grasp_ee_offset    = torch.zeros(n, 3, device=d)

        # place_rate 실시간 추적
        self._stat_placed   = 0
        self._stat_episodes = 0

        # 뷰포트 카메라: 로봇 왼쪽 뒤에서 작업대 방향으로
        # super().__init__ 이후 호출해야 viewport가 준비됨
        # 로봇 뒤 우측에서 통로 방향(+x)으로 — 선반 양쪽, 로봇+테이블 전경
        self.sim.set_camera_view(
            eye=[-1.5, -2.0, 1.5],
            target=[1.5, 0.5, 0.3],
        )

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

        # 박스 → YCB 003_cracker_box (isaacsim 패키지 위치에서 동적 탐색)
        def _find_ycb_cracker() -> str:
            spec = importlib.util.find_spec("isaacsim")
            if spec and spec.submodule_search_locations:
                isaacsim_dir = list(spec.submodule_search_locations)[0]
                matches = glob.glob(os.path.join(
                    isaacsim_dir, "extscache", "omni.replicator.core-*",
                    "omni", "replicator", "core", "tests", "data", "objects",
                    "003_cracker_box_physics.usd"
                ))
                if matches:
                    return matches[0]
            raise FileNotFoundError("003_cracker_box_physics.usd not found in isaacsim extscache")
        _YCB_CRACKER = _find_ycb_cracker()
        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=UsdFileCfg(
                usd_path=_YCB_CRACKER,
                scale=(0.7, 0.7, 0.7),  # 원본 ~7×5×20cm → ~5×3.5×14cm
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.50)),
        )
        self.box = RigidObject(box_cfg)

        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # 테이블 → PackingTable USD (산업용 작업대)
        table_spawn = UsdFileCfg(
            usd_path=f"{_ISAAC_CLOUD}/Isaac/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        )
        table_spawn.func("/World/envs/env_0/Table", table_spawn,
                         translation=(0.45, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["box"]   = self.box

        if self.cfg.enable_background:
            warehouse_cfg = UsdFileCfg(
                usd_path=f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
            )
            warehouse_cfg.func(
                "/World/Warehouse", warehouse_cfg,
                translation=(-2.95, -3.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )
            dome_light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.92, 1.0))
            dome_light.func("/World/DomeLight", dome_light)
            sl = sim_utils.SphereLightCfg(intensity=15000.0, color=(1.0, 0.97, 0.88), radius=0.08)
            sl.func("/World/SL0", sl, translation=(0.5,  0.0, 2.5))
            sl.func("/World/SL1", sl, translation=(0.5,  1.5, 2.5))
            sl.func("/World/SL2", sl, translation=(0.5, -1.5, 2.5))

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
        ee_pos, _ = self._get_ee_pose()
        joint_pos = self.robot.data.joint_pos   # (N, 9)
        joint_vel = self.robot.data.joint_vel   # (N, 9)
        gripper_w = joint_pos[:, 7:8] + joint_pos[:, 8:9]

        box_pos  = self.box.data.root_pos_w     # (N, 3)
        box_quat = self.box.data.root_quat_w    # (N, 4)

        # box_or_goal_rel: 상태에 따라 다른 벡터 제공
        #   not grasped → box_pos - ee_pos  (approach 방향: 박스 찾아가기)
        #   grasped     → box_pos_carried - goal_pos  (transport 방향: goal까지 남은 벡터)
        # 이렇게 하면 정책이 항상 "다음에 가야 할 방향"을 obs[0:3]에서 직접 읽을 수 있음
        box_pos_carried = ee_pos + self._grasp_ee_offset  # grasped 아닌 env는 의미없음
        approach_rel = box_pos - ee_pos                   # not grasped: EE→box
        transport_rel = box_pos_carried - self._goal_pos_w  # grasped: box→goal (잔여 벡터)

        grasped_f = self._grasped.float().unsqueeze(1)    # (N, 1)
        box_or_goal_rel = (
            approach_rel * (1.0 - grasped_f)             # not grasped: approach 방향
            + transport_rel * grasped_f                   # grasped: transport 방향
        )

        # 카메라 DR 노이즈 (box_or_goal_rel에만 적용)
        noise = torch.randn_like(box_or_goal_rel) * self._camera_noise_std.unsqueeze(1)
        box_or_goal_rel_noisy = box_or_goal_rel + noise

        obs = torch.cat([
            box_or_goal_rel_noisy,           # (N, 3) — approach 또는 transport 방향
            box_quat,                        # (N, 4)
            self._box_mass.unsqueeze(1),     # (N, 1)
            gripper_w,                       # (N, 1)
            self._goal_pos_w - ee_pos,       # (N, 3) — goal 위치 (절대 참조)
            joint_pos[:, :9],                # (N, 9)
            joint_vel[:, :9],                # (N, 9)
            self._grasped.float().unsqueeze(1),  # (N, 1) — 파지 플래그 (phase 구분)
        ], dim=1)   # (N, 31)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        dist_ee_box  = (ee_pos - box_pos).norm(dim=1)

        # 파지 판정
        newly_grasped = (~self._grasped) & (dist_ee_box < self.cfg.grasp_dist_threshold)
        self._grasped |= newly_grasped
        # grasp 발동 순간: offset=0 → box_pos_carried = ee_pos (항상)
        # threshold=0.35m이라 랜덤 offset 시 goal까지 목표 EE 위치가 불확정 → 학습 불가
        # offset=0이면 "EE를 goal로 이동"이라는 단순하고 일관된 task로 고정됨
        if newly_grasped.any():
            new_ids = newly_grasped.nonzero(as_tuple=True)[0]
            self._frozen_box_state[new_ids] = self.box.data.root_state_w[new_ids].clone()
            self._grasp_ee_offset[new_ids]  = 0.0  # offset 제거: box = EE 위치

        # 박스 현재 위치 (EE + offset으로 운반 중)
        box_pos_carried = ee_pos + self._grasp_ee_offset  # grasped 아닌 env는 의미없음
        dist_box_goal = (box_pos_carried - self._goal_pos_w).norm(dim=1)

        # grasp 첫 step: prev_dist를 현재 dist로 동기화 → transport delta spike 방지
        # (reset 시 box-goal 거리로 초기화했지만, EE 이동 후 실제 grasp 위치가 다를 수 있음)
        if newly_grasped.any():
            new_ids = newly_grasped.nonzero(as_tuple=True)[0]
            self._prev_dist_box_goal[new_ids] = dist_box_goal[new_ids].detach()

        # 낙하 판정 — 박스가 테이블 아래(z<0.30)로 떨어진 경우
        dropped = self._grasped & (box_pos[:, 2] < 0.30)

        # 거치 성공: 박스 위치가 goal에 도달 (EE가 박스를 실제로 운반해야 함)
        placed = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)

        not_grasped = (~self._grasped).float()
        grasped_f   = self._grasped.float()

        # Approach: Exp(-dist*5) — EE가 박스에 가까울수록 지수적 보상 (비파지 시만)
        approach = self.cfg.rew_approach * torch.exp(-dist_ee_box * 5.0) * not_grasped

        # Transport: Potential-based shaping — 박스가 goal 방향으로 이동한 만큼
        # shaping = rew_transport * (prev - cur)
        # 3cm 직선 이동: 50 * 0.03 = 1.5/step (clamp 불필요, 자연적으로 bounded)
        # random walk → E[delta] = 0 → E[transport] = 0 (advantage signal 없음)
        # 올바른 이동 → delta > 0 → 양의 advantage 생성 → gradient 발생
        delta_goal = self._prev_dist_box_goal - dist_box_goal  # 양수 = goal에 가까워짐
        transport  = self.cfg.rew_transport * delta_goal * grasped_f

        # Goal Proximity: Exp(-dist*3) — grasped 시 goal 근처일수록 지속 보상
        # scale=3 (기존 5→3): dist=0.35m에서 exp(-1.05)=0.35 (학습 가능한 gradient 범위)
        # scale=5이면 dist=0.35m에서 exp(-1.75)=0.17 (너무 작아 먼 거리서 gradient 거의 없음)
        goal_prox = self.cfg.rew_goal_prox * torch.exp(-dist_box_goal * 3.0) * grasped_f

        # Transport Dense: -dist_box_goal — grasped 시 거리 패널티 (dense gradient)
        # scale 3.0 (기존 0.5의 6배): dist=0.35m → -1.05/step (VF 학습 신호 강화)
        transport_dst = -self.cfg.rew_transport_dst * dist_box_goal * grasped_f

        # grasped 상태에서만 prev 업데이트 (비파지 시 오염 방지)
        self._prev_dist_box_goal = torch.where(
            self._grasped, dist_box_goal.detach(), self._prev_dist_box_goal
        )

        # 진단 로그 — tensorboard + 콘솔에서 학습 내부 상태 확인
        log = self.extras.setdefault("log", {})
        log["dist_ee_box"]   = dist_ee_box.mean().item()
        log["grasp_rate"]    = self._grasped.float().mean().item() * 100.0  # %
        log["dist_box_goal"] = (dist_box_goal * grasped_f).sum().item() / (grasped_f.sum().item() + 1e-6)

        rew = (
            approach
            + transport
            + goal_prox
            + transport_dst
            + self.cfg.rew_grasp * newly_grasped.float()
            + self.cfg.rew_place * placed.float()
            + self.cfg.rew_drop  * dropped.float()
            + self.cfg.rew_time
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
        terminated = placed
        timed_out  = self.episode_length_buf >= self.max_episode_length - 1

        done = terminated | timed_out
        self._stat_placed   += placed.sum().item()
        self._stat_episodes += done.sum().item()
        if self._stat_episodes > 0:
            self.extras.setdefault("log", {})["place_rate"] = self._stat_placed / self._stat_episodes * 100

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

        # Franka home-ish 자세: EE z≈0.487m — PackingTable 상면(z≈0.5m) 근접
        # [0,-0.785,0,-2.356,0,1.571,0.785]: 박스(z=0.5m)까지 dist≈0.05~0.15m < grasp_threshold
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
        box_state[:, 2] = self.scene.env_origins[env_ids_t, 2] + 0.5    # PackingTable 상면 z≈0.5m
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
        # prev_dist_box_goal: reset 직후 실제 box-goal 거리로 초기화
        # 999.0으로 두면 grasp 첫 step에서 transport reward가 폭발함 (999 - 0.35 = 998.65)
        # box spawn 위치와 goal 위치 기반으로 실제 초기 거리를 근사
        # (정확한 EE 위치는 physics step 전이라 미확정이므로 box_pos 기준)
        box_pos_reset = box_state[:, :3]
        goal_pos_reset = self._goal_pos_w[env_ids_t]
        init_dist = (box_pos_reset - goal_pos_reset).norm(dim=1)
        self._prev_dist_box_goal[env_ids_t] = init_dist.detach()
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
