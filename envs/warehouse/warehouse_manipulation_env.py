"""Phase 2 — 창고 Pick & Place 환경.

로봇: Franka Panda (7-DOF 암 + 평행 그리퍼)
임무: 박스를 집어 지정 선반 위치에 내려놓기
액션: 4D Cartesian IK [dx, dy, dz (±3cm/step), gripper (-1→open, +1→close)]

관측 (30차원):
  box_rel(3) + box_quat(4) + box_mass(1) + gripper(1) +
  goal_rel(3) + jpos(9) + jvel(9)
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
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

# NVIDIA 공개 에셋 S3 (Isaac Sim 5.1 기준)
_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
except ImportError:
    from isaaclab_assets import FRANKA_PANDA_CFG  # type: ignore

# 측면 배치 태스크: box spawn y≈0, goal y=±0.32-0.35 → free win 불가 (0.32m 이격)
# Teacher 100% 달성 검증 완료 (수직 lift보다 IK 단순: base 관절 주도)
PLACE_GOALS = [
    (0.30,  0.32, 0.50),
    (0.30, -0.32, 0.50),
    (0.32,  0.35, 0.50),
    (0.32, -0.35, 0.50),
]

OBS_DIM = 33


@configclass
class WarehouseManipulationEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 15.0
    action_space = 4             # 4D Cartesian IK: [dx, dy, dz, gripper]
    observation_space = OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2 ** 27,  # 5096 env 스택 오버플로 방지
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=3.0, replicate_physics=True
    )

    rew_approach:   float =  0.5    # -dist_ee_box * not_grasped
    rew_grasp:      float = 30.0   # one-time grasp bonus
    rew_transport:  float =  300.0  # Teacher 검증값 — 1000은 VF 분산 과대
    rew_align:      float =  0.3   # EE 실속도 방향 × goal_dir cosine (grasped gate) — 골디락스 검증값
    rew_goal_dist:  float =  2.0   # grasped_f gate → 잡은 동안만 절대거리 패널티
    rew_place:      float = 800.0  # Teacher 훈련값 복원
    rew_drop:       float =  0.0   # 비활성화: 매 스텝 페널티 → 보상 분산 폭발
    rew_time:       float = -0.02  # Teacher 훈련값

    box_size_range: tuple[float, float] = (0.04, 0.08)
    box_mass_range: tuple[float, float] = (0.3, 2.0)

    grasp_dist_threshold: float = 0.25  # Teacher 훈련값 복원 (0.11→0.25)
    place_dist_threshold: float = 0.12  # Teacher 훈련값 복원 (0.13→0.12)

    force_grasp_on_reset: bool = False
    enable_background: bool = False


class WarehouseManipulationEnv(DirectRLEnv):
    cfg: WarehouseManipulationEnvCfg

    def __init__(self, cfg: WarehouseManipulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        d = self.device

        body_names = list(self.robot.data.body_names)
        self._ee_body_idx  = body_names.index("panda_hand")
        self._jac_body_idx = self._ee_body_idx - 1  # get_jacobians: base 제외 offset

        self._goal_pos_w         = torch.zeros(n, 3, device=d)
        self._box_mass           = torch.ones(n, device=d)
        self._grasped            = torch.zeros(n, dtype=torch.bool, device=d)
        self._actions            = torch.zeros(n, self.cfg.action_space, device=d)
        self._prev_dist_box_goal = torch.full((n,), 999.0, device=d)
        self._frozen_box_state   = torch.zeros(n, 13, device=d)
        self._grasp_ee_offset    = torch.zeros(n, 3, device=d)

        self._stat_placed   = 0
        self._stat_episodes = 0

    def _setup_scene(self):
        franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.robot = Articulation(franka_cfg)

        # 박스 → YCB 003_cracker_box (isaacsim extscache 동적 탐색)
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

        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=UsdFileCfg(
                usd_path=_find_ycb_cracker(),
                scale=(0.7, 0.7, 0.7),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.50)),
        )
        self.box = RigidObject(box_cfg)

        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        # 테이블 → PackingTable USD (산업용 작업대, 상면 z≈0.50m)
        table_spawn = UsdFileCfg(
            usd_path=f"{_ISAAC_CLOUD}/Isaac/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )
        table_spawn.func("/World/envs/env_0/Table", table_spawn,
                         translation=(0.85, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["box"]   = self.box

        dome_light = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.9, 0.92, 1.0))
        dome_light.func("/World/DomeLight", dome_light)

        if self.cfg.enable_background:
            warehouse_cfg = UsdFileCfg(
                usd_path=f"{_ISAAC_CLOUD}/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
            )
            warehouse_cfg.func(
                "/World/Warehouse", warehouse_cfg,
                translation=(-2.95, -3.0, 0.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )
            sl = sim_utils.SphereLightCfg(intensity=15000.0, color=(1.0, 0.97, 0.88), radius=0.08)
            sl.func("/World/SL0", sl, translation=(0.5,  0.0, 2.5))
            sl.func("/World/SL1", sl, translation=(0.5,  1.5, 2.5))
            sl.func("/World/SL2", sl, translation=(0.5, -1.5, 2.5))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        n = self.num_envs
        delta_pos = self._actions[:, :3] * 0.03  # max 3cm/step

        # DLS IK: Δq = J^T (J J^T + λI)^{-1} Δx
        jac = self.robot.root_physx_view.get_jacobians()
        J   = jac[:, self._jac_body_idx, :3, :7]
        lam = 0.05  # 특이점 폭발 방지 (0.01에서 EE 17cm/step 버그 확인됨)
        JT      = J.transpose(-2, -1)
        JJT_reg = torch.bmm(J, JT) + lam * torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        J_dls   = torch.bmm(JT, torch.linalg.inv(JJT_reg))
        delta_q = torch.bmm(J_dls, delta_pos.unsqueeze(-1)).squeeze(-1)
        delta_q = delta_q.clamp(-0.1, 0.1)  # 관절당 최대 0.1rad/step

        joint_target = self.robot.data.joint_pos[:, :7] + delta_q
        self.robot.set_joint_position_target(joint_target, joint_ids=list(range(7)))

        gripper_pos = ((self._actions[:, 3:4] + 1.0) / 2.0) * 0.04
        self.robot.set_joint_position_target(gripper_pos.expand(-1, 2), joint_ids=[7, 8])

        if self._grasped.any():
            grasped_ids = self._grasped.nonzero(as_tuple=True)[0]
            ee_pos, _ = self._get_ee_pose()
            frozen = self._frozen_box_state[grasped_ids].clone()
            frozen[:, :3] = ee_pos[grasped_ids] + self._grasp_ee_offset[grasped_ids]
            frozen[:, 7:13] = 0.0
            self.box.write_root_state_to_sim(frozen, grasped_ids)

    def _get_observations(self) -> dict:
        ee_pos, _      = self._get_ee_pose()
        joint_pos      = self.robot.data.joint_pos
        joint_vel      = self.robot.data.joint_vel
        gripper_w      = joint_pos[:, 7:8] + joint_pos[:, 8:9]
        box_pos_carried = ee_pos + self._grasp_ee_offset
        box_pos  = self.box.data.root_pos_w
        box_quat = self.box.data.root_quat_w

        # goal direction: actual box pos before grasp, carried pos after grasp
        # (avoids discontinuity when grasp_ee_offset jumps from 0 to box-ee)
        goal_rel = self._goal_pos_w - torch.where(
            self._grasped.unsqueeze(1).expand(-1, 3),
            box_pos_carried,
            box_pos,
        )

        obs = torch.cat([
            box_pos - ee_pos,
            box_quat,
            self._box_mass.unsqueeze(1),
            gripper_w,
            goal_rel,
            joint_pos[:, :9],
            joint_vel[:, :9],
            ee_pos,
        ], dim=1)   # (N, 33)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        dist_ee_box = (ee_pos - box_pos).norm(dim=1)

        newly_grasped = (~self._grasped) & (dist_ee_box < self.cfg.grasp_dist_threshold)
        self._grasped |= newly_grasped
        if newly_grasped.any():
            new_ids = newly_grasped.nonzero(as_tuple=True)[0]
            self._frozen_box_state[new_ids] = self.box.data.root_state_w[new_ids].clone()
            self._grasp_ee_offset[new_ids]  = box_pos[new_ids] - ee_pos[new_ids]

        box_pos_carried = ee_pos + self._grasp_ee_offset
        dist_box_goal   = (box_pos_carried - self._goal_pos_w).norm(dim=1)
        goal_rel        = self._goal_pos_w - box_pos_carried  # box→goal 방향벡터 (align reward용)

        dropped = self._grasped & (box_pos[:, 2] < 0.30)
        placed  = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)

        not_grasped = (~self._grasped).float()
        grasped_f   = self._grasped.float()

        approach   = -self.cfg.rew_approach  * dist_ee_box  * not_grasped
        goal_dense = -self.cfg.rew_goal_dist * dist_box_goal * grasped_f

        # delta tracking: box_pos_carried (EE proxy when grasped) 사용
        # box.data.root_pos_w는 write_root_state_to_sim 타이밍 문제로 EE를 100% 추종하지 않음
        # → box_pos_carried = ee_pos + grasp_ee_offset 로 일관되게 계산
        dist_box_real = (box_pos_carried - self._goal_pos_w).norm(dim=1)
        delta_goal = (self._prev_dist_box_goal - dist_box_real).clamp(-0.1, 0.1)
        transport  = self.cfg.rew_transport * delta_goal * grasped_f

        self._prev_dist_box_goal = dist_box_real.detach()

        # EE 실속도 방향 × goal 방향 cosine (grasped 동안만)
        # body_lin_vel_w: 실제 EE 선속도(m/s) — 관절공간 아닌 태스크공간 직접 사용
        ee_vel   = self.robot.data.body_lin_vel_w[:, self._ee_body_idx, :]
        ee_speed = ee_vel.norm(dim=1, keepdim=True).clamp(min=1e-4)
        ee_vel_dir = ee_vel / ee_speed                                    # (N, 3) unit
        goal_dir   = goal_rel / (goal_rel.norm(dim=1, keepdim=True).clamp(min=1e-4))
        alignment  = (ee_vel_dir * goal_dir).sum(dim=1)                   # cosine [-1, 1]
        rew_align  = self.cfg.rew_align * alignment * grasped_f

        log = self.extras.setdefault("log", {})
        log["dist_ee_box"]   = dist_ee_box.mean().item()
        log["grasp_rate"]    = grasped_f.mean().item() * 100.0
        log["dist_box_goal"] = (dist_box_goal * grasped_f).sum().item() / (grasped_f.sum().item() + 1e-6)

        return (
            approach
            + goal_dense
            + transport
            + rew_align
            + self.cfg.rew_grasp * newly_grasped.float()
            + self.cfg.rew_place * placed.float()
            + self.cfg.rew_drop  * dropped.float()
            + self.cfg.rew_time
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        box_pos_carried = ee_pos + self._grasp_ee_offset
        dist_box_goal   = (box_pos_carried - self._goal_pos_w).norm(dim=1)
        placed  = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)
        dropped = self._grasped & (box_pos[:, 2] < 0.30)

        terminated = placed
        timed_out  = self.episode_length_buf >= self.max_episode_length - 1

        done = terminated | timed_out
        self._stat_placed   += placed.sum().item()
        self._stat_episodes += done.sum().item()
        if self._stat_episodes > 0:
            self.extras.setdefault("log", {})["place_rate"] = (
                self._stat_placed / self._stat_episodes * 100
            )

        return terminated, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.long()
        n = env_ids_t.shape[0]

        reach_pose = torch.tensor(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            device=self.device
        ).unsqueeze(0).expand(n, -1)
        self.robot.set_joint_position_target(reach_pose, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(reach_pose, torch.zeros_like(reach_pose), env_ids=env_ids_t)

        self._box_mass[env_ids_t] = sample_uniform(
            self.cfg.box_mass_range[0], self.cfg.box_mass_range[1], (n,), device=self.device
        )

        box_state = self.box.data.default_root_state[env_ids_t].clone()

        if self.cfg.force_grasp_on_reset:
            # force_grasp: box를 reach_pose EE 실측 위치에 spawn (진단 확인값)
            # EE local = [0.307, 0.000, 0.590] — 기존 [0.33, 0, 0.50]은 table 높이로 잘못됨
            # step 1 box teleport(0.093m) 제거, noisy_box_rel 일관성 확보
            box_state[:, 0] = self.scene.env_origins[env_ids_t, 0] + 0.307
            box_state[:, 1] = self.scene.env_origins[env_ids_t, 1] + 0.000
            box_state[:, 2] = self.scene.env_origins[env_ids_t, 2] + 0.590
        else:
            box_state[:, 0] = self.scene.env_origins[env_ids_t, 0] + sample_uniform(0.45, 0.55, (n,), device=self.device)
            box_state[:, 1] = self.scene.env_origins[env_ids_t, 1] + sample_uniform(-0.15, 0.15, (n,), device=self.device)
            box_state[:, 2] = self.scene.env_origins[env_ids_t, 2] + 0.50
        self.box.write_root_state_to_sim(box_state, env_ids_t)

        if self.cfg.force_grasp_on_reset:
            # PLACE_GOALS: 고정 4개 위치 중 에피소드마다 랜덤 선택 (local frame)
            # Teacher 100% 달성 조건 그대로 → 랜덤 policy 성공률 ≈ 0% → advantage 차이 극명
            goal_indices = torch.randint(0, len(PLACE_GOALS), (n,), device=self.device)
            place_goals  = torch.tensor(PLACE_GOALS, device=self.device, dtype=torch.float32)
            self._goal_pos_w[env_ids_t, 0] = self.scene.env_origins[env_ids_t, 0] + place_goals[goal_indices, 0]
            self._goal_pos_w[env_ids_t, 1] = self.scene.env_origins[env_ids_t, 1] + place_goals[goal_indices, 1]
            self._goal_pos_w[env_ids_t, 2] = self.scene.env_origins[env_ids_t, 2] + place_goals[goal_indices, 2]
        else:
            # Curriculum: goal을 박스 반경 0.14~0.20m 내 spawn
            theta = sample_uniform(0.0, 6.2832, (n,), device=self.device)
            r     = sample_uniform(0.14, 0.20,  (n,), device=self.device)
            self._goal_pos_w[env_ids_t, 0] = box_state[:, 0] + r * torch.cos(theta)
            self._goal_pos_w[env_ids_t, 1] = box_state[:, 1] + r * torch.sin(theta)
            self._goal_pos_w[env_ids_t, 2] = self.scene.env_origins[env_ids_t, 2] + 0.50

        init_dist = (box_state[:, :3] - self._goal_pos_w[env_ids_t]).norm(dim=1)
        if self.cfg.force_grasp_on_reset:
            self._grasped[env_ids_t]          = True
            self._frozen_box_state[env_ids_t] = box_state.clone()
            self._grasp_ee_offset[env_ids_t]  = 0.0
            self._prev_dist_box_goal[env_ids_t] = init_dist
        else:
            self._grasped[env_ids_t]          = False
            self._frozen_box_state[env_ids_t] = 0.0
            self._grasp_ee_offset[env_ids_t]  = 0.0
            self._prev_dist_box_goal[env_ids_t] = init_dist

        # DEBUG: 첫 번째 env의 실제 좌표 출력
        if 0 in env_ids_t.tolist():
            idx = (env_ids_t == 0).nonzero(as_tuple=True)[0][0]
            box_local  = box_state[idx, :3] - self.scene.env_origins[0]
            goal_local = self._goal_pos_w[0] - self.scene.env_origins[0]
            dist_init  = (box_state[idx, :3] - self._goal_pos_w[0]).norm().item()
            print(f"[DBG] box_local={box_local.tolist()}, goal_local={goal_local.tolist()}, "
                  f"init_dist={dist_init:.3f}m")

    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_pos, ee_quat
