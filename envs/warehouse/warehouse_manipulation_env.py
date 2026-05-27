"""Phase 2 — 창고 Full Pick & Place 환경 (정공법).

로봇: Franka Panda (7-DOF 암 + 평행 그리퍼)
임무: 테이블 위 박스를 직접 집어 목적지에 내려놓기
액션: 4D Cartesian IK [dx, dy, dz (±3cm/step), gripper (-1→open, +1→close)]

관측 (31차원):
  box_rel(3) + box_quat(4) + box_mass(1) + gripper(1) +
  goal_rel(3) + jpos(9) + jvel(9) + is_grasped(1)
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

_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
except ImportError:
    from isaaclab_assets import FRANKA_PANDA_CFG  # type: ignore

OBS_DIM = 31  # +1 for is_grasped


@configclass
class WarehouseManipulationEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    action_space = 4
    observation_space = OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            gpu_collision_stack_size=2 ** 27,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=3.0, replicate_physics=True
    )

    # 보상 가중치
    rew_approach:   float = 2.0    # exp(-d*5) 기반, not_grasped 시
    rew_grasp:      float = 50.0   # one-time grasp bonus (축소: 200→50)
    rew_transport:  float = 300.0  # delta 기반: (prev_dist - curr_dist) * grasped
    rew_transport_dense: float = 3.0  # 방향 힌트용 exp(-dist*3) * grasped
    rew_place:      float = 200.0  # 최종 거치 성공 (축소: 500→200)
    rew_time:       float = -0.1   # 시간 패널티 (완화: -0.5→-0.1)

    grasp_dist_threshold: float = 0.09
    place_dist_threshold: float = 0.20

    # 커리큘럼: 박스 spawn 거리 (EE 기준)
    # 훈련 스크립트에서 단계별로 올림
    box_spawn_dist: float = 0.15   # 시작: 0.15m (grasp_threshold보다 충분히 멀게)

    force_grasp_on_reset: bool = False
    enable_background:    bool = False


class WarehouseManipulationEnv(DirectRLEnv):
    cfg: WarehouseManipulationEnvCfg

    def __init__(self, cfg: WarehouseManipulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        d = self.device

        body_names = list(self.robot.data.body_names)
        self._ee_body_idx  = body_names.index("panda_leftfinger")
        self._jac_body_idx = self._ee_body_idx - 1

        self._goal_pos_w         = torch.zeros(n, 3, device=d)
        self._box_mass           = torch.ones(n, device=d)
        self._grasped            = torch.zeros(n, dtype=torch.bool, device=d)
        self._actions            = torch.zeros(n, self.cfg.action_space, device=d)
        self._grasp_ee_offset    = torch.zeros(n, 3, device=d)
        self._frozen_box_state   = torch.zeros(n, 13, device=d)
        self._prev_dist_ee_box   = torch.full((n,), 999.0, device=d)
        self._prev_dist_box_goal = torch.full((n,), 999.0, device=d)

        self._stat_placed   = 0
        self._stat_episodes = 0
        self._stat_window   = 500  # 최근 N 에피소드 기준 place_rate 계산

    def _setup_scene(self):
        franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        franka_cfg.init_state.pos = (0.0, 0.0, 0.80)
        self.robot = Articulation(franka_cfg)

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
            raise FileNotFoundError("003_cracker_box_physics.usd not found")

        box_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=UsdFileCfg(
                usd_path=_find_ycb_cracker(),
                scale=(0.7, 0.7, 0.7),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, 1.15)),
        )
        self.box = RigidObject(box_cfg)

        spawn_ground_plane("/World/ground", GroundPlaneCfg())

        table_spawn = UsdFileCfg(
            usd_path=f"{_ISAAC_CLOUD}/Isaac/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        table_spawn.func("/World/envs/env_0/Table", table_spawn,
                         translation=(1.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))

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
            warehouse_cfg.func("/World/Warehouse", warehouse_cfg,
                               translation=(-2.95, -3.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        n = self.num_envs
        delta_pos = self._actions[:, :3] * 0.03

        # DLS IK
        jac = self.robot.root_physx_view.get_jacobians()
        J   = jac[:, self._jac_body_idx, :3, :7]
        lam = 0.05
        JT      = J.transpose(-2, -1)
        JJT_reg = torch.bmm(J, JT) + lam * torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        J_dls   = torch.bmm(JT, torch.linalg.inv(JJT_reg))
        delta_q = torch.bmm(J_dls, delta_pos.unsqueeze(-1)).squeeze(-1).clamp(-0.1, 0.1)

        new_q = (self.robot.data.joint_pos[:, :7] + delta_q).clamp(-2.8, 2.8)

        # 그리퍼: action[-1] > 0 → 닫기, < 0 → 열기
        gripper_pos = ((self._actions[:, 3:4] + 1.0) / 2.0) * 0.04

        target_q = self.robot.data.joint_pos.clone()
        target_q[:, :7] = new_q
        target_q[:, 7:9] = gripper_pos.expand(-1, 2)
        self.robot.set_joint_position_target(target_q)

        # grasped 상태: 박스를 EE에 고정 (물리적 그리퍼 대신 kinematic lock)
        if self._grasped.any():
            ee_pos, _ = self._get_ee_pose()
            grasped_ids = self._grasped.nonzero(as_tuple=True)[0]
            frozen = self._frozen_box_state[grasped_ids].clone()
            frozen[:, :3] = ee_pos[grasped_ids] + self._grasp_ee_offset[grasped_ids]
            frozen[:, 7:13] = 0.0
            self.box.write_root_state_to_sim(frozen, grasped_ids)

    def _get_observations(self) -> dict:
        ee_pos, _  = self._get_ee_pose()
        joint_pos  = self.robot.data.joint_pos
        joint_vel  = self.robot.data.joint_vel
        gripper_w  = joint_pos[:, 7:8] + joint_pos[:, 8:9]
        box_pos    = self.box.data.root_pos_w
        box_quat   = self.box.data.root_quat_w

        # grasped면 carried 위치 기준으로 goal_rel 계산
        box_pos_eff = torch.where(
            self._grasped.unsqueeze(1).expand(-1, 3),
            ee_pos + self._grasp_ee_offset,
            box_pos,
        )
        goal_rel = self._goal_pos_w - box_pos_eff

        obs = torch.cat([
            box_pos - ee_pos,           # 3
            box_quat,                   # 4
            self._box_mass.unsqueeze(1),# 1
            gripper_w,                  # 1
            goal_rel,                   # 3
            joint_pos[:, :9],           # 9
            joint_vel[:, :9],           # 9
            self._grasped.float().unsqueeze(1),  # 1  ← 파지 여부
        ], dim=1)  # (N, 31)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        dist_ee_box = (ee_pos - box_pos).norm(dim=1)

        # grasp 감지: 엄격해진 threshold (0.04m) — 실제로 접근해야 잡힘
        newly_grasped = (~self._grasped) & (dist_ee_box < self.cfg.grasp_dist_threshold)
        self._grasped |= newly_grasped
        if newly_grasped.any():
            new_ids = newly_grasped.nonzero(as_tuple=True)[0]
            self._frozen_box_state[new_ids] = self.box.data.root_state_w[new_ids].clone()
            self._grasp_ee_offset[new_ids]  = box_pos[new_ids] - ee_pos[new_ids]
            # grasp 직후 prev_dist 초기화 — delta 튐 방지
            box_pos_eff_new = ee_pos[new_ids] + self._grasp_ee_offset[new_ids]
            self._prev_dist_box_goal[new_ids] = (box_pos_eff_new - self._goal_pos_w[new_ids]).norm(dim=1)

        box_pos_eff = torch.where(
            self._grasped.unsqueeze(1).expand(-1, 3),
            ee_pos + self._grasp_ee_offset,
            box_pos,
        )
        dist_box_goal = (box_pos_eff - self._goal_pos_w).norm(dim=1)

        not_grasped = (~self._grasped).float()
        grasped_f   = self._grasped.float()

        # [1단계] Approach: exp shaping — not_grasped 시에만
        rew_approach = self.cfg.rew_approach * torch.exp(-dist_ee_box * 5.0) * not_grasped

        # [2단계] Grasp bonus (one-time)
        rew_grasp = self.cfg.rew_grasp * newly_grasped.float()

        # [3단계] Transport: dense(방향 힌트) + delta(이동 보상) 하이브리드
        dist_delta    = self._prev_dist_box_goal - dist_box_goal
        rew_transport = (self.cfg.rew_transport * dist_delta
                         + self.cfg.rew_transport_dense * torch.exp(-dist_box_goal * 3.0)
                         ) * grasped_f

        # [4단계] Place
        placed    = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)
        rew_place = self.cfg.rew_place * placed.float()

        log = self.extras.setdefault("log", {})
        log["dist_ee_box"]     = dist_ee_box.mean().item()
        log["grasp_rate"]      = grasped_f.mean().item() * 100.0
        log["dist_box_goal"]   = (dist_box_goal * grasped_f).sum().item() / (grasped_f.sum().item() + 1e-6)
        log["transport_delta"] = (self._prev_dist_box_goal - dist_box_goal)[self._grasped].mean().item() if self._grasped.any() else 0.0

        self._prev_dist_box_goal = dist_box_goal.detach().clone()

        return rew_approach + rew_grasp + rew_transport + rew_place + self.cfg.rew_time

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos, _ = self._get_ee_pose()
        box_pos_eff = torch.where(
            self._grasped.unsqueeze(1).expand(-1, 3),
            ee_pos + self._grasp_ee_offset,
            self.box.data.root_pos_w,
        )
        dist_box_goal = (box_pos_eff - self._goal_pos_w).norm(dim=1)
        placed    = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        done = placed | timed_out
        self._stat_placed   += placed.sum().item()
        self._stat_episodes += done.sum().item()
        # rolling window: 500 에피소드마다 리셋
        if self._stat_episodes >= self._stat_window:
            self._stat_placed   = 0
            self._stat_episodes = 0
        if self._stat_episodes > 0:
            self.extras.setdefault("log", {})["place_rate"] = (
                self._stat_placed / self._stat_episodes * 100
            )
        return placed, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.long()
        n = env_ids_t.shape[0]

        # 홈 포즈 (그리퍼 열린 상태)
        home_pose = torch.tensor(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04],
            device=self.device
        ).unsqueeze(0).expand(n, -1)
        self.robot.set_joint_position_target(home_pose, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(home_pose, torch.zeros_like(home_pose), env_ids=env_ids_t)

        self._box_mass[env_ids_t] = sample_uniform(0.3, 2.0, (n,), device=self.device)

        # 커리큘럼: 훈련 스크립트(train_manipulation.py)에서 box_spawn_dist를 직접 설정

        # 박스: EE 앞 box_spawn_dist 거리에 소환 (커리큘럼)
        ee_pos_reset, _ = self._get_ee_pose()
        ee_pos_n = ee_pos_reset[env_ids_t]

        theta = sample_uniform(0.0, 6.2832, (n,), device=self.device)
        d     = self.cfg.box_spawn_dist
        box_state = self.box.data.default_root_state[env_ids_t].clone()
        # 테이블 범위 안으로 클램핑: x [0.55, 1.45], y [-0.30, 0.30]
        box_state[:, 0] = (ee_pos_n[:, 0] + d * torch.cos(theta)).clamp(0.55, 1.45)
        box_state[:, 1] = (ee_pos_n[:, 1] + d * torch.sin(theta)).clamp(-0.30, 0.30)
        box_state[:, 2] = 1.15
        box_state[:, 7:13] = 0.0
        self.box.write_root_state_to_sim(box_state, env_ids_t)

        # goal: 박스에서 0.25~0.40m 떨어진 위치, 테이블 범위 안으로 클램핑
        theta = sample_uniform(0.0, 6.2832, (n,), device=self.device)
        r     = sample_uniform(0.25, 0.40,  (n,), device=self.device)
        self._goal_pos_w[env_ids_t, 0] = (box_state[:, 0] + r * torch.cos(theta)).clamp(0.55, 1.45)
        self._goal_pos_w[env_ids_t, 1] = (box_state[:, 1] + r * torch.sin(theta)).clamp(-0.30, 0.30)
        self._goal_pos_w[env_ids_t, 2] = 1.15

        self._grasped[env_ids_t]            = False
        self._frozen_box_state[env_ids_t]   = 0.0
        self._grasp_ee_offset[env_ids_t]    = 0.0
        self._prev_dist_ee_box[env_ids_t]   = 999.0
        self._prev_dist_box_goal[env_ids_t] = (box_state[:, :3] - self._goal_pos_w[env_ids_t]).norm(dim=1)

    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_pos, ee_quat
