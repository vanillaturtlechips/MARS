"""Phase 2 Transport 환경 — 운반 + 내려놓기 독립 학습.

Phase 2 Hierarchical 아키텍처의 Transport Policy 전용.
Force_grasp 100%로 시작, 박스를 goal까지 옮긴 뒤 그리퍼를 열어 물리적으로 내려놓기.

Phase 2 재설계의 핵심 변경 사항:
  1. obs에서 'goal - ee_pos'를 1번 위치에 명시 (기존: box_rel이 1번, grasped 시 (0,0,0))
  2. release 행동 분리: gripper action < -0.3 → 물리 handoff (기존: 박스를 쥔 채 placed)
  3. 보상: -dist×k/step (손익분기점 없음) (기존: delta/potential 모두 local optimum 존재)
  4. Transport + Place만 학습 (approach/grasp 없음)

관측 (23차원):
  goal_rel(3) + dist_to_goal(1) + ee_vel(3) +
  jpos(7) + jvel(7) + gripper_w(1) + is_grasped(1)

액션: 4D Cartesian IK [dx, dy, dz (±3cm/step), gripper (-1→open, +1→close)]
  gripper < -0.3 && grasped → 그리퍼 열어 박스 물리 해제
"""

from __future__ import annotations

import glob
import importlib.util
import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
except ImportError:
    from isaaclab_assets import FRANKA_PANDA_CFG  # type: ignore

_ISAAC_CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
TRANSPORT_OBS_DIM = 23


@configclass
class WarehouseTransportEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0
    action_space = 4
    observation_space = TRANSPORT_OBS_DIM
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(gpu_collision_stack_size=2 ** 27),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=3.0, replicate_physics=True
    )

    # 보상
    rew_carry_dist:   float = 0.3    # -dist × k / step (grasped 중). 손익분기점 없음
    rew_release_near: float = 50.0   # one-time: goal 0.25m 이내에서 release 시 보너스
    rew_place:        float = 300.0  # 박스가 goal 근처에 착지+정착
    rew_time:         float = -0.02  # 약한 시간 압박

    # 판정 기준
    near_release_dist:        float = 0.12   # 이 거리 이내일 때만 release 허용 (공간 gating)
    place_dist_threshold:     float = 0.15   # 착지 후 goal까지 이 거리 이내 → 성공
    release_action_threshold: float = -0.3   # gripper action 이 값 이하 → release
    settle_steps:             int   = 8      # release 후 N step 이상 지나야 place 판정

    # 커리큘럼 (train_transport.py에서 직접 설정)
    goal_spawn_dist: float = 0.10


class WarehouseTransportEnv(DirectRLEnv):
    cfg: WarehouseTransportEnvCfg

    def __init__(self, cfg: WarehouseTransportEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, d = self.num_envs, self.device

        body_names = list(self.robot.data.body_names)
        self._ee_body_idx  = body_names.index("panda_leftfinger")
        self._jac_body_idx = self._ee_body_idx - 1

        self._goal_pos_w       = torch.zeros(n, 3, device=d)
        self._grasped          = torch.zeros(n, dtype=torch.bool, device=d)
        self._pending          = torch.zeros(n, dtype=torch.bool, device=d)
        self._frozen_box_state = torch.zeros(n, 13, device=d)
        self._newly_released   = torch.zeros(n, dtype=torch.bool, device=d)
        self._steps_after_rel  = torch.zeros(n, device=d)
        self._actions          = torch.zeros(n, 4, device=d)

        self._stat_placed   = 0
        self._stat_episodes = 0
        self._stat_window   = 200

    # ── Scene ────────────────────────────────────────────────────────

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
                    "003_cracker_box_physics.usd",
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

    # ── Action ───────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        n = self.num_envs
        delta_pos = self._actions[:, :3] * 0.03

        # DLS IK
        jac     = self.robot.root_physx_view.get_jacobians()
        J       = jac[:, self._jac_body_idx, :3, :7]
        lam     = 0.05
        JT      = J.transpose(-2, -1)
        JJT_reg = torch.bmm(J, JT) + lam * torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        J_dls   = torch.bmm(JT, torch.linalg.inv(JJT_reg))
        delta_q = torch.bmm(J_dls, delta_pos.unsqueeze(-1)).squeeze(-1).clamp(-0.1, 0.1)
        new_q   = (self.robot.data.joint_pos[:, :7] + delta_q).clamp(-2.8, 2.8)

        gripper_pos = ((self._actions[:, 3:4] + 1.0) / 2.0) * 0.04
        target_q = self.robot.data.joint_pos.clone()
        target_q[:, :7] = new_q
        target_q[:, 7:9] = gripper_pos.expand(-1, 2)
        self.robot.set_joint_position_target(target_q)

        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_vel = self.robot.data.body_lin_vel_w[:, self._ee_body_idx]

        # Release: 공간 gating + gripper action → 물리 handoff
        # near_release_dist 이내일 때만 허용: random policy가 즉시 release하는 것을 방지
        dist_for_gate = (ee_pos - self._goal_pos_w).norm(dim=1)
        self._newly_released[:] = False
        wants_release = (self._grasped
                         & (self._actions[:, 3] < self.cfg.release_action_threshold)
                         & (dist_for_gate < self.cfg.near_release_dist))
        if wants_release.any():
            rel_ids = wants_release.nonzero(as_tuple=True)[0]
            state = self._frozen_box_state[rel_ids].clone()
            state[:, :3]    = ee_pos[rel_ids]  # 현재 EE 위치에서 해제
            state[:, 7:13]  = 0.0              # 속도 0: EE vel 인계 시 box 날아가는 버그 수정
            self.box.write_root_state_to_sim(state, rel_ids)
            self._grasped[rel_ids]         = False
            self._newly_released[rel_ids]  = True
            self._steps_after_rel[rel_ids] = 0

        # Kinematic lock: 아직 grasped인 envs
        if self._grasped.any():
            grasped_ids = self._grasped.nonzero(as_tuple=True)[0]
            frozen = self._frozen_box_state[grasped_ids].clone()
            frozen[:, :3]   = ee_pos[grasped_ids]   # offset=0: box = EE
            frozen[:, 7:13] = 0.0
            self.box.write_root_state_to_sim(frozen, grasped_ids)

    # ── Obs ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        ee_pos    = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_vel    = self.robot.data.body_lin_vel_w[:, self._ee_body_idx]
        jpos      = self.robot.data.joint_pos
        jvel      = self.robot.data.joint_vel
        gripper_w = jpos[:, 7:8] + jpos[:, 8:9]
        box_pos   = self.box.data.root_pos_w

        # goal_rel:
        #   grasped  → goal - ee_pos  (박스=EE이므로 EE를 goal로 이동시키면 됨)
        #   released → goal - box_pos (박스가 goal 근처에 착지했는지 추적)
        ref_pos  = torch.where(self._grasped.unsqueeze(1).expand(-1, 3), ee_pos, box_pos)
        goal_rel = self._goal_pos_w - ref_pos          # (N, 3)
        dist     = goal_rel.norm(dim=1, keepdim=True)  # (N, 1)

        obs = torch.cat([
            goal_rel,                                   # 3: 1번 위치에 핵심 방향 신호
            dist,                                       # 1: 거리 스칼라
            ee_vel,                                     # 3: EE 속도 (모멘텀 인식)
            jpos[:, :7],                                # 7: arm joints
            jvel[:, :7],                                # 7: arm joints velocity
            gripper_w,                                  # 1: 그리퍼 상태
            self._grasped.float().unsqueeze(1),         # 1: carrying vs released
        ], dim=1)  # (N, 23)
        return {"policy": obs}

    # ── Reward ───────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_body_idx]
        box_pos = self.box.data.root_pos_w

        # Step-1 force_grasp 활성화 (HOME_EE 추정 오류 방지)
        if self._pending.any():
            act_mask = self._pending & ~self._grasped
            if act_mask.any():
                act_ids  = act_mask.nonzero(as_tuple=True)[0]
                ee_now   = ee_pos[act_ids]
                env_orig = self.scene.env_origins[act_ids]
                n_act    = len(act_ids)

                # goal: EE 기준 goal_spawn_dist 거리에 랜덤 배치
                r  = sample_uniform(
                    self.cfg.goal_spawn_dist * 0.7,
                    self.cfg.goal_spawn_dist * 1.3,
                    (n_act,), device=self.device,
                )
                th = sample_uniform(0.0, 6.2832, (n_act,), device=self.device)
                lx = (ee_now[:, 0] - env_orig[:, 0] + r * torch.cos(th)).clamp(0.55, 1.45)
                ly = (ee_now[:, 1] - env_orig[:, 1] + r * torch.sin(th)).clamp(-0.30, 0.30)
                self._goal_pos_w[act_ids, 0] = env_orig[:, 0] + lx
                self._goal_pos_w[act_ids, 1] = env_orig[:, 1] + ly
                self._goal_pos_w[act_ids, 2] = 1.15

                # 박스를 EE 위치에 snap
                state = self.box.data.root_state_w[act_ids].clone()
                state[:, :3]   = ee_now
                state[:, 7:13] = 0.0
                self.box.write_root_state_to_sim(state, act_ids)
                self._frozen_box_state[act_ids] = state
                self._grasped[act_ids]          = True
                self._pending[act_ids]          = False

        grasped_f = self._grasped.float()
        ref_pos   = torch.where(self._grasped.unsqueeze(1).expand(-1, 3), ee_pos, box_pos)
        dist_to_goal = (ref_pos - self._goal_pos_w).norm(dim=1)

        # [1] Carry: 거리에 비례한 step 패널티 (grasped 중만. 손익분기점 없음)
        rew_carry = -self.cfg.rew_carry_dist * dist_to_goal * grasped_f

        # [2] Release near goal: goal 0.25m 이내에서 그리퍼 열면 one-time 보너스
        rew_release = self.cfg.rew_release_near * (
            self._newly_released & (dist_to_goal < self.cfg.near_release_dist)
        ).float()

        # [3] Place: release 후 settle_steps 이상 경과 + goal 이내 착지
        self._steps_after_rel = torch.where(
            ~self._grasped,
            self._steps_after_rel + 1.0,
            torch.zeros_like(self._steps_after_rel),
        )
        settled = self._steps_after_rel >= self.cfg.settle_steps
        placed  = ~self._grasped & settled & (dist_to_goal < self.cfg.place_dist_threshold)
        rew_place = self.cfg.rew_place * placed.float()

        # [4] 시간 패널티 (약함)
        rew_time = self.cfg.rew_time

        # Logging
        log = self.extras.setdefault("log", {})
        log["dist_to_goal"]   = dist_to_goal.mean().item()
        log["carry_dist"]     = (dist_to_goal * grasped_f).sum().item() / (grasped_f.sum().item() + 1e-6)
        log["grasp_rate"]     = grasped_f.mean().item() * 100.0
        log["release_rate"]   = self._newly_released.float().mean().item() * 100.0
        log["placed_this_step"] = placed.float().mean().item() * 100.0

        self._stat_placed   += placed.sum().item()
        self._stat_episodes += placed.sum().item()  # 임시 (done에서 정확히 계산)

        return rew_carry + rew_release + rew_place + rew_time

    # ── Done ─────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        box_pos = self.box.data.root_pos_w
        ref_pos = torch.where(self._grasped.unsqueeze(1).expand(-1, 3),
                              self.robot.data.body_pos_w[:, self._ee_body_idx], box_pos)

        dist_to_goal = (ref_pos - self._goal_pos_w).norm(dim=1)
        settled  = self._steps_after_rel >= self.cfg.settle_steps
        placed   = ~self._grasped & settled & (dist_to_goal < self.cfg.place_dist_threshold)
        fell_off = box_pos[:, 2] < 0.9   # 테이블에서 떨어진 경우 조기 종료
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        done = placed | fell_off | timed_out

        self._stat_placed   += (placed & done).sum().item()
        self._stat_episodes += done.sum().item()
        if self._stat_episodes >= self._stat_window:
            self._stat_placed   = 0
            self._stat_episodes = 0
        if self._stat_episodes > 0:
            self.extras.setdefault("log", {})["place_rate"] = (
                self._stat_placed / self._stat_episodes * 100.0
            )

        return placed | fell_off, timed_out

    # ── Reset ────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_t = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.long()
        n = env_ids_t.shape[0]

        # 홈 포즈 (그리퍼 닫힌 상태: 이미 박스를 쥐고 있음)
        home_pose = torch.tensor(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0],
            device=self.device,
        ).unsqueeze(0).expand(n, -1)
        self.robot.set_joint_position_target(home_pose, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(home_pose, torch.zeros_like(home_pose), env_ids=env_ids_t)

        # 박스를 일단 EE 근처에 배치 (step-1에서 실제 EE로 snap)
        box_state = self.box.data.default_root_state[env_ids_t].clone()
        box_state[:, :3]   = self.scene.env_origins[env_ids_t] + torch.tensor([0.75, 0.0, 1.15], device=self.device)
        box_state[:, 7:13] = 0.0
        self.box.write_root_state_to_sim(box_state, env_ids_t)

        # goal은 step-1에서 실제 EE 기준으로 설정 (pending 표시)
        self._grasped[env_ids_t]          = False
        self._pending[env_ids_t]          = True
        self._frozen_box_state[env_ids_t] = 0.0
        self._newly_released[env_ids_t]   = False
        self._steps_after_rel[env_ids_t]  = 0.0

    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.robot.data.body_pos_w[:, self._ee_body_idx],
            self.robot.data.body_quat_w[:, self._ee_body_idx],
        )
