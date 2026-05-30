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

OBS_DIM = 23  # transport env과 동일: goal_rel(3)+dist(1)+ee_vel(3)+jpos7(7)+jvel7(7)+gripper(1)+grasped(1)


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
    rew_approach:   float = 0.3    # exp(-d*5) 기반, not_grasped 시 (낮춤: hover local optimum 방지)
    rew_grasp:      float = 50.0   # one-time grasp bonus (축소: 200→50)
    rew_transport:  float = 300.0  # (사용 안 함) delta 기반 — local optimum 유발하여 폐기
    rew_transport_dense: float = 0.0   # (사용 안 함)
    rew_carry_dist: float = 1.0    # transport env 동일: -dist×k/step 연속 패널티 (손익분기점 없음)
    rew_near_place: float = 80.0       # one-time: box가 0.22m 이내 첫 진입 시 보너스
    rew_place:      float = 200.0  # 최종 거치 성공 (축소: 500→200)
    rew_time:        float = -0.5   # approach 단계 시간 패널티 (not_grasped)
    rew_time_grasped: float = -0.1  # transport 단계: 음수 유지 → 이동해야만 net positive

    grasp_dist_threshold: float = 0.15
    place_dist_threshold: float = 0.15  # 0.10→0.15m 완화: place 경험 빈도 증가

    # release 관련 (Phase 2 transport에서 검증된 값)
    near_release_dist:        float = 0.20  # XY 거리: 이 안에서만 release 허용
    release_action_threshold: float = -0.3  # gripper action < 이 값 → release
    settle_steps:             int   = 10    # release 후 box 안착 판정 대기 (physics steps)
    rew_release_near:         float = 50.0  # goal 근처 release one-time bonus
    rew_grip_penalty:         float = 1.5   # goal 근처 gripper 닫힘 soft 패널티

    # 커리큘럼: 박스 spawn 거리 (EE 기준)
    # 훈련 스크립트에서 단계별로 올림
    box_spawn_dist:  float = 0.20   # 시작: 0.20m (grasp_threshold보다 충분히 멀게)
    goal_spawn_dist: float = 0.10   # transport env와 동일 (0.10m, 0.7~1.3× 변동)

    force_grasp_on_reset:  bool  = False
    force_grasp_fraction:  float = 0.50   # force_grasp 적용 비율 (0.50 = 50%)
    enable_background:     bool  = False


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
        self._cmd_ee_pos         = torch.zeros(n, 3, device=d)   # transport env 방식: 누적 위치 명령
        self._home_q             = torch.zeros(n, 7, device=d)   # reset 후 pending 중 joint 기준
        self._grasp_ee_offset    = torch.zeros(n, 3, device=d)
        self._frozen_box_state   = torch.zeros(n, 13, device=d)
        self._prev_dist_ee_box    = torch.full((n,), 999.0, device=d)
        self._prev_dist_box_goal  = torch.full((n,), 999.0, device=d)
        self._near_place_awarded  = torch.zeros(n, dtype=torch.bool, device=d)
        self._force_grasped_mask  = torch.zeros(n, dtype=torch.bool, device=d)
        self._force_grasp_pending = torch.zeros(n, dtype=torch.bool, device=d)

        # release 추적 텐서 (Phase 2 transport 방식)
        self._has_released    = torch.zeros(n, dtype=torch.bool,  device=d)
        self._steps_after_rel = torch.zeros(n, dtype=torch.long,  device=d)
        self._newly_released  = torch.zeros(n, dtype=torch.bool,  device=d)
        self._force_released  = torch.zeros(n, dtype=torch.bool,  device=d)
        # bootstrap: 훈련 스크립트에서 주입
        self._bootstrap_n  = 0
        self._bootstrap_p  = 0.0
        self._current_iter = 9999

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
        self._newly_released[:] = False
        self._force_released[:] = False
        # transport env과 동일: env step마다 1번만 cmd 누적 (decimation=2 호출 전)
        self._cmd_ee_pos += self._actions[:, :3] * 0.01

    def _apply_action(self) -> None:
        n = self.num_envs
        ee_pos, _ = self._get_ee_pose()

        # transport env과 동일: cmd_ee_pos 추적 방식 DLS IK
        delta_to_cmd = self._cmd_ee_pos - ee_pos
        jac = self.robot.root_physx_view.get_jacobians()
        J   = jac[:, self._jac_body_idx, :3, :7]
        lam = 0.05
        JT      = J.transpose(-2, -1)
        JJT_reg = torch.bmm(J, JT) + lam * torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        J_dls   = torch.bmm(JT, torch.linalg.inv(JJT_reg))
        delta_q = torch.bmm(J_dls, delta_to_cmd.unsqueeze(-1)).squeeze(-1).clamp(-0.3, 0.3)

        new_q = (self.robot.data.joint_pos[:, :7] + delta_q).clamp(-2.8, 2.8)

        # 그리퍼: action[-1] > 0 → 닫기, < 0 → 열기
        gripper_pos = ((self._actions[:, 3:4] + 1.0) / 2.0) * 0.04

        target_q = self.robot.data.joint_pos.clone()
        target_q[:, :7] = new_q
        target_q[:, 7:9] = gripper_pos.expand(-1, 2)
        self.robot.set_joint_position_target(target_q)

        # grasped 상태: 박스를 EE에 고정 (물리적 그리퍼 대신 kinematic lock)
        if self._grasped.any():
            grasped_ids = self._grasped.nonzero(as_tuple=True)[0]
            frozen = self._frozen_box_state[grasped_ids].clone()
            frozen[:, :3] = ee_pos[grasped_ids] + self._grasp_ee_offset[grasped_ids]
            frozen[:, 7:13] = 0.0
            self.box.write_root_state_to_sim(frozen, grasped_ids)

        # ── Release (Phase 2 transport 방식) ──────────────────────────
        # Bootstrap: 초기 N iter 동안 near-goal에서 강제 release
        if self._bootstrap_n > 0:
            p_now = self._bootstrap_p * max(0.0, 1.0 - self._current_iter / self._bootstrap_n)
            near_xy_bs = (ee_pos[:, :2] - self._goal_pos_w[:, :2]).norm(dim=1)
            force_rel = (self._grasped & (near_xy_bs < self.cfg.near_release_dist)
                         & (torch.rand(n, device=self.device) < p_now))
            self._actions[force_rel, 3] = -1.0
            self._force_released |= force_rel
        else:
            force_rel = torch.zeros(n, dtype=torch.bool, device=self.device)

        near_xy = (ee_pos[:, :2] - self._goal_pos_w[:, :2]).norm(dim=1)
        wants_release = (self._grasped
                         & (self._actions[:, 3] < self.cfg.release_action_threshold)
                         & (near_xy < self.cfg.near_release_dist)) | force_rel

        if wants_release.any():
            rel_ids = wants_release.nonzero(as_tuple=True)[0]
            rel_state = self._frozen_box_state[rel_ids].clone()
            rel_state[:, :3] = ee_pos[rel_ids]
            rel_state[:, 2] -= 0.08   # finger collision 방지 (Bug 10과 동일)
            rel_state[:, 7:13] = 0.0
            self.box.write_root_state_to_sim(rel_state, rel_ids)
            self._grasped[rel_ids]         = False
            self._newly_released[rel_ids]  = True
            self._has_released[rel_ids]    = True
            self._steps_after_rel[rel_ids] = 0

        # release 후 안착 대기 카운터
        self._steps_after_rel[~self._grasped & self._has_released] += 1

    def _get_observations(self) -> dict:
        ee_pos, _  = self._get_ee_pose()
        joint_pos  = self.robot.data.joint_pos
        joint_vel  = self.robot.data.joint_vel
        gripper_w  = joint_pos[:, 7:8] + joint_pos[:, 8:9]
        box_pos    = self.box.data.root_pos_w
        ee_vel     = self.robot.data.body_lin_vel_w[:, self._ee_body_idx]

        # transport env과 동일한 구조:
        # not_grasped → goal_rel = box-ee (approach 방향)
        # grasped     → goal_rel = goal-ee (transport 방향, transport model 호환)
        goal_rel = torch.where(
            self._grasped.unsqueeze(1).expand(-1, 3),
            self._goal_pos_w - ee_pos,   # grasped: box-to-goal (transport env 동일)
            box_pos - ee_pos,            # not grasped: ee-to-box (approach)
        )
        dist = goal_rel.norm(dim=1, keepdim=True)

        obs = torch.cat([
            goal_rel,                                    # 3
            dist,                                        # 1
            ee_vel,                                      # 3
            joint_pos[:, :7],                            # 7
            joint_vel[:, :7],                            # 7
            gripper_w,                                   # 1
            self._grasped.float().unsqueeze(1),          # 1
        ], dim=1)  # (N, 23)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        ee_pos, _ = self._get_ee_pose()
        box_pos   = self.box.data.root_pos_w

        # force_grasp: reset이 아닌 step 1에서 실제 EE 기준으로 활성화 (HOME_EE 추정 오류 완전 제거)
        if self._force_grasp_pending.any():
            act_mask = self._force_grasp_pending & ~self._grasped
            if act_mask.any():
                act_ids  = act_mask.nonzero(as_tuple=True)[0]
                ee_now   = ee_pos[act_ids]
                env_orig = self.scene.env_origins[act_ids]
                n_act    = len(act_ids)
                r  = sample_uniform(self.cfg.goal_spawn_dist * 0.7, self.cfg.goal_spawn_dist * 1.3, (n_act,), device=self.device)
                th = sample_uniform(0.0, 6.2832, (n_act,), device=self.device)
                lx = (ee_now[:, 0] - env_orig[:, 0] + r * torch.cos(th)).clamp(0.55, 1.45)
                ly = (ee_now[:, 1] - env_orig[:, 1] + r * torch.sin(th)).clamp(-0.30, 0.30)
                self._goal_pos_w[act_ids, 0] = env_orig[:, 0] + lx
                self._goal_pos_w[act_ids, 1] = env_orig[:, 1] + ly
                self._goal_pos_w[act_ids, 2] = 1.15
                # prev_dist = 실제 dist (spike 없음)
                g_vec = torch.stack([lx - (ee_now[:, 0] - env_orig[:, 0]),
                                     ly - (ee_now[:, 1] - env_orig[:, 1]),
                                     torch.zeros(n_act, device=self.device)], dim=1)
                self._prev_dist_box_goal[act_ids] = g_vec.norm(dim=1)
                self._grasp_ee_offset[act_ids]    = 0.0
                self._frozen_box_state[act_ids]   = self.box.data.root_state_w[act_ids].clone()
                self._grasped[act_ids]            = True
                self._force_grasped_mask[act_ids] = True
                self._force_grasp_pending[act_ids]= False

        dist_ee_box = (ee_pos - box_pos).norm(dim=1)

        # grasp 감지: 엄격해진 threshold (0.04m) — 실제로 접근해야 잡힘
        newly_grasped = (~self._grasped) & (dist_ee_box < self.cfg.grasp_dist_threshold)
        self._grasped |= newly_grasped
        if newly_grasped.any():
            new_ids = newly_grasped.nonzero(as_tuple=True)[0]
            self._frozen_box_state[new_ids] = self.box.data.root_state_w[new_ids].clone()
            # offset=0: box → EE에 snap (force_grasp와 동일한 obs → 학습된 transport 행동 전이)
            self._grasp_ee_offset[new_ids]  = 0.0
            # goal 재설정: grasp 시점 EE 기준 0.25~0.45m (force_grasp 분포 일치)
            n_new        = len(new_ids)
            env_orig_new = self.scene.env_origins[new_ids]
            r_new  = sample_uniform(self.cfg.goal_spawn_dist * 0.7, self.cfg.goal_spawn_dist * 1.3, (n_new,), device=self.device)
            th_new = sample_uniform(0.0, 6.2832, (n_new,), device=self.device)
            lx_new = (ee_pos[new_ids, 0] - env_orig_new[:, 0] + r_new * torch.cos(th_new)).clamp(0.55, 1.45)
            ly_new = (ee_pos[new_ids, 1] - env_orig_new[:, 1] + r_new * torch.sin(th_new)).clamp(-0.30, 0.30)
            self._goal_pos_w[new_ids, 0] = env_orig_new[:, 0] + lx_new
            self._goal_pos_w[new_ids, 1] = env_orig_new[:, 1] + ly_new
            self._goal_pos_w[new_ids, 2] = 1.15
            g_vec_new = torch.stack([lx_new - (ee_pos[new_ids, 0] - env_orig_new[:, 0]),
                                     ly_new - (ee_pos[new_ids, 1] - env_orig_new[:, 1]),
                                     torch.zeros(n_new, device=self.device)], dim=1)
            self._prev_dist_box_goal[new_ids] = g_vec_new.norm(dim=1)

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

        # [3단계] Transport: transport env과 동일한 -dist×k/step 연속 패널티
        # delta/potential은 local optimum 존재 → transport env에서 폐기, 동일하게 따름
        rew_transport = -self.cfg.rew_carry_dist * dist_box_goal * grasped_f

        # [3.5단계] Near-place one-time bonus: box가 0.22m 이내 첫 진입 시
        newly_near = self._grasped & (dist_box_goal < 0.22) & (~self._near_place_awarded)
        self._near_place_awarded |= newly_near
        rew_near_place = self.cfg.rew_near_place * newly_near.float()

        # [4단계] 실제 물리 착지 (Phase 2 transport 방식)
        settled   = self._has_released & (self._steps_after_rel >= self.cfg.settle_steps)
        placed    = ~self._grasped & settled & (dist_box_goal < self.cfg.place_dist_threshold)
        rew_place = self.cfg.rew_place * placed.float()

        # [5단계] goal 근처 release one-time bonus
        near_xy_rel = (ee_pos[:, :2] - self._goal_pos_w[:, :2]).norm(dim=1)
        rew_release_near = self.cfg.rew_release_near * (
            self._newly_released & (near_xy_rel < self.cfg.near_release_dist)
        ).float()

        # [6단계] grip penalty: goal 근처서 계속 쥐고 있으면 soft 패널티
        near_xy = (ee_pos[:, :2] - self._goal_pos_w[:, :2]).norm(dim=1)
        grip_pen_w = (1.0 - near_xy / self.cfg.near_release_dist).clamp(0.0, 1.0)
        rew_grip_pen = -self.cfg.rew_grip_penalty * grip_pen_w * grasped_f

        log = self.extras.setdefault("log", {})
        log["dist_ee_box"]     = dist_ee_box.mean().item()
        log["grasp_rate"]      = grasped_f.mean().item() * 100.0
        log["dist_box_goal"]   = (dist_box_goal * grasped_f).sum().item() / (grasped_f.sum().item() + 1e-6)
        log["transport_delta"] = (self._prev_dist_box_goal - dist_box_goal)[self._grasped].mean().item() if self._grasped.any() else 0.0
        log["release_rate"]    = self._newly_released.float().mean().item() * 100.0
        log["force_rel_rate"]  = self._force_released.float().mean().item() * 100.0
        # force_grasp 제외한 자연 grasp rate (커리큘럼 기준 지표)
        natural_mask = ~self._force_grasped_mask
        if natural_mask.any():
            log["natural_grasp_rate"] = self._grasped[natural_mask].float().mean().item() * 100.0

        self._prev_dist_box_goal = dist_box_goal.detach().clone()

        # approach 단계 -0.5/step, transport(grasped) 단계 -0.1/step
        rew_time_actual = self.cfg.rew_time * not_grasped + self.cfg.rew_time_grasped * grasped_f
        return (rew_approach + rew_grasp + rew_transport + rew_near_place
                + rew_place + rew_release_near + rew_grip_pen + rew_time_actual)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos, _ = self._get_ee_pose()
        box_pos = self.box.data.root_pos_w
        box_pos_eff = torch.where(
            self._grasped.unsqueeze(1).expand(-1, 3),
            ee_pos + self._grasp_ee_offset,
            box_pos,
        )
        dist_box_goal = (box_pos_eff - self._goal_pos_w).norm(dim=1)

        # 실제 물리 착지 (Phase 2 transport 방식)
        settled  = self._has_released & (self._steps_after_rel >= self.cfg.settle_steps)
        placed   = ~self._grasped & settled & (dist_box_goal < self.cfg.place_dist_threshold)
        fell_off = ~self._grasped & self._has_released & (box_pos[:, 2] < 0.9)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1

        done = placed | fell_off | timed_out
        self._stat_placed   += (placed & done).sum().item()
        self._stat_episodes += done.sum().item()
        if self._stat_episodes >= self._stat_window:
            self._stat_placed   = 0
            self._stat_episodes = 0

        log = self.extras.setdefault("log", {})
        log["term_placed"]   = placed.float().mean().item() * 100.0
        log["term_fell_off"] = fell_off.float().mean().item() * 100.0
        if self._stat_episodes > 0:
            log["place_rate"] = self._stat_placed / self._stat_episodes * 100.0

        return placed | fell_off, timed_out

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
        # env origin을 기준으로 local 좌표계에서 클램핑 후 world 좌표로 변환
        # (world = ee_pos_n이므로, local = world - env_origin)
        env_orig = self.scene.env_origins[env_ids_t]  # (n, 3)
        local_ee_x = ee_pos_n[:, 0] - env_orig[:, 0]
        local_ee_y = ee_pos_n[:, 1] - env_orig[:, 1]
        local_box_x = (local_ee_x + d * torch.cos(theta)).clamp(0.55, 1.45)
        local_box_y = (local_ee_y + d * torch.sin(theta)).clamp(-0.30, 0.30)
        box_state[:, 0] = env_orig[:, 0] + local_box_x
        box_state[:, 1] = env_orig[:, 1] + local_box_y
        box_state[:, 2] = 1.15
        box_state[:, 7:13] = 0.0
        self.box.write_root_state_to_sim(box_state, env_ids_t)

        # goal: box에서 goal_spawn_dist 거리 (transport env 동일 방식)
        theta = sample_uniform(0.0, 6.2832, (n,), device=self.device)
        r     = sample_uniform(self.cfg.goal_spawn_dist * 0.7, self.cfg.goal_spawn_dist * 1.3, (n,), device=self.device)
        local_goal_x = (local_box_x + r * torch.cos(theta)).clamp(0.55, 1.45)
        local_goal_y = (local_box_y + r * torch.sin(theta)).clamp(-0.30, 0.30)
        self._goal_pos_w[env_ids_t, 0] = env_orig[:, 0] + local_goal_x
        self._goal_pos_w[env_ids_t, 1] = env_orig[:, 1] + local_goal_y
        self._goal_pos_w[env_ids_t, 2] = 1.15

        self._grasped[env_ids_t]             = False
        self._force_grasped_mask[env_ids_t]  = False
        self._force_grasp_pending[env_ids_t] = False
        self._frozen_box_state[env_ids_t]    = 0.0
        self._grasp_ee_offset[env_ids_t]     = 0.0
        self._prev_dist_ee_box[env_ids_t]    = 999.0
        self._prev_dist_box_goal[env_ids_t]  = (box_state[:, :3] - self._goal_pos_w[env_ids_t]).norm(dim=1)
        self._near_place_awarded[env_ids_t]  = False
        self._has_released[env_ids_t]        = False
        self._steps_after_rel[env_ids_t]     = 0
        self._newly_released[env_ids_t]      = False
        self._force_released[env_ids_t]      = False
        # cmd_ee_pos를 현재 EE 위치로 초기화 (transport env 방식)
        self._cmd_ee_pos[env_ids_t]          = ee_pos_n
        self._home_q[env_ids_t]              = home_pose[:, :7]

        # force_grasp: reset에서는 pending 표시만 → 실제 EE 기반 활성화는 step 1에서 (_get_rewards)
        if self.cfg.force_grasp_on_reset and n > 0:
            half     = max(1, int(n * self.cfg.force_grasp_fraction))
            perm     = torch.randperm(n, device=self.device)
            fg_ids   = env_ids_t[perm[:half]]
            self._force_grasp_pending[fg_ids] = True

    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_pos, ee_quat
