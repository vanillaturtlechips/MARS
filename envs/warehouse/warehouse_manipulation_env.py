"""Phase 2 — 창고 Pick & Place 환경.

로봇: Franka Panda (7-DOF 암 + 평행 그리퍼)
임무: 박스를 집어 지정 선반 위치에 내려놓기

Teacher 관측 (특권 정보, 훈련 전용):
  box_rel(3) + box_quat(4) + box_mass(1) + gripper(1) +
  goal_rel(3) + jpos(9) + jvel(9) = 30차원

Student 관측 (배포용, 실제 센서):
  ee_pos(3) + gripper(1) + goal_rel(3) + jpos(9) + jvel(9) = 25차원
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

try:
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
except ImportError:
    from isaaclab_assets import FRANKA_PANDA_CFG  # type: ignore

# 수직 transport 태스크: box spawn z=0.53, goal z=0.75-0.90
# spawn → goal z차이 ≥ 0.22m > place_dist(0.17m) → free win 구조적 불가
# 수평 이동 없이 위아래만 → y workspace 한계 무관, table 충돌 없음
PLACE_GOALS = [
    (0.28, 0.00, 0.75),
    (0.32, 0.00, 0.75),
    (0.28, 0.00, 0.85),
    (0.32, 0.00, 0.85),
]

TEACHER_OBS_DIM = 30
STUDENT_OBS_DIM = 25


@configclass
class WarehouseManipulationEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0   # 15→5s: timeout 비용 증가 + random walk 억제
    action_space = 4             # [dx, dy, dz, gripper]
    observation_space = TEACHER_OBS_DIM
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
    rew_transport:  float = 10.0   # delta_dist × 100 × grasped
    rew_align:      float =  0.0   # exploit 원인 — 비활성화
    rew_goal_dist:  float =  5.0   # -dist_box_goal × grasped (dense shaping)
    rew_place:      float = 100.0  # terminal bonus
    rew_drop:       float =   0.0
    rew_time:       float = -0.50

    box_size_range: tuple[float, float] = (0.04, 0.08)
    box_mass_range: tuple[float, float] = (0.3, 2.0)

    grasp_dist_threshold: float = 0.11
    place_dist_threshold: float = 0.13  # 0.17→0.13: random walk 진입 방지

    student_mode: bool = False


@configclass
class WarehouseManipulationStudentEnvCfg(WarehouseManipulationEnvCfg):
    observation_space = STUDENT_OBS_DIM
    student_mode = True


class WarehouseManipulationEnv(DirectRLEnv):
    cfg: WarehouseManipulationEnvCfg

    def __init__(self, cfg: WarehouseManipulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n = self.num_envs
        d = self.device

        body_names = list(self.robot.data.body_names)
        self._ee_body_idx  = body_names.index("panda_hand")
        self._jac_body_idx = self._ee_body_idx - 1

        self._goal_pos_w         = torch.zeros(n, 3, device=d)
        self._box_mass           = torch.ones(n, device=d)
        self._grasped            = torch.zeros(n, dtype=torch.bool, device=d)
        self._actions            = torch.zeros(n, 4, device=d)
        self._prev_dist_box_goal = torch.full((n,), 999.0, device=d)
        self._frozen_box_state   = torch.zeros(n, 13, device=d)
        self._grasp_ee_offset    = torch.zeros(n, 3, device=d)

        self._stat_placed   = 0
        self._stat_episodes = 0

    def _setup_scene(self):
        franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        for actuator in franka_cfg.actuators.values():
            if hasattr(actuator, 'stiffness') and actuator.stiffness < 400:
                actuator.stiffness = 400.0
                actuator.damping   = 40.0
        self.robot = Articulation(franka_cfg)

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        n = self.num_envs
        delta_pos = self._actions[:, :3] * 0.03

        jac = self.robot.root_physx_view.get_jacobians()
        J   = jac[:, self._jac_body_idx, :3, :7]
        lam = 0.01
        JT      = J.transpose(-2, -1)
        JJT_reg = torch.bmm(J, JT) + lam * torch.eye(3, device=self.device).unsqueeze(0).expand(n, -1, -1)
        J_dls   = torch.bmm(JT, torch.linalg.inv(JJT_reg))
        delta_q = torch.bmm(J_dls, delta_pos.unsqueeze(-1)).squeeze(-1)

        joint_target = self.robot.data.joint_pos[:, :7] + delta_q
        self.robot.set_joint_position_target(joint_target, joint_ids=list(range(7)))

        gripper_pos = ((self._actions[:, 3:4] + 1.0) / 2.0) * 0.04
        self.robot.set_joint_position_target(
            gripper_pos.expand(-1, 2).clone(), joint_ids=[7, 8]
        )

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

        if self.cfg.student_mode:
            obs = torch.cat([
                ee_pos,
                gripper_w,
                goal_rel,
                joint_pos[:, :9],
                joint_vel[:, :9],
            ], dim=1)   # (N, 25)
        else:
            obs = torch.cat([
                box_pos - ee_pos,
                box_quat,
                self._box_mass.unsqueeze(1),
                gripper_w,
                goal_rel,
                joint_pos[:, :9],
                joint_vel[:, :9],
            ], dim=1)   # (N, 30)

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

        dropped = self._grasped & (box_pos[:, 2] < 0.30)
        placed  = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)

        not_grasped = (~self._grasped).float()
        grasped_f   = self._grasped.float()

        approach   = -self.cfg.rew_approach  * dist_ee_box  * not_grasped
        goal_dense = -self.cfg.rew_goal_dist * dist_box_goal * grasped_f

        # delta tracking: use sim box pos (not ee_pos proxy before grasp)
        # after grasp, box.data.root_pos_w == box_pos_carried (written in _apply_action)
        dist_box_real = (box_pos - self._goal_pos_w).norm(dim=1)
        delta_goal = (self._prev_dist_box_goal - dist_box_real).clamp(-0.1, 0.1)
        transport  = self.cfg.rew_transport * delta_goal * 100.0 * grasped_f

        goal_dir  = (self._goal_pos_w - box_pos_carried).clone()
        goal_norm = goal_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        act_dir   = self._actions[:, :3]
        act_norm  = act_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        cos_sim   = (act_dir / act_norm * (goal_dir / goal_norm)).sum(dim=1)
        alignment = self.cfg.rew_align * cos_sim * grasped_f

        self._prev_dist_box_goal = dist_box_real.detach()

        log = self.extras.setdefault("log", {})
        log["dist_ee_box"]   = dist_ee_box.mean().item()
        log["grasp_rate"]    = grasped_f.mean().item() * 100.0
        log["dist_box_goal"] = (dist_box_goal * grasped_f).sum().item() / (grasped_f.sum().item() + 1e-6)

        return (
            approach
            + goal_dense
            + transport
            + alignment
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

        box_state = self.box.data.default_root_state[env_ids_t].clone()
        box_state[:, 0] = self.scene.env_origins[env_ids_t, 0] + sample_uniform(0.25, 0.38, (n,), device=self.device)
        box_state[:, 1] = self.scene.env_origins[env_ids_t, 1] + sample_uniform(-0.05, 0.05, (n,), device=self.device)
        box_state[:, 2] = self.scene.env_origins[env_ids_t, 2] + 0.53
        self.box.write_root_state_to_sim(box_state, env_ids_t)

        self._box_mass[env_ids_t] = sample_uniform(
            self.cfg.box_mass_range[0], self.cfg.box_mass_range[1], (n,), device=self.device
        )

        goal_idx = torch.randint(len(PLACE_GOALS), (n,), device=self.device)
        goals    = torch.tensor(PLACE_GOALS, device=self.device)[goal_idx]
        self._goal_pos_w[env_ids_t] = goals + self.scene.env_origins[env_ids_t]

        self._grasped[env_ids_t]            = False
        self._frozen_box_state[env_ids_t]   = 0.0
        self._grasp_ee_offset[env_ids_t]    = 0.0
        # initialize with actual box-to-goal distance (no jump at first grasp)
        self._prev_dist_box_goal[env_ids_t] = (
            box_state[:, :3] - self._goal_pos_w[env_ids_t]
        ).norm(dim=1)

    def _get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_pos, ee_quat
