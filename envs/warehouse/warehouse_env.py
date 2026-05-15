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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, sample_uniform


@configclass
class WarehouseNavEnvCfg(DirectRLEnvCfg):
    decimation = 4              # 60 Hz sim / 4 = 15 Hz policy
    episode_length_s = 20.0
    action_space = 3            # [cmd_vx, cmd_vy, cmd_omega] in body frame, [-1, 1]
    observation_space = 6       # [goal_x_body, goal_y_body, goal_dist, vx_body, vy_body, omega_z]
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    robot_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.8), metallic=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.15)),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=12.0, replicate_physics=True
    )

    # velocity limits
    max_vx: float = 1.5     # m/s
    max_vy: float = 1.0     # m/s
    max_omega: float = 2.0  # rad/s

    # goal
    goal_radius: float = 0.35   # success radius [m]
    goal_range: float = 4.0     # max goal distance [m]
    goal_min_dist: float = 1.0  # min goal distance [m]

    # reward
    rew_dist: float = -0.3
    rew_goal: float = 10.0
    rew_time: float = -0.001


class WarehouseNavEnv(DirectRLEnv):
    cfg: WarehouseNavEnvCfg

    def __init__(self, cfg: WarehouseNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)

    def _setup_scene(self):
        self.robot = RigidObject(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        quat = self.robot.data.root_quat_w  # (N, 4) wxyz
        _, _, yaw = euler_xyz_from_quat(quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        vx_b = self._actions[:, 0] * self.cfg.max_vx
        vy_b = self._actions[:, 1] * self.cfg.max_vy
        omega = self._actions[:, 2] * self.cfg.max_omega

        vel_cmd = torch.zeros(self.num_envs, 6, device=self.device)
        vel_cmd[:, 0] = cos_yaw * vx_b - sin_yaw * vy_b   # world vx
        vel_cmd[:, 1] = sin_yaw * vx_b + cos_yaw * vy_b   # world vy
        vel_cmd[:, 5] = omega                               # yaw rate (world z)

        self.robot.write_root_velocity_to_sim(vel_cmd)

    def _get_observations(self) -> dict:
        pos_w = self.robot.data.root_pos_w[:, :2]
        quat = self.robot.data.root_quat_w
        lin_vel_w = self.robot.data.root_lin_vel_w
        ang_vel_w = self.robot.data.root_ang_vel_w

        goal_vec_w = self._goal_pos_w - pos_w
        goal_dist = goal_vec_w.norm(dim=1, keepdim=True).clamp(max=10.0)

        goal_3d = torch.cat([goal_vec_w, torch.zeros(self.num_envs, 1, device=self.device)], dim=1)
        goal_body = quat_apply_inverse(quat, goal_3d)[:, :2]

        vel_body = quat_apply_inverse(quat, lin_vel_w)[:, :2]
        omega_z = ang_vel_w[:, 2:3]

        obs = torch.cat([goal_body, goal_dist, vel_body, omega_z], dim=1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos_w = self.robot.data.root_pos_w[:, :2]
        dist = (pos_w - self._goal_pos_w).norm(dim=1)
        goal_reached = (dist < self.cfg.goal_radius).float()
        return self.cfg.rew_dist * dist + self.cfg.rew_goal * goal_reached + self.cfg.rew_time

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w = self.robot.data.root_pos_w[:, :2]
        dist = (pos_w - self._goal_pos_w).norm(dim=1)
        goal_reached = dist < self.cfg.goal_radius
        local_pos = pos_w - self.scene.env_origins[:, :2]  # env 로컬 좌표로 변환
        out_of_bounds = local_pos.abs().max(dim=1).values > 8.0
        terminated = goal_reached | out_of_bounds
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Small random spawn offset and random yaw
        default_root_state[:, :2] += sample_uniform(-0.5, 0.5, (n, 2), device=self.device)
        yaw = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
        zeros = torch.zeros(n, device=self.device)
        default_root_state[:, 3] = torch.cos(yaw / 2)   # quat w
        default_root_state[:, 4] = zeros                 # quat x
        default_root_state[:, 5] = zeros                 # quat y
        default_root_state[:, 6] = torch.sin(yaw / 2)   # quat z
        default_root_state[:, 7:] = 0.0                 # zero velocity

        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        # Random goal position (polar coords from env origin)
        angle = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
        dist = sample_uniform(self.cfg.goal_min_dist, self.cfg.goal_range, (n,), device=self.device)
        self._goal_pos_w[env_ids] = self.scene.env_origins[env_ids, :2] + torch.stack(
            [dist * torch.cos(angle), dist * torch.sin(angle)], dim=1
        )
