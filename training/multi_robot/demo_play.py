"""데모 플레이어 — 학습된 IPPO/MAPPO 정책을 시각화.

훈련 환경(cuboid)과 동일한 물리 구조를 유지하면서
iw_hub 로봇 + 창고 배경으로 시각만 교체.

실행:
  python training/multi_robot/demo_play.py \
    --checkpoint logs/warehouse_ippo/model_400.pt \
    --livestream 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MARS 데모 플레이어")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs",   type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, sample_uniform
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_marl_env import (
    WarehouseMARLEnv, WarehouseMARLEnvCfg,
    N_ROBOTS, OBS_PER_ROBOT, SPAWN_OFFSETS, ROBOT_COLLISION_DIST,
)
from envs.warehouse.warehouse_obstacle_env import SHELF_CENTERS, SHELF_HALF, _shelf_aabb_dist, _goal_in_shelf
from envs.warehouse.ippo_wrapper import IPPOReshapeWrapper


IW_HUB_USD = f"{ISAAC_NUCLEUS_DIR}/Isaac/Robots/Idealworks/iw_hub/iw_hub.usd"
WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"


class WarehouseDemoEnvCfg(WarehouseMARLEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=14.0, replicate_physics=True
    )


class WarehouseDemoEnv(WarehouseMARLEnv):
    """시각화 전용 환경 — 물리는 동일, 시각 에셋만 교체."""

    def _setup_scene(self):
        # ── 로봇: iw_hub USD (물리 cuboid 대신) ──────────────────────
        self.robots: list[RigidObject] = []
        robot_colors = [
            (0.2, 0.4, 0.8),
            (0.8, 0.3, 0.2),
            (0.2, 0.7, 0.3),
        ]
        for i in range(N_ROBOTS):
            try:
                spawn_cfg = sim_utils.UsdFileCfg(
                    usd_path=IW_HUB_USD,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        linear_damping=2.0,
                        angular_damping=5.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )
                print(f"[Demo] iw_hub USD 로드 성공 — Robot_{i}")
            except Exception:
                # iw_hub 없으면 cuboid fallback
                spawn_cfg = sim_utils.CuboidCfg(
                    size=(0.5, 0.4, 0.3),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        linear_damping=2.0,
                        angular_damping=5.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=robot_colors[i], metallic=0.3
                    ),
                )
                print(f"[Demo] iw_hub 없음, Cuboid fallback — Robot_{i}")

            robot_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Robot_{i}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(SPAWN_OFFSETS[i][0], SPAWN_OFFSETS[i][1], 0.15)
                ),
            )
            self.robots.append(RigidObject(robot_cfg))

        # ── 바닥: 창고 USD 시도, 실패 시 GroundPlane ─────────────────
        try:
            warehouse_cfg = sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD)
            warehouse_cfg.func("/World/Warehouse", warehouse_cfg,
                               translation=(0.0, 0.0, 0.0),
                               orientation=(1.0, 0.0, 0.0, 0.0))
            print("[Demo] 창고 USD 로드 성공")
        except Exception:
            spawn_ground_plane("/World/ground", GroundPlaneCfg())
            print("[Demo] 창고 USD 없음, GroundPlane fallback")

        # ── 선반: cuboid (물리 충돌 유지) ────────────────────────────
        shelf_cfg_base = sim_utils.CuboidCfg(
            size=(3.0, 0.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.38, 0.18), metallic=0.0
            ),
        )
        for s_i, (cx, cy, cz) in enumerate(SHELF_CENTERS):
            shelf_cfg_base.func(
                f"/World/envs/env_0/Shelf_{s_i}",
                shelf_cfg_base,
                translation=(cx, cy, cz),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, robot in enumerate(self.robots):
            self.scene.rigid_objects[f"robot_{i}"] = robot

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 0.98, 0.95))
        light_cfg.func("/World/Light", light_cfg)


def main():
    env_cfg = WarehouseDemoEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = WarehouseDemoEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)
    env = IPPOReshapeWrapper(env, N_ROBOTS, OBS_PER_ROBOT)

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env = 24
    runner_cfg.max_iterations = 999999
    runner_cfg.experiment_name = "warehouse_demo"
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.01,  # 데모: 노이즈 최소화 (결정론적 행동)
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"
    cfg_dict["algorithm"]["entropy_coef"] = 0.0

    runner = OnPolicyRunner(env, cfg_dict, log_dir="/tmp/demo", device=env.device)
    runner.load(args.checkpoint)

    print(f"\n[Demo] 체크포인트 로드: {args.checkpoint}")
    print(f"[Demo] 로봇 {N_ROBOTS}대, noise_std=0.01 (결정론적)\n")

    runner.learn(num_learning_iterations=999999, init_at_random_ep_len=False)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
