"""Phase 1.5 — 창고 맵 (선반 장애물) 처음부터 훈련."""

import sys
import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs.warehouse  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import class_to_dict

from envs.warehouse.agents.rsl_rl_ppo_cfg import WarehouseObstacleNavPPORunnerCfg
from envs.warehouse.warehouse_obstacle_env import WarehouseObstacleNavEnvCfg

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "warehouse_obstacle_nav")
ENV_ID   = "Isaac-WarehouseObstacleNav-Direct-v0"
NUM_ENVS = 1024
DEVICE   = "cuda"

sys.stderr.write(f"[train] {ENV_ID} x {NUM_ENVS} | 처음부터 훈련 (adaptive lr, desired_kl=0.05, 2000 iter)\n")
sys.stderr.flush()

env_cfg = WarehouseObstacleNavEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env = gym.make(ENV_ID, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

runner_cfg = WarehouseObstacleNavPPORunnerCfg()
runner_cfg.device = DEVICE
runner_cfg.max_iterations = 2000
runner_cfg_dict = class_to_dict(runner_cfg)

runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=LOG_DIR, device=DEVICE)

runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

sys.stderr.write(f"[train] Done. Logs: {LOG_DIR}\n")
sys.stderr.flush()

env.close()
simulation_app.close()
