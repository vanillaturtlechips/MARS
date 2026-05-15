"""Phase 1 — Single robot warehouse navigation (PPO, Direct RL)."""

import sys
import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym

# Register our custom env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs.warehouse  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import class_to_dict

from envs.warehouse.agents.rsl_rl_ppo_cfg import WarehouseNavPPORunnerCfg
from envs.warehouse.warehouse_env import WarehouseNavEnvCfg

ENV_ID = "Isaac-WarehouseNav-Direct-v0"
NUM_ENVS = 1024
DEVICE = "cuda"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "warehouse_nav")

sys.stderr.write(f"[train] {ENV_ID} x {NUM_ENVS} envs | PPO | {DEVICE}\n")
sys.stderr.flush()

# -- env --
env_cfg = WarehouseNavEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env = gym.make(ENV_ID, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

# -- runner --
runner_cfg = WarehouseNavPPORunnerCfg()
runner_cfg.device = DEVICE
runner_cfg_dict = class_to_dict(runner_cfg)

runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=LOG_DIR, device=DEVICE)

sys.stderr.write("[train] Starting Phase 1 training...\n")
sys.stderr.write(f"[train] Max iterations: {runner_cfg.max_iterations}\n")
sys.stderr.flush()

runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

sys.stderr.write(f"[train] Done. Logs: {LOG_DIR}\n")
sys.stderr.flush()

env.close()
simulation_app.close()
