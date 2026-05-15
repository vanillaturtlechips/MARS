"""100-iter quick test — 보상 상승 여부 조기 확인용."""

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

from envs.warehouse.agents.rsl_rl_ppo_cfg import WarehouseNavPPORunnerCfg
from envs.warehouse.warehouse_env import WarehouseNavEnvCfg

ENV_ID = "Isaac-WarehouseNav-Direct-v0"
NUM_ENVS = 256    # 적게 써서 빠르게 실행
DEVICE = "cuda"
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "warehouse_nav_test")

env_cfg = WarehouseNavEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env = gym.make(ENV_ID, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

runner_cfg = WarehouseNavPPORunnerCfg()
runner_cfg.device = DEVICE
runner_cfg.max_iterations = 100
runner_cfg_dict = class_to_dict(runner_cfg)

runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=LOG_DIR, device=DEVICE)

sys.stderr.write("[quick-test] 100 iter 조기 진단 시작...\n")
sys.stderr.write("[quick-test] 초반 30 iter 안에 보상 상승세 있어야 정상\n")
sys.stderr.flush()

runner.learn(num_learning_iterations=100, init_at_random_ep_len=True)

sys.stderr.write("[quick-test] 완료.\n")
sys.stderr.flush()

env.close()
simulation_app.close()
