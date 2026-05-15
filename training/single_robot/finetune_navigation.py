"""Phase 1.5 — 창고 맵 fine-tuning (model_999.pt → obstacle 환경)."""

import sys
import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs.warehouse  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import class_to_dict

from envs.warehouse.agents.rsl_rl_ppo_cfg import WarehouseNavPPORunnerCfg
from envs.warehouse.warehouse_obstacle_env import WarehouseObstacleNavEnvCfg

PRETRAINED = os.path.join(
    os.path.dirname(__file__), "..", "..", "logs", "warehouse_nav", "model_999.pt"
)
LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "warehouse_obstacle_nav")
ENV_ID   = "Isaac-WarehouseObstacleNav-Direct-v0"
NUM_ENVS = 1024
DEVICE   = "cuda"

sys.stderr.write(f"[finetune] {ENV_ID} x {NUM_ENVS} | fine-tuning from model_999.pt\n")
sys.stderr.flush()

# -- 환경 --
env_cfg = WarehouseObstacleNavEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env = gym.make(ENV_ID, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

# -- runner --
runner_cfg = WarehouseNavPPORunnerCfg()
runner_cfg.device = DEVICE
runner_cfg.max_iterations = 500
runner_cfg.algorithm.learning_rate = 1e-4   # fine-tuning: 낮은 lr
runner_cfg_dict = class_to_dict(runner_cfg)

runner = OnPolicyRunner(env, runner_cfg_dict, log_dir=LOG_DIR, device=DEVICE)

# -- 사전학습 가중치 로드 --
# obs_dim 6→7로 늘어났으므로 첫 번째 레이어(actor.0, critic.0)는
# 기존 6열 복사 + 새 7번째 열(장애물 거리)은 0으로 초기화
ckpt = torch.load(PRETRAINED, map_location=DEVICE)
old_sd = ckpt["model_state_dict"]
new_sd = runner.alg.policy.state_dict()

for key in new_sd:
    if key not in old_sd:
        continue
    old_w = old_sd[key]
    new_w = new_sd[key]
    if old_w.shape == new_w.shape:
        new_sd[key] = old_w                               # 그대로 복사
    elif key in ("actor.0.weight", "critic.0.weight"):
        # (256, 6) → (256, 7): 7번째 열을 0으로 패딩
        padded = torch.zeros_like(new_w)
        padded[:, :old_w.shape[1]] = old_w
        new_sd[key] = padded
        sys.stderr.write(f"[finetune] {key}: padded {old_w.shape} → {new_w.shape}\n")

runner.alg.policy.load_state_dict(new_sd)
sys.stderr.write(f"[finetune] Pre-trained weights loaded (iter={ckpt['iter']})\n")
sys.stderr.flush()

runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)

sys.stderr.write(f"[finetune] Done. Logs: {LOG_DIR}\n")
sys.stderr.flush()

env.close()
simulation_app.close()
