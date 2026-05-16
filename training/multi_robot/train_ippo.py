"""Phase 3 — IPPO 베이스라인 훈련.

각 로봇을 독립 PPO로 학습. Parameter Sharing으로 VRAM 절감.
수렴 확인 후 train_marl.py (MAPPO)와 비교.

실행:
  python training/multi_robot/train_ippo.py
  python training/multi_robot/train_ippo.py --headless  # 헤드리스
  python training/multi_robot/train_ippo.py --num_envs 64  # VRAM 부족 시
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3 IPPO 베이스라인 훈련")
parser.add_argument("--num_envs",  type=int,   default=128)
parser.add_argument("--max_iter",  type=int,   default=3000)
parser.add_argument("--checkpoint", type=str, default=None, help="이어서 훈련할 체크포인트")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_marl_env import WarehouseMARLEnv, WarehouseMARLEnvCfg, N_ROBOTS

# IPPO: obs_dim = 로봇 1대 관측, act_dim = 로봇 1대 행동
# Parameter Sharing: N 로봇이 동일 네트워크 공유
OBS_DIM = 7 + (N_ROBOTS - 1)   # 9
ACT_DIM = 3


def make_ippo_runner_cfg(num_envs: int, max_iter: int) -> RslRlOnPolicyRunnerCfg:
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env  = 24
    runner_cfg.max_iterations     = max_iter
    runner_cfg.save_interval      = 200
    runner_cfg.experiment_name    = "warehouse_ippo"
    runner_cfg.run_name           = f"ippo_n{N_ROBOTS}_env{num_envs}"
    runner_cfg.logger             = "tensorboard"
    runner_cfg.empirical_normalization = True

    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    return runner_cfg


def main():
    # 환경 설정
    env_cfg = WarehouseMARLEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    # IPPO: 각 로봇 독립 처리 → obs/act를 로봇 1대 기준으로 분할
    env_cfg.observation_space = OBS_DIM
    env_cfg.action_space      = ACT_DIM

    env = WarehouseMARLEnv(env_cfg)

    runner_cfg = make_ippo_runner_cfg(args.num_envs, args.max_iter)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir="logs/warehouse_ippo", device=env.device)

    if args.checkpoint:
        runner.load(args.checkpoint)

    print(f"\n[IPPO] 로봇 {N_ROBOTS}대, {args.num_envs} envs, {args.max_iter} iter")
    print(f"[IPPO] 조기 진단: 30 iter 안에 mean_reward 상승세 확인")
    print(f"[IPPO] 없으면 potential_reward.py 파라미터 재검토\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
