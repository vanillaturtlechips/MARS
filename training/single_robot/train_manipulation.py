"""Phase 2 — Teacher PPO 훈련.

실행:
  # Teacher 훈련 (특권 정보 사용)
  python training/single_robot/train_manipulation.py

  # Student 훈련 (Teacher 체크포인트에서)
  python training/single_robot/train_manipulation.py \
    --student --teacher_ckpt logs/warehouse_manipulation_teacher/model_XXX.pt

  # 헤드리스
  python training/single_robot/train_manipulation.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 Pick & Place 훈련")
parser.add_argument("--num_envs",     type=int,   default=256)
parser.add_argument("--max_iter",     type=int,   default=3000)
parser.add_argument("--student",      action="store_true", default=False)
parser.add_argument("--teacher_ckpt", type=str,   default=None)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv,
    WarehouseManipulationEnvCfg,
    WarehouseManipulationStudentEnvCfg,
    TEACHER_OBS_DIM,
    STUDENT_OBS_DIM,
)
from envs.warehouse.agents.rsl_rl_ppo_cfg import RslRlPpoActorCriticCfg, RslRlOnPolicyRunnerCfg


def make_runner_cfg(obs_dim: int, mode: str, max_iter: int) -> RslRlOnPolicyRunnerCfg:
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env  = 32
    runner_cfg.max_iterations     = max_iter
    runner_cfg.save_interval      = 300
    runner_cfg.experiment_name    = f"warehouse_manipulation_{mode}"
    runner_cfg.logger             = "tensorboard"
    runner_cfg.empirical_normalization = True

    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0 if mode == "teacher" else 0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    return runner_cfg


def main():
    if args.student:
        env_cfg  = WarehouseManipulationStudentEnvCfg()
        mode     = "student"
        log_dir  = "logs/warehouse_manipulation_student"
        obs_dim  = STUDENT_OBS_DIM
    else:
        env_cfg  = WarehouseManipulationEnvCfg()
        mode     = "teacher"
        log_dir  = "logs/warehouse_manipulation_teacher"
        obs_dim  = TEACHER_OBS_DIM

    env_cfg.scene.num_envs = args.num_envs
    env = WarehouseManipulationEnv(env_cfg)

    runner_cfg = make_runner_cfg(obs_dim, mode, args.max_iter)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=env.device)

    if args.student and args.teacher_ckpt:
        print(f"[Student] Teacher 체크포인트에서 Actor 초기화: {args.teacher_ckpt}")
        runner.load(args.teacher_ckpt)

    print(f"\n[{mode.upper()}] obs_dim={obs_dim}, {args.num_envs} envs, {args.max_iter} iter")
    print(f"조기 진단: 50 iter 안에 rew_grasp 상승 확인")
    print(f"없으면 grasp_dist_threshold 또는 박스 초기 위치 범위 조정\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
