"""Phase 2 — Pick & Place 훈련 (카메라 DR 직접 훈련).

실행:
  python training/single_robot/train_manipulation.py --headless --num_envs 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 Pick & Place 훈련")
parser.add_argument("--num_envs",      type=int,   default=256)
parser.add_argument("--max_iter",      type=int,   default=3000)
parser.add_argument("--resume_ckpt",   type=str,   default=None)
parser.add_argument("--lr",            type=float, default=1e-3,  help="PPO learning rate")
parser.add_argument("--save_interval", type=int,   default=300,   help="체크포인트 저장 주기")
# 하위 호환 — 무시됨
parser.add_argument("--student",       action="store_true", default=False)
parser.add_argument("--teacher_ckpt",  type=str,   default=None)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv,
    WarehouseManipulationEnvCfg,
    WarehouseManipulationStudentEnvCfg,
    TEACHER_OBS_DIM,
    STUDENT_OBS_DIM,
)


def make_runner_cfg(obs_dim: int, mode: str, max_iter: int) -> RslRlOnPolicyRunnerCfg:
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env  = 128
    runner_cfg.max_iterations     = max_iter
    runner_cfg.save_interval      = args.save_interval
    runner_cfg.experiment_name    = f"warehouse_manipulation_{mode}"
    runner_cfg.logger             = "tensorboard"
    # obs 정규화 활성화: transport_dst=-3*dist는 grasp phase 내내 음수이므로
    # VF가 큰 음수 return을 예측해야 함 → 정규화로 스케일 안정화
    runner_cfg.empirical_normalization = True

    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    return runner_cfg


def main():
    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = WarehouseManipulationEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = make_runner_cfg(TEACHER_OBS_DIM, "manipulation", args.max_iter)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"
    # entropy_coef 증가: transport phase에서 탐색 유지 (0.001 → 0.01)
    # 초기 surrogate_loss≈0은 entropy 부족으로 정책이 조기 수렴한 것
    # 0.01이면 탐색을 유지하면서도 place reward로 수렴 가능
    cfg_dict["algorithm"]["entropy_coef"] = 0.01
    cfg_dict["algorithm"]["learning_rate"] = args.lr
    runner = OnPolicyRunner(env, cfg_dict, log_dir="logs/warehouse_manipulation", device=env.device)

    if args.resume_ckpt:
        print(f"[Resume] {args.resume_ckpt}")
        runner.load(args.resume_ckpt)

    print(f"\n[MANIPULATION] obs_dim={TEACHER_OBS_DIM}, {args.num_envs} envs, {args.max_iter} iter")
    print(f"카메라 DR: σ ∈ [{env_cfg.camera_noise_min*100:.1f}cm, {env_cfg.camera_noise_max*100:.1f}cm] per-episode\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
