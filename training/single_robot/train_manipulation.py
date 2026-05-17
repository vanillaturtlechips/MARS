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
parser.add_argument("--resume_ckpt",  type=str,   default=None, help="이어서 훈련할 체크포인트 경로")
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
    env = RslRlVecEnvWrapper(env)

    runner_cfg = make_runner_cfg(obs_dim, mode, args.max_iter)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"   # rsl_rl 3.x 필수
    cfg_dict["algorithm"]["entropy_coef"] = 0.001  # noise_std 발산 억제
    runner = OnPolicyRunner(env, cfg_dict, log_dir=log_dir, device=env.device)

    if args.resume_ckpt:
        print(f"[Resume] 체크포인트에서 이어서 훈련: {args.resume_ckpt}")
        runner.load(args.resume_ckpt)
        # runner.current_learning_iteration 이 로드된 iter로 설정됨
        # runner.learn()은 max_iter가 아니라 num_learning_iterations만큼 추가 진행
    elif args.student and args.teacher_ckpt:
        print(f"[Student] Teacher 체크포인트에서 hidden layer 초기화: {args.teacher_ckpt}")
        ckpt       = torch.load(args.teacher_ckpt, map_location=runner.device, weights_only=False)
        teacher_sd = ckpt["model_state_dict"]
        student_sd = runner.alg.policy.state_dict()
        # shape 일치하는 레이어만 로드 (입력 레이어 Teacher 33차원 vs Student 25차원 제외)
        filtered = {k: v for k, v in teacher_sd.items()
                    if k in student_sd and v.shape == student_sd[k].shape}
        runner.alg.policy.load_state_dict(filtered, strict=False)
        print(f"  로드 완료 — {len(filtered)}/{len(teacher_sd)} keys 매칭")

    print(f"\n[{mode.upper()}] obs_dim={obs_dim}, {args.num_envs} envs, {args.max_iter} iter")
    print(f"조기 진단: 50 iter 안에 rew_grasp 상승 확인")
    print(f"없으면 grasp_dist_threshold 또는 박스 초기 위치 범위 조정\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
