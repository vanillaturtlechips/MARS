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
parser.add_argument("--student",        action="store_true", default=False)
parser.add_argument("--transport_only", action="store_true", default=False,
                    help="Stage 1: 에피소드 시작 시 즉시 grasped — transport만 학습")
parser.add_argument("--force_grasp",   action="store_true", default=False,
                    help="Teacher 재훈련용: 에피소드 시작 시 즉시 grasped (Teacher obs, 30D)")
parser.add_argument("--bc_ckpt",       type=str,   default=None,
                    help="BC 사전훈련 actor 가중치 (train_bc.py 출력) — Student PPO 초기화용")
parser.add_argument("--teacher_ckpt",  type=str,   default=None,  help="미사용 (하위 호환)")
parser.add_argument("--noise_std",     type=float, default=1.0,
                    help="PPO actor 초기 탐색 노이즈 (BC init 시 0.1 권장)")
parser.add_argument("--warmup_iters", type=int,   default=0,
                    help="BC 사용 시 actor frozen, critic만 먼저 N iter 훈련 후 joint 훈련")
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
    WarehouseManipulationStudentTransportEnvCfg,
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
        init_noise_std=args.noise_std,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    return runner_cfg


def main():
    if args.transport_only:
        env_cfg = WarehouseManipulationStudentTransportEnvCfg()
        obs_dim  = STUDENT_OBS_DIM
        log_dir  = "logs/warehouse_manipulation_transport"
        mode     = "transport"
    elif args.student:
        env_cfg = WarehouseManipulationStudentEnvCfg()
        obs_dim  = STUDENT_OBS_DIM
        log_dir  = "logs/warehouse_manipulation_student"
        mode     = "student"
    elif args.force_grasp:
        env_cfg  = WarehouseManipulationEnvCfg()
        env_cfg.force_grasp_on_reset = True
        obs_dim  = TEACHER_OBS_DIM
        log_dir  = "logs/warehouse_manipulation_teacher_transport"
        mode     = "teacher_transport"
    else:
        env_cfg  = WarehouseManipulationEnvCfg()
        obs_dim  = TEACHER_OBS_DIM
        log_dir  = "logs/warehouse_manipulation"
        mode     = "manipulation"

    env_cfg.scene.num_envs = args.num_envs
    env = WarehouseManipulationEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = make_runner_cfg(obs_dim, mode, args.max_iter)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"
    cfg_dict["algorithm"]["entropy_coef"] = 0.0001
    cfg_dict["algorithm"]["learning_rate"] = args.lr
    runner = OnPolicyRunner(env, cfg_dict, log_dir=log_dir, device=env.device)

    if args.resume_ckpt:
        print(f"[Resume] {args.resume_ckpt}")
        runner.load(args.resume_ckpt)

    ac = None
    if args.bc_ckpt:
        bc_state = torch.load(args.bc_ckpt, map_location=env.device)
        actor_state = {k[len("actor."):]: v for k, v in bc_state.items() if k.startswith("actor.")}
        # rsl_rl 버전별 actor_critic 경로 탐색
        ac = (getattr(runner.alg, "actor_critic", None)
              or getattr(runner, "policy", None)
              or getattr(runner.alg, "policy", None))
        ac.actor.load_state_dict(actor_state)
        print(f"[BC Init] Actor 가중치 로드: {args.bc_ckpt}")

    if args.force_grasp:
        tag = "TEACHER-TRANSPORT"
    elif args.transport_only:
        tag = "TRANSPORT-ONLY"
    elif args.student:
        tag = "STUDENT"
    else:
        tag = "TEACHER"
    _ = tag  # suppress unused warning
    print(f"\n[{tag}] obs_dim={obs_dim}, {args.num_envs} envs, {args.max_iter} iter\n")

    if ac is not None and args.warmup_iters > 0:
        # Phase 1: actor frozen, critic만 웜업
        for param in ac.actor.parameters():
            param.requires_grad = False
        print(f"[Warmup] Actor frozen — critic {args.warmup_iters} iter 웜업 중...")
        runner.learn(num_learning_iterations=args.warmup_iters, init_at_random_ep_len=True)
        # Phase 2: actor unfreeze, joint 훈련
        for param in ac.actor.parameters():
            param.requires_grad = True
        print(f"[Warmup] Actor unfrozen — joint 훈련 시작")
        remaining = args.max_iter - args.warmup_iters
        if remaining > 0:
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=False)
    else:
        runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
