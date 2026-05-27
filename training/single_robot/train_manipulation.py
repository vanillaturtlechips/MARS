"""Phase 2 — 커리큘럼 Pick & Place 훈련.

5단계:
  1단계 (0~500 iter):    box_spawn_dist=0.15  → approach+grasp 학습
  2단계 (500~1000 iter): box_spawn_dist=0.20  → approach+grasp 심화
  3단계 (1000~1500 iter): box_spawn_dist=0.30 → transport 입문
  4단계 (1500~2000 iter): box_spawn_dist=0.38 → transport 심화
  5단계 (2000~3000 iter): box_spawn_dist=0.45 → 풀 pick & place

실행:
  python training/single_robot/train_manipulation.py --headless --num_envs 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",      type=int,   default=256)
parser.add_argument("--max_iter",      type=int,   default=3000)
parser.add_argument("--resume_ckpt",   type=str,   default=None)
parser.add_argument("--lr",            type=float, default=1e-3)
parser.add_argument("--save_interval", type=int,   default=300)
parser.add_argument("--std_max",       type=float, default=0.8)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv,
    WarehouseManipulationEnvCfg,
    OBS_DIM,
)

# 커리큘럼 단계: (시작 iter, box_spawn_dist)
CURRICULUM = [
    (0,    0.15),   # 1단계: 15cm  → approach+grasp 학습
    (500,  0.20),   # 2단계: 20cm  → approach+grasp 심화
    (1000, 0.30),   # 3단계: 30cm  → transport 입문
    (1500, 0.38),   # 4단계: 38cm  → transport 심화
    (2000, 0.45),   # 5단계: 45cm  → 풀 pick & place
]


def _apply_curriculum(env: WarehouseManipulationEnv, iteration: int) -> None:
    """iter 기준으로 box_spawn_dist를 단계적으로 올림."""
    dist = CURRICULUM[0][1]
    for start_iter, spawn_dist in CURRICULUM:
        if iteration >= start_iter:
            dist = spawn_dist
    if env.cfg.box_spawn_dist != dist:
        print(f"[Curriculum] iter={iteration}: box_spawn_dist {env.cfg.box_spawn_dist:.2f} → {dist:.2f}")
        env.cfg.box_spawn_dist = dist


def main():
    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.box_spawn_dist = CURRICULUM[0][1]
    env = WarehouseManipulationEnv(env_cfg)

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env     = 128
    runner_cfg.max_iterations        = args.max_iter
    runner_cfg.save_interval         = args.save_interval
    runner_cfg.experiment_name       = "warehouse_manipulation_full"
    runner_cfg.logger                = "tensorboard"
    runner_cfg.empirical_normalization = True
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"]    = "PPO"
    cfg_dict["algorithm"]["entropy_coef"]  = 0.005
    cfg_dict["algorithm"]["learning_rate"] = args.lr
    runner = OnPolicyRunner(env_wrapped, cfg_dict,
                            log_dir="logs/warehouse_manipulation_full",
                            device=env_wrapped.device)

    if args.resume_ckpt:
        runner.load(args.resume_ckpt)
        runner.alg.policy.std.data.fill_(0.6)  # transport 탐색을 위해 std 리셋
        print("[Resume] action noise std → 0.6 (강제 리셋)")

    print(f"\n[Phase 2] obs={OBS_DIM}D, {args.num_envs} envs, curriculum 3단계\n")
    print(f"커리큘럼: {CURRICULUM}\n")

    import torch as _torch
    STD_MAX = args.std_max

    # learn(1) 반복으로 매 iter마다 커리큘럼 적용
    # rsl_rl OnPolicyRunner.learn()은 current_learning_iteration을 누적하므로 안전
    for iteration in range(args.max_iter):
        _apply_curriculum(env, iteration)
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        with _torch.no_grad():
            runner.alg.policy.std.data.clamp_(0.1, STD_MAX)

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
