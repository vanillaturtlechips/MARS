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
parser.add_argument("--std_max",       type=float, default=0.4)
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
    (0,    0.20),   # 1단계: 20cm  → approach+grasp 학습
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
        print("[Resume] checkpoint loaded (std 유지)")

    start_iter = runner.current_learning_iteration  # resume 시 900, 신규 시 0
    print(f"\n[Phase 2] obs={OBS_DIM}D, {args.num_envs} envs, curriculum 5단계\n")
    print(f"커리큘럼: {CURRICULUM}\n")
    print(f"[시작 iter] {start_iter} → {args.max_iter}\n")

    import torch as _torch
    STD_MAX = args.std_max
    # transport 학습 감지 후 std 낮춤
    TRANSPORT_THRESHOLD = 0.002   # transport_delta 이 값 초과 시 std 낮춤
    TRANSPORT_CONFIRM   = 30      # 연속 N iter 이상 유지돼야 전환
    STD_MAX_TRANSPORT   = 0.5     # transport 학습 후 std 상한
    _transport_consec   = 0       # 연속 카운터
    _std_lowered        = False   # 한 번만 낮춤

    for iteration in range(start_iter, args.max_iter):
        _apply_curriculum(env, iteration)
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        with _torch.no_grad():
            runner.alg.policy.std.data.clamp_(0.1, STD_MAX)

        # transport_delta 모니터링 → std 자동 낮춤
        if not _std_lowered:
            log = env.extras.get("log", {})
            td = log.get("transport_delta", 0.0)
            if td > TRANSPORT_THRESHOLD:
                _transport_consec += 1
            else:
                _transport_consec = 0
            if _transport_consec >= TRANSPORT_CONFIRM:
                STD_MAX = STD_MAX_TRANSPORT
                _std_lowered = True
                print(f"[Auto-STD] transport_delta {td:.4f} > {TRANSPORT_THRESHOLD} "
                      f"({TRANSPORT_CONFIRM}iter 유지) → std_max {args.std_max} → {STD_MAX}")

        # rsl_rl 내부 카운터가 고정되어 자동 저장이 안 되므로 수동 저장
        if (iteration + 1) % args.save_interval == 0:
            runner.current_learning_iteration = iteration + 1  # resume 시 start_iter 복원용
            save_path = f"logs/warehouse_manipulation_full/model_{iteration + 1}.pt"
            runner.save(save_path)
            print(f"[Save] iter={iteration + 1} → {save_path}")

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
