"""Phase 2 — 커리큘럼 Pick & Place 훈련.

3단계:
  1단계 (0~500 iter):   box_spawn_dist=0.05  → grasp 학습
  2단계 (500~1500 iter): box_spawn_dist=0.20  → approach+grasp 학습
  3단계 (1500~3000 iter): box_spawn_dist=0.45 → 풀 pick & place

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
# 커리큘럼 단계: (시작 iter, box_spawn_dist)
CURRICULUM = [
    (0,    0.05),   # 1단계: EE 바로 앞 5cm  → grasp만 학습
    (500,  0.20),   # 2단계: 20cm            → approach+grasp
    (1500, 0.45),   # 3단계: 45cm            → 풀 pick & place
]


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
    cfg_dict["algorithm"]["entropy_coef"]  = 0.0005
    cfg_dict["algorithm"]["learning_rate"] = args.lr
    runner = OnPolicyRunner(env_wrapped, cfg_dict,
                            log_dir="logs/warehouse_manipulation_full",
                            device=env_wrapped.device)

    if args.resume_ckpt:
        runner.load(args.resume_ckpt)

    print(f"\n[Phase 2] obs={OBS_DIM}D, {args.num_envs} envs, curriculum 3단계\n")
    print(f"커리큘럼: {CURRICULUM}\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
