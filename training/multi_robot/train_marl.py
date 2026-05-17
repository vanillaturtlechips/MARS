"""Phase 3 — MAPPO 훈련 (IPPO 체크포인트에서 fine-tuning).

CTDE: 훈련 시 공유 Critic이 전체 상태를 봄.
      실행 시 각 Actor는 자기 관측만 사용.

실행:
  # IPPO 수렴 후 MAPPO fine-tuning
  python training/multi_robot/train_marl.py \
    --ippo_ckpt logs/warehouse_ippo/model_XXX.pt

  # 처음부터 MAPPO
  python training/multi_robot/train_marl.py --from_scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3 MAPPO 훈련")
parser.add_argument("--num_envs",    type=int,   default=128)
parser.add_argument("--max_iter",    type=int,   default=5000)
parser.add_argument("--ippo_ckpt",   type=str,   default=None, help="IPPO 체크포인트 경로")
parser.add_argument("--from_scratch", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_marl_env import WarehouseMARLEnv, WarehouseMARLEnvCfg, N_ROBOTS, OBS_PER_ROBOT
from envs.warehouse.ippo_wrapper import IPPOReshapeWrapper

ACT_DIM = 3


def make_mappo_runner_cfg(num_envs: int, max_iter: int) -> RslRlOnPolicyRunnerCfg:
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env  = 24
    runner_cfg.max_iterations     = max_iter
    runner_cfg.save_interval      = 200
    runner_cfg.experiment_name    = "warehouse_mappo"
    runner_cfg.run_name           = f"mappo_n{N_ROBOTS}_env{num_envs}"
    runner_cfg.logger             = "tensorboard"
    runner_cfg.empirical_normalization = True

    # Asymmetric Actor-Critic: actor obs ≠ critic obs
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,   # IPPO fine-tuning이면 noise 낮게 시작
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    return runner_cfg


def main():
    env_cfg = WarehouseMARLEnvCfg()
    env_cfg.scene.num_envs    = args.num_envs
    env_cfg.observation_space = OBS_PER_ROBOT * N_ROBOTS   # 27 (joint)
    env_cfg.action_space      = ACT_DIM * N_ROBOTS          # 9

    env = WarehouseMARLEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)
    env = IPPOReshapeWrapper(env, N_ROBOTS, OBS_PER_ROBOT)
    # actor obs=9 (per-robot) — IPPO 체크포인트와 호환

    runner_cfg = make_mappo_runner_cfg(args.num_envs, args.max_iter)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"
    cfg_dict["algorithm"]["entropy_coef"] = 0.001  # per-robot 보상으로 신호 깨끗해져 낮은 값 충분
    cfg_dict["obs_groups"] = None  # to_dict()가 {}로 만들면 resolve_obs_groups가 거부 → None으로 auto-detect
    runner = OnPolicyRunner(env, cfg_dict, log_dir="logs/warehouse_mappo", device=env.device)

    if args.ippo_ckpt and not args.from_scratch:
        print(f"[MAPPO] IPPO 체크포인트 로드: {args.ippo_ckpt}")
        runner.load(args.ippo_ckpt)
    elif args.from_scratch:
        print("[MAPPO] 처음부터 훈련")
    else:
        print("[경고] --ippo_ckpt 없음. --from_scratch 또는 --ippo_ckpt 지정 권장")

    print(f"\n[True CTDE MAPPO]")
    print(f"  Actor obs : {OBS_PER_ROBOT}-dim per-robot")
    print(f"  Critic obs: {OBS_PER_ROBOT * N_ROBOTS}-dim global state")
    print(f"  Reward    : per-robot (credit assignment)")
    print(f"  {args.num_envs} envs × {N_ROBOTS} robots = {args.num_envs * N_ROBOTS} virtual envs\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
