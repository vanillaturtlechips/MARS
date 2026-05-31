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
parser.add_argument("--num_envs",       type=int,   default=128)
parser.add_argument("--max_iter",       type=int,   default=5000)
parser.add_argument("--ippo_ckpt",      type=str,   default=None, help="IPPO 체크포인트 (actor-only 로드)")
parser.add_argument("--mappo_ckpt",     type=str,   default=None, help="MAPPO 체크포인트 (full 로드, 이어서 훈련용)")
parser.add_argument("--from_scratch",   action="store_true", default=False)
parser.add_argument("--reset_noise_std", type=float, default=None,
                    help="체크포인트 로드 후 noise_std 강제 설정 (예: 0.5). 미지정 시 체크포인트 값 유지")
parser.add_argument("--enable_obstacles", action="store_true", default=False,
                    help="동적 장애물(갑자기 출현) 활성화 — model_9999 fine-tune용")
parser.add_argument("--enable_battery", action="store_true", default=False,
                    help="배터리/충전소 활성화 — obs 17→20D (model_9999는 ippo_ckpt로 actor partial transfer)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# AppLauncher가 sys.path를 덮어쓰므로 rsl_rl 소스 경로를 강제 주입
import os as _os
_rsl_rl_src = "/workspace/rsl_rl"
if _os.path.isdir(_rsl_rl_src) and _rsl_rl_src not in sys.path:
    sys.path.insert(0, _rsl_rl_src)

import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_marl_env import (
    WarehouseMARLEnv, WarehouseMARLEnvCfg, N_ROBOTS, OBS_PER_ROBOT, obs_per_robot,
)
from envs.warehouse.ippo_wrapper import IPPOReshapeWrapper

ACT_DIM = 3


def make_mappo_runner_cfg(num_envs: int, max_iter: int, exp_name: str = "warehouse_mappo") -> RslRlOnPolicyRunnerCfg:
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env  = 24
    runner_cfg.max_iterations     = max_iter
    runner_cfg.save_interval      = 200
    runner_cfg.experiment_name    = exp_name
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
    obr = obs_per_robot(args.enable_battery)   # 17 또는 20
    env_cfg = WarehouseMARLEnvCfg()
    env_cfg.scene.num_envs    = args.num_envs
    env_cfg.enable_battery    = args.enable_battery
    env_cfg.observation_space = obr * N_ROBOTS
    env_cfg.state_space       = obr * N_ROBOTS
    env_cfg.action_space      = ACT_DIM * N_ROBOTS          # 9
    env_cfg.enable_dynamic_obstacles = args.enable_obstacles
    if args.enable_obstacles:
        print("[MAPPO] 동적 장애물 활성화 — obs 차원 유지(min 거리)")
    if args.enable_battery:
        print(f"[MAPPO] 배터리/충전소 활성화 — obs {obr}D. model_9999는 --ippo_ckpt로 actor partial transfer 권장")

    env = WarehouseMARLEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)
    env = IPPOReshapeWrapper(env, N_ROBOTS, obr)
    # actor obs=obr per-robot (battery 시 20D) — IPPO 체크포인트 actor partial 호환

    # 배터리 정책은 obs 20D 별개 → 기존 17D 모델 덮어쓰기 방지로 경로 분리
    exp_name = "warehouse_mappo_battery" if args.enable_battery else "warehouse_mappo"
    runner_cfg = make_mappo_runner_cfg(args.num_envs, args.max_iter, exp_name)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"
    # 배터리는 충전 행동 탐색이 필요 → entropy_coef 상향(0.001→0.01)으로 조기 수렴 방지
    cfg_dict["algorithm"]["entropy_coef"] = 0.01 if args.enable_battery else 0.001
    runner = OnPolicyRunner(env, cfg_dict, log_dir=f"logs/{exp_name}", device=env.device)

    if args.mappo_ckpt and not args.from_scratch:
        print(f"[MAPPO] full 로드 (이어서 훈련): {args.mappo_ckpt}")
        runner.load(args.mappo_ckpt)
        if args.reset_noise_std is not None:
            runner.alg.policy.std.data.fill_(args.reset_noise_std)
            print(f"[MAPPO] noise_std 강제 설정: {args.reset_noise_std}")
    elif args.ippo_ckpt and not args.from_scratch:
        print(f"[MAPPO] actor partial 로드 (IPPO→MAPPO): {args.ippo_ckpt}")
        ckpt = torch.load(args.ippo_ckpt, map_location=env.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        actor_sd = {k: v for k, v in sd.items() if k.startswith("actor.")}
        # obs 차원이 다르면(battery: 17→20) 입력층 size mismatch → shape 일치 레이어만 transfer
        model_sd = runner.alg.policy.state_dict()
        filtered = {k: v for k, v in actor_sd.items()
                    if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in actor_sd if k not in filtered]
        runner.alg.policy.load_state_dict(filtered, strict=False)
        print(f"[MAPPO] actor {len(filtered)}층 transfer | 입력층 등 {len(skipped)}개 새로 초기화: {skipped}")
        if args.reset_noise_std is not None:
            runner.alg.policy.std.data.fill_(args.reset_noise_std)
            print(f"[MAPPO] noise_std 강제 설정: {args.reset_noise_std}")
        if args.reset_noise_std is not None:
            runner.alg.policy.std.data.fill_(args.reset_noise_std)
            print(f"[MAPPO] noise_std 강제 설정: {args.reset_noise_std}")
    elif args.from_scratch:
        print("[MAPPO] 처음부터 훈련")
    else:
        print("[경고] --mappo_ckpt 또는 --ippo_ckpt 지정 권장")

    print(f"\n[True CTDE MAPPO]")
    print(f"  Actor obs : {obr}-dim per-robot (goal+vel+shelf+relative_robots{'+battery+charger' if args.enable_battery else ''})")
    print(f"  Critic obs: {obr * N_ROBOTS}-dim global state")
    print(f"  Reward    : per-robot (credit assignment)")
    print(f"  {args.num_envs} envs × {N_ROBOTS} robots = {args.num_envs * N_ROBOTS} virtual envs\n")

    runner.learn(num_learning_iterations=args.max_iter, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
