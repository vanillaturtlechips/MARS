"""Phase 2 Transport Policy 학습.

Transport + Place 독립 학습 (approach/grasp 없음).
place_rate 기반 커리큘럼으로 goal 거리를 단계적으로 늘림.

커리큘럼:
  1단계 (goal=0.20m): place_rate 40% 이상 15iter 평균 유지 → 진급
  2단계 (goal=0.35m): place_rate 40% 이상
  3단계 (goal=0.50m): 최종

실행:
  python training/single_robot/train_transport.py --headless --num_envs 2048
  python training/single_robot/train_transport.py --headless --num_envs 2048 --max_iter 50  # 빠른 테스트
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",      type=int,   default=2048)
parser.add_argument("--max_iter",      type=int,   default=600)
parser.add_argument("--resume_ckpt",   type=str,   default=None)
parser.add_argument("--lr",            type=float, default=3e-4)
parser.add_argument("--save_interval", type=int,   default=100)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_transport_env import (
    WarehouseTransportEnv,
    WarehouseTransportEnvCfg,
    TRANSPORT_OBS_DIM,
)

# place_rate 이동평균 기반 커리큘럼: (goal_spawn_dist, 진급 place_rate%, window)
CURRICULUM_STAGES = [
    (0.10, 50.0, 15),   # 1단계: 매우 가까움 → release 행동 부트스트랩
    (0.20, 40.0, 15),   # 2단계: carry 0.20m 학습
    (0.35, 35.0, 15),   # 3단계: carry 0.35m
    (0.50, None, None), # 4단계: 최종
]


class TransportCurriculumManager:
    def __init__(self, env: WarehouseTransportEnv) -> None:
        self.env     = env
        self.stage   = 0
        self._history: list[float] = []
        self._apply()

    def _apply(self) -> None:
        dist = CURRICULUM_STAGES[self.stage][0]
        if self.env.cfg.goal_spawn_dist != dist:
            print(f"[Curriculum] Stage {self.stage + 1}/{len(CURRICULUM_STAGES)}: "
                  f"goal_spawn_dist → {dist:.2f}m")
            self.env.cfg.goal_spawn_dist = dist

    def step(self, place_rate: float) -> None:
        if self.stage >= len(CURRICULUM_STAGES) - 1:
            return
        _, threshold, window = CURRICULUM_STAGES[self.stage]
        self._history.append(float(place_rate))
        if len(self._history) > window:
            self._history.pop(0)
        if len(self._history) < window:
            return
        avg = sum(self._history) / window
        if avg >= threshold:
            old = CURRICULUM_STAGES[self.stage][0]
            self.stage    += 1
            self._history  = []
            new = CURRICULUM_STAGES[self.stage][0]
            print(f"[Curriculum] 진급! 이동평균 place_rate={avg:.1f}% ≥ {threshold:.0f}% "
                  f"(최근 {window}iter) → goal {old:.2f} → {new:.2f}m")
            self._apply()


def main():
    env_cfg = WarehouseTransportEnvCfg()
    env_cfg.scene.num_envs  = args.num_envs
    env_cfg.goal_spawn_dist = CURRICULUM_STAGES[0][0]
    env = WarehouseTransportEnv(env_cfg)

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env     = 128
    runner_cfg.max_iterations        = args.max_iter
    runner_cfg.save_interval         = args.save_interval
    runner_cfg.experiment_name       = "warehouse_transport"
    runner_cfg.logger                = "tensorboard"
    runner_cfg.empirical_normalization = True
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"]   = "PPO"
    cfg_dict["algorithm"]["entropy_coef"] = 0.001  # 낮춤: entropy pressure가 std 감소 방해
    cfg_dict["algorithm"]["learning_rate"] = args.lr
    runner = OnPolicyRunner(env_wrapped, cfg_dict,
                            log_dir="logs/warehouse_transport",
                            device=env_wrapped.device)

    if args.resume_ckpt:
        runner.load(args.resume_ckpt)
        print(f"[Resume] {args.resume_ckpt}")

    curriculum = TransportCurriculumManager(env)
    start_iter = runner.current_learning_iteration

    print(f"\n{'='*60}")
    print(f"[Transport Policy] obs={TRANSPORT_OBS_DIM}D, {args.num_envs} envs")
    print(f"커리큘럼: {CURRICULUM_STAGES}")
    print(f"[시작 iter] {start_iter} → {args.max_iter}")
    print(f"{'='*60}\n")

    import torch as _torch

    for iteration in range(start_iter, args.max_iter):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        with _torch.no_grad():
            runner.alg.policy.std.data.clamp_(0.05, 0.5)

        log        = env.extras.get("log", {})
        place_rate = float(log.get("place_rate", 0.0))
        carry_dist = float(log.get("carry_dist", 0.0))
        grasp_rate = float(log.get("grasp_rate", 0.0))

        curriculum.step(place_rate)

        print(f"[iter {iteration+1:4d}] place_rate={place_rate:.1f}%  "
              f"carry_dist={carry_dist:.3f}m  "
              f"grasp_rate={grasp_rate:.1f}%  "
              f"stage={curriculum.stage+1}/{len(CURRICULUM_STAGES)}")

        if (iteration + 1) % args.save_interval == 0:
            runner.current_learning_iteration = iteration + 1
            path = f"logs/warehouse_transport/model_{iteration+1}.pt"
            runner.save(path)
            print(f"[Save] iter={iteration+1} → {path}")

    env_wrapped.close()


if __name__ == "__main__":
    main()
