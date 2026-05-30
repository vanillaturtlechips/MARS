"""Pick & Place Fine-tuning.

model_2999.pt (approach+grasp+transport) 기반으로
실제 물리 release+place를 추가 학습.

Phase 2 transport에서 검증된 구조:
  - near_goal gating (0.20m 이내서만 release)
  - bootstrap (초기 N iter 강제 release로 place 경험 주입)
  - settle (release 후 box 안착 확인)

커리큘럼 (place_rate 기반):
  Stage 1 (box_spawn=0.20m): 10iter 평균 place_rate ≥ 20%
  Stage 2 (box_spawn=0.30m): 15iter 평균 place_rate ≥ 20%
  Stage 3 (box_spawn=0.45m): 최종

실행:
  python training/single_robot/train_pickplace.py \\
      --headless --num_envs 2048 \\
      --teacher_ckpt logs/warehouse_transport_phase2/model_600.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",      type=int,   default=2048)
parser.add_argument("--max_iter",      type=int,   default=600)
parser.add_argument("--teacher_ckpt",  type=str,   default="logs/warehouse_transport_phase2/model_600.pt",
                    help="Transport Phase 2 체크포인트 (model_600.pt)")
parser.add_argument("--resume_ckpt",   type=str,   default=None,
                    help="중간 저장 체크포인트 재개")
parser.add_argument("--lr",            type=float, default=1e-4,
                    help="Fine-tuning lr (Teacher보다 낮게)")
parser.add_argument("--save_interval", type=int,   default=100)
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

CURRICULUM_STAGES = [
    (0.20, 20.0, 10),   # Stage 1: 10iter 평균 place_rate ≥ 20%
    (0.30, 20.0, 15),   # Stage 2: 15iter 평균 place_rate ≥ 20%
    (0.45, None, None), # Stage 3: 최종
]

BOOTSTRAP_N = 40    # 40 iter 동안 강제 release
BOOTSTRAP_P = 0.15  # 최대 15% → linear decay


class PickPlaceCurriculumManager:
    def __init__(self, env: WarehouseManipulationEnv) -> None:
        self.env     = env
        self.stage   = 0
        self._history: list[float] = []
        self._apply()

    def _apply(self) -> None:
        dist = CURRICULUM_STAGES[self.stage][0]
        if self.env.cfg.box_spawn_dist != dist:
            print(f"[Curriculum] Stage {self.stage + 1}/{len(CURRICULUM_STAGES)}: "
                  f"box_spawn_dist → {dist:.2f}m")
            self.env.cfg.box_spawn_dist = dist

    def step(self, place_rate: float) -> int:
        if self.stage >= len(CURRICULUM_STAGES) - 1:
            return self.stage
        _, threshold, window = CURRICULUM_STAGES[self.stage]
        self._history.append(float(place_rate))
        if len(self._history) > window:
            self._history.pop(0)
        if len(self._history) < window:
            return self.stage
        avg = sum(self._history) / window
        if avg >= threshold:
            old = CURRICULUM_STAGES[self.stage][0]
            self.stage   += 1
            self._history = []
            new = CURRICULUM_STAGES[self.stage][0]
            print(f"[Curriculum] 진급! 이동평균 place_rate={avg:.1f}% ≥ {threshold:.0f}% "
                  f"→ box_spawn {old:.2f} → {new:.2f}m")
            self._apply()
        return self.stage


def main():
    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs       = args.num_envs
    env_cfg.box_spawn_dist       = CURRICULUM_STAGES[0][0]
    env_cfg.force_grasp_on_reset = True    # 50% force_grasp: approach+grasp도 계속 연습
    env_cfg.near_release_dist    = 0.20
    env_cfg.place_dist_threshold = 0.20    # Phase 2와 동일하게 완화
    env_cfg.rew_release_near     = 50.0
    env_cfg.rew_grip_penalty     = 0.0   # scratch 훈련 시 transport local optimum 방지
    env_cfg.episode_length_s     = 12.0   # approach+grasp+transport+place: 충분한 시간

    env = WarehouseManipulationEnv(env_cfg)

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env      = 128
    runner_cfg.max_iterations         = args.max_iter
    runner_cfg.save_interval          = args.save_interval
    runner_cfg.experiment_name        = "warehouse_pickplace"
    runner_cfg.logger                 = "tensorboard"
    runner_cfg.empirical_normalization = True
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.10,
        actor_hidden_dims=[256, 128, 64],   # transport Phase 2와 동일 → model_600.pt 로드 가능
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"]    = "PPO"
    cfg_dict["algorithm"]["entropy_coef"]  = 0.005
    cfg_dict["algorithm"]["learning_rate"] = args.lr
    runner = OnPolicyRunner(env_wrapped, cfg_dict,
                            log_dir="logs/warehouse_pickplace",
                            device=env_wrapped.device)

    import torch as _torch

    if args.resume_ckpt:
        runner.load(args.resume_ckpt)
        print(f"[Resume] {args.resume_ckpt}")
    elif args.teacher_ckpt:
        runner.load(args.teacher_ckpt)
        runner.current_learning_iteration = 0
        print(f"[Load] Teacher 체크포인트: {args.teacher_ckpt}")
        print(f"       current_iter 0으로 리셋 (fine-tuning 시작)")
    else:
        print("[Init] 랜덤 초기화 (Teacher 체크포인트 없음)")

    # Bootstrap 주입
    env._bootstrap_n = BOOTSTRAP_N
    env._bootstrap_p = BOOTSTRAP_P

    curriculum = PickPlaceCurriculumManager(env)
    start_iter = runner.current_learning_iteration

    print(f"\n{'='*60}")
    print(f"[Pick & Place Fine-tuning] obs={OBS_DIM}D, {args.num_envs} envs")
    print(f"  force_grasp=True(50%)  near_release_dist={env_cfg.near_release_dist}m")
    print(f"  place_dist_threshold={env_cfg.place_dist_threshold}m")
    print(f"  bootstrap: N={BOOTSTRAP_N} iter, p={BOOTSTRAP_P} (linear decay)")
    print(f"  커리큘럼: {CURRICULUM_STAGES}")
    print(f"  [시작 iter] {start_iter} → {args.max_iter}")
    print(f"{'='*60}\n")

    for iteration in range(start_iter, args.max_iter):
        env._current_iter = iteration
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        with _torch.no_grad():
            runner.alg.policy.std.data.clamp_(0.02, 0.30)

        log          = env.extras.get("log", {})
        place_rate   = float(log.get("place_rate",        0.0))
        grasp_rate   = float(log.get("grasp_rate",        0.0))
        nat_grasp    = float(log.get("natural_grasp_rate",grasp_rate))
        dist_box_goal= float(log.get("dist_box_goal",     0.0))
        release_rate = float(log.get("release_rate",      0.0))
        force_rel    = float(log.get("force_rel_rate",    0.0))

        stage = curriculum.step(place_rate)
        p_now = BOOTSTRAP_P * max(0.0, 1.0 - iteration / BOOTSTRAP_N)

        print(f"[iter {iteration+1:4d}] place_rate={place_rate:.1f}%  "
              f"grasp={nat_grasp:.1f}%  dist={dist_box_goal:.3f}m  "
              f"release={release_rate:.1f}%  force={force_rel:.1f}%(p={p_now:.3f})  "
              f"stage={stage+1}/{len(CURRICULUM_STAGES)}")

        if (iteration + 1) % args.save_interval == 0:
            runner.current_learning_iteration = iteration + 1
            path = f"logs/warehouse_pickplace/model_{iteration+1}.pt"
            runner.save(path)
            print(f"[Save] iter={iteration+1} → {path}")

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
