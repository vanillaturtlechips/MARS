"""Phase 2 Transport — Release + Place 학습.

Phase 1 (carry-only, term_reached) 체크포인트에서 시작.
Phase 2 추가 사항:
  - 그리퍼 release 활성화: gripper action < -0.3 && near_release_dist(0.20m) 이내
  - 박스 물리 착지 → settled & dist < place_dist_threshold(0.20m) → placed 성공
  - goal z = 1.15m (packing table top) 고정 — EE home z(≈1.30m)와 분리
  - Phase 1의 "EE reaches goal" local optimum 차단 (reached → done 제거)

커리큘럼 (place_rate 이동평균 기반):
  Stage 1 (goal=0.15m): 10iter 평균 place_rate ≥ 30%
  Stage 2 (goal=0.30m): 15iter 평균 place_rate ≥ 25%
  Stage 3 (goal=0.50m): 최종

실행:
  python training/single_robot/train_transport_phase2.py \\
      --headless --num_envs 2048 \\
      --phase1_ckpt logs/warehouse_transport/model_XXX.pt

  # 빠른 테스트 (50 iter)
  python training/single_robot/train_transport_phase2.py \\
      --headless --num_envs 512 --max_iter 50 \\
      --phase1_ckpt logs/warehouse_transport/model_XXX.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",      type=int,   default=2048)
parser.add_argument("--max_iter",      type=int,   default=600)
parser.add_argument("--phase1_ckpt",   type=str,   default=None,
                    help="Phase 1 carry-only 체크포인트 경로 (없으면 랜덤 초기화)")
parser.add_argument("--resume_ckpt",   type=str,   default=None,
                    help="Phase 2 중간 저장 체크포인트 재개")
parser.add_argument("--lr",            type=float, default=1e-4,
                    help="Phase 1보다 낮은 lr (fine-tuning)")
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

# place_rate 이동평균 기반 커리큘럼
CURRICULUM_STAGES = [
    (0.15, 30.0, 10),   # Stage 1: 10iter 평균 place_rate ≥ 30%
    (0.30, 25.0, 15),   # Stage 2: 15iter 평균 place_rate ≥ 25%
    (0.50, None, None), # Stage 3: 최종
]


class Phase2CurriculumManager:
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

    def step(self, place_rate: float) -> int:
        """place_rate(물리 착지율)로 진급 판단. 현재 stage 반환."""
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
                  f"(최근 {window}iter) → goal {old:.2f} → {new:.2f}m")
            self._apply()
        return self.stage


def main():
    env_cfg = WarehouseTransportEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # Phase 2 설정
    env_cfg.enable_release       = True
    env_cfg.place_goal_z         = 1.15   # packing table top (EE home z와 분리)
    env_cfg.near_release_dist    = 0.20   # Phase 1(0.12)보다 넓게: release 탐색 용이
    env_cfg.place_dist_threshold = 0.20   # 물리 착지 오차 허용 (Phase 1보다 관대)
    env_cfg.rew_place            = 200.0  # 착지 성공 terminal bonus
    env_cfg.rew_release_near     = 50.0   # goal 근처 release one-time bonus (유지)
    env_cfg.rew_carry_dist       = 1.0    # -dist/step (유지)
    env_cfg.rew_time             = -0.05  # Phase 2: 시간 압박 강화 (release 지연 억제)
    env_cfg.rew_grip_penalty     = 1.5    # near-goal gripper-close 패널티 (soft, xy 기준)
    env_cfg.episode_length_s     = 10.0   # Phase 1(8s)보다 길게: release + 착지 시간 확보
    env_cfg.goal_spawn_dist      = CURRICULUM_STAGES[0][0]

    env = WarehouseTransportEnv(env_cfg)

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env      = 128
    runner_cfg.max_iterations         = args.max_iter
    runner_cfg.save_interval          = args.save_interval
    runner_cfg.experiment_name        = "warehouse_transport_phase2"
    runner_cfg.logger                 = "tensorboard"
    runner_cfg.empirical_normalization = True
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.10,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"]    = "PPO"
    cfg_dict["algorithm"]["entropy_coef"]  = 0.005   # Phase 1보다 약간 높게: release 탐색 유도
    cfg_dict["algorithm"]["learning_rate"] = args.lr
    runner = OnPolicyRunner(env_wrapped, cfg_dict,
                            log_dir="logs/warehouse_transport_phase2",
                            device=env_wrapped.device)

    import torch as _torch

    if args.resume_ckpt:
        runner.load(args.resume_ckpt)
        print(f"[Resume] Phase 2 체크포인트: {args.resume_ckpt}")
    elif args.phase1_ckpt:
        runner.load(args.phase1_ckpt)
        # Phase 1 체크포인트: runner.current_learning_iteration이 Phase 1 기준이므로 0으로 리셋
        runner.current_learning_iteration = 0
        print(f"[Load] Phase 1 carry 정책 로드: {args.phase1_ckpt}")
        print(f"       current_iter 0으로 리셋 (Phase 2 새로 시작)")
    else:
        # 랜덤 초기화: actor output layer zero-init
        with _torch.no_grad():
            runner.alg.policy.actor[-1].weight.data.zero_()
            runner.alg.policy.actor[-1].bias.data.zero_()
        print("[Init] 랜덤 초기화 (Phase 1 체크포인트 없음)")

    grip_bias = runner.alg.policy.actor[-1].bias.data[-1].item()
    print(f"[Grip Init] actor bias[-1] = {grip_bias:+.3f}  (bias override 없음 — bootstrap+penalty로 탐색)")

    curriculum = Phase2CurriculumManager(env)
    start_iter = runner.current_learning_iteration

    # Bootstrap 설정: env에 주입 (매 iter _current_iter도 갱신)
    BOOTSTRAP_N = 40    # 40 iter 동안 강제 release (linear decay)
    BOOTSTRAP_P = 0.15  # 최대 15% 확률 → iter 증가에 따라 0까지 감소
    env._bootstrap_n = BOOTSTRAP_N
    env._bootstrap_p = BOOTSTRAP_P

    print(f"\n{'='*60}")
    print(f"[Phase 2 Transport] obs={TRANSPORT_OBS_DIM}D, {args.num_envs} envs")
    print(f"  enable_release=True  near_release_dist={env_cfg.near_release_dist}m")
    print(f"  place_goal_z={env_cfg.place_goal_z}m  place_dist_threshold={env_cfg.place_dist_threshold}m")
    print(f"  rew_place={env_cfg.rew_place}  rew_grip_penalty={env_cfg.rew_grip_penalty}")
    print(f"  bootstrap: N={BOOTSTRAP_N} iter, p={BOOTSTRAP_P} (linear decay)")
    print(f"  커리큘럼: {CURRICULUM_STAGES}")
    print(f"  [시작 iter] {start_iter} → {args.max_iter}")
    print(f"{'='*60}\n")

    for iteration in range(start_iter, args.max_iter):
        env._current_iter = iteration  # bootstrap p_now 계산용
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        with _torch.no_grad():
            # Phase 2: 탐색 범위 축소 (release 행동은 gripper 하나라 크게 필요 없음)
            runner.alg.policy.std.data.clamp_(0.02, 0.25)

        log        = env.extras.get("log", {})
        place_rate = float(log.get("place_rate",  0.0))
        carry_dist = float(log.get("carry_dist",  0.0))
        ik_err     = float(log.get("ik_err",      0.0))
        grip_mean  = float(log.get("grip_action_mean", 0.0))
        grip_std   = float(log.get("grip_action_std",  0.0))
        reached_pct   = float(log.get("term_reached",   0.0))
        placed_pct    = float(log.get("term_placed",    0.0))
        release_pct   = float(log.get("release_rate",   0.0))
        force_rel_pct = float(log.get("force_rel_rate", 0.0))

        stage = curriculum.step(place_rate)

        # 처음 3 iter: 초기 상태 진단
        if iteration - start_iter < 3:
            wn  = runner.alg.policy.actor[-1].weight.data.norm().item()
            std = runner.alg.policy.std.data.mean().item()
            grip_bias = runner.alg.policy.actor[-1].bias.data[-1].item()
            print(f"  [Diag iter{iteration+1}] w_norm={wn:.4f}  std={std:.4f}  "
                  f"grip_bias={grip_bias:+.3f}  carry_dist={carry_dist:.4f}")

        p_now = BOOTSTRAP_P * max(0.0, 1.0 - iteration / BOOTSTRAP_N)
        print(f"[iter {iteration+1:4d}] place_rate={place_rate:.1f}%  "
              f"carry_dist={carry_dist:.3f}m  "
              f"grip={grip_mean:+.2f}±{grip_std:.2f}  "
              f"release={release_pct:.1f}%  force={force_rel_pct:.1f}%(p={p_now:.3f})  "
              f"placed={placed_pct:.1f}%  stage={stage+1}/{len(CURRICULUM_STAGES)}")

        if (iteration + 1) % args.save_interval == 0:
            runner.current_learning_iteration = iteration + 1
            path = f"logs/warehouse_transport_phase2/model_{iteration+1}.pt"
            runner.save(path)
            print(f"[Save] iter={iteration+1} → {path}")

    env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
