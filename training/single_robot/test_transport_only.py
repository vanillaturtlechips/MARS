"""Transport-only 50-iter 진단 테스트.

목적:
  kinematic lock + potential-based reward 조합에서
  transport를 아예 학습할 수 있는지 확인.

설정:
  - force_grasp_fraction=1.0  : 모든 env 에피소드 시작부터 grasped
  - rew_transport_dense=2.0   : potential-based (-dist 기반, 항상 gradient 존재)
  - rew_transport=0.0         : delta reward 제거
  - approach/grasp 보상 전부 0: transport 신호만 남김
  - box_spawn_dist=0.30       : 고정 거리 (커리큘럼 없음)
  - 50 iter, 체크포인트 없음

판단 기준:
  [PASS] surrogate_loss > 0.001 AND transport_delta > 0 → kinematic lock으로 transport 학습 가능
         → hierarchical policy (grasp + transport 분리) 로 재설계
  [FAIL] surrogate_loss ≈ 0 OR transport_delta 계속 음수 → kinematic lock 자체가 문제
         → Phase 2 폐기, 물리 기반 grasp 또는 완전 재설계

실행:
  python training/single_robot/test_transport_only.py --headless --num_envs 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iter", type=int, default=50)
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


def main():
    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    # ── transport-only 설정 ──────────────────────────────────────────
    env_cfg.force_grasp_on_reset  = True
    env_cfg.force_grasp_fraction  = 1.0   # 100% 처음부터 grasped
    env_cfg.box_spawn_dist        = 0.30  # 고정 (커리큘럼 없음)

    # approach/grasp 보상 제거: transport 신호만 남김
    env_cfg.rew_approach   = 0.0
    env_cfg.rew_grasp      = 0.0
    env_cfg.rew_time       = 0.0          # approach 패널티 없음
    env_cfg.rew_time_grasped = -0.1       # stand-still 방지

    # delta reward 제거 → potential-based 단독
    env_cfg.rew_transport       = 0.0
    env_cfg.rew_transport_dense = 2.0    # exp(-dist*5)*grasped: 항상 goal 방향 gradient 존재
    env_cfg.rew_near_place      = 80.0
    env_cfg.rew_place           = 200.0
    # ────────────────────────────────────────────────────────────────

    env = WarehouseManipulationEnv(env_cfg)

    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env   = 128
    runner_cfg.max_iterations      = args.max_iter
    runner_cfg.save_interval       = 999999   # 저장 없음
    runner_cfg.experiment_name     = "transport_only_test"
    runner_cfg.logger              = "tensorboard"
    runner_cfg.empirical_normalization = True
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"]   = "PPO"
    cfg_dict["algorithm"]["entropy_coef"] = 0.005
    cfg_dict["algorithm"]["learning_rate"] = 3e-4
    runner = OnPolicyRunner(env_wrapped, cfg_dict,
                            log_dir="logs/transport_only_test",
                            device=env_wrapped.device)

    print(f"\n{'='*60}")
    print(f"[Transport-only Test] {args.num_envs} envs, {args.max_iter} iter")
    print(f"  force_grasp=100%, reward=potential-based(dense=2.0)")
    print(f"  판단 기준:")
    print(f"    PASS: surrogate_loss > 0.001 AND transport_delta > 0")
    print(f"    FAIL: surrogate_loss ≈ 0 OR transport_delta < 0 지속")
    print(f"{'='*60}\n")

    import torch as _torch
    results = []

    for iteration in range(args.max_iter):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        with _torch.no_grad():
            runner.alg.policy.std.data.clamp_(0.1, 0.4)

        log = env.extras.get("log", {})
        td        = log.get("transport_delta", 0.0)
        gr        = log.get("grasp_rate", 0.0)
        dist_goal = log.get("dist_box_goal", 0.0)
        place_r   = log.get("place_rate", 0.0)

        # surrogate_loss는 rsl_rl이 이미 stdout으로 출력함
        # 핵심 지표만 추가 출력
        print(f"[iter {iteration+1:3d}] transport_delta={td:+.5f}  "
              f"dist_box_goal={dist_goal:.3f}  "
              f"grasp_rate={gr:.1f}%  "
              f"place_rate={place_r:.1f}%")

        results.append(td)

    # ── 최종 판정 ────────────────────────────────────────────────────
    last20 = results[-20:] if len(results) >= 20 else results
    avg_td = sum(last20) / len(last20)

    print(f"\n{'='*60}")
    print(f"[결과] 마지막 20iter 평균 transport_delta = {avg_td:+.5f}")
    if avg_td > 0.0005:
        print("[PASS] transport_delta 양수 수렴 → kinematic lock으로 transport 학습 가능")
        print("       → hierarchical policy (grasp + transport 분리) 재설계 진행")
    elif avg_td > 0:
        print("[MARGINAL] transport_delta 약간 양수 → 학습 시작했지만 미약")
        print("           → iter를 200으로 늘려서 추가 확인 필요")
    else:
        print("[FAIL] transport_delta 음수 지속 → kinematic lock 자체가 문제")
        print("       → Phase 2 폐기, 물리 기반 grasp로 완전 재설계 필요")
    print(f"{'='*60}\n")

    env_wrapped.close()


if __name__ == "__main__":
    main()
