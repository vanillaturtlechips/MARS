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
parser.add_argument("--diff_drive", action="store_true", default=False,
                    help="레벨1: vy(옆걸음) 하드 차단 → 커브로만 방향전환. (절벽 — 단발은 실패 확인됨)")
parser.add_argument("--rew_spin", type=float, default=0.0,
                    help="|omega| 페널티 계수(제자리 휙휙 회전 억제). 전이 중엔 0 권장(회전 억누르면 학습 방해)")
parser.add_argument("--strafe_curriculum", action="store_true", default=False,
                    help="레벨1 커리큘럼: max_vy를 1.0→0으로 성공률 게이트 적응 감소(하드컷 절벽 회피). model_10998에서 시작")
parser.add_argument("--curriculum_thresh", type=float, default=0.8,
                    help="이 도달률 이상이면 max_vy 한 단계 감소(문헌 표준 0.8). task 천장 낮으면 baseline 약간 아래로")
parser.add_argument("--curriculum_collapse", type=float, default=0.5,
                    help="도달률 이 밑으로 붕괴하면 max_vy 한 단계 되돌림(revert)")
parser.add_argument("--curriculum_step", type=float, default=0.1, help="max_vy 감소 폭")
parser.add_argument("--curriculum_window", type=int, default=200, help="성공률 측정 윈도우(완료 에피소드 수)")
parser.add_argument("--rew_heading", type=float, default=0.0,
                    help="heading→goal 속도투영 보상(돌아서 굴러가기 학습). 0.3~0.5 권장(전이 가속)")
parser.add_argument("--max_omega", type=float, default=None,
                    help="회전 권한 상향(strafe 대체). None=기본 2.0 유지, 권장 2.6(≈1.3×)")
parser.add_argument("--entropy_coef", type=float, default=None,
                    help="PPO 엔트로피 계수. None=기본(0.001). 커리큘럼 노이즈 폭발 억제엔 0.0 권장")
parser.add_argument("--freeze_std", action="store_true", default=False,
                    help="액션 노이즈 std를 학습 불가로 동결 — PPO surrogate gradient에 의한 std 폭발 차단. "
                         "entropy_coef=0이어도 std는 학습되어 자동 증가하는 문제 해결. "
                         "커리큘럼 훈련에 강력 권장. --reset_noise_std와 함께 사용하면 시작 std도 통제 가능.")
parser.add_argument("--extended_obs", action="store_true", default=False,
                    help="obs 17→19D 확장 (가장 가까운 장애물 body-frame xy 추가). "
                         "diff-drive에서 omega 기반 회피 학습용 정보. model_10998(17D) 호환 깨짐 → from_scratch 권장.")
parser.add_argument("--rew_evasive_turn", type=float, default=0.0,
                    help="확장보상: 가까운 장애물 반대방향 omega에 +. 권장 0.1. extended_obs=True 필요.")
parser.add_argument("--rew_clearance", type=float, default=0.0,
                    help="확장보상: 정면 콘에 가까운 장애물 없을 때 +/step. 권장 0.05. extended_obs=True 필요.")
parser.add_argument("--no_obs_norm", action="store_true", default=False,
                    help="obs 정규화 끔(raw obs). model_10998(정규화 없이 학습)에서 warm-start할 때 규격 일치용. "
                         "from_scratch엔 비권장(raw obs는 그래디언트 죽어 학습 안 됨).")
parser.add_argument("--rew_shelf_prox", type=float, default=0.0,
                    help="정적 선반 근접 페널티(표면 가까이 거리비례 -/step). 권장 -2.0. 선반 회피 직접 신호.")
parser.add_argument("--goal_shelf_margin", type=float, default=0.8,
                    help="목표를 선반에서 뗄 여유(작을수록 선반 근처/뒤 목표 → 우회 필수). 회피 학습엔 0.15 권장.")
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


def make_mappo_runner_cfg(num_envs: int, max_iter: int, exp_name: str = "warehouse_mappo",
                          obs_norm: bool = True) -> RslRlOnPolicyRunnerCfg:
    runner_cfg = RslRlOnPolicyRunnerCfg()
    runner_cfg.num_steps_per_env  = 24
    runner_cfg.max_iterations     = max_iter
    runner_cfg.save_interval      = 200
    runner_cfg.experiment_name    = exp_name
    runner_cfg.run_name           = f"mappo_n{N_ROBOTS}_env{num_envs}"
    runner_cfg.logger             = "tensorboard"
    # 정규화: from_scratch는 ON 필수(raw obs는 그래디언트 죽어 학습 안 됨, std 0.8 고정 확인).
    # warm-start(model_10998는 정규화 없이 학습됨)는 규격 일치 위해 OFF(--no_obs_norm).
    #   이 rsl_rl은 empirical_normalization을 actor/critic_obs_normalization로 매핑, 정규화 모듈을
    #   policy 안에 둬 model_state_dict에 저장(OnPolicyRunner.load로 복원). 추론 스크립트도 동일 설정 필요.
    runner_cfg.empirical_normalization = obs_norm

    # Asymmetric Actor-Critic: actor obs ≠ critic obs
    runner_cfg.policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,   # IPPO fine-tuning이면 noise 낮게 시작
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    return runner_cfg


def main():
    obr = obs_per_robot(args.enable_battery, args.extended_obs)   # 17/19/20/22
    env_cfg = WarehouseMARLEnvCfg()
    env_cfg.scene.num_envs    = args.num_envs
    env_cfg.enable_battery    = args.enable_battery
    env_cfg.extended_obstacle_obs = args.extended_obs
    env_cfg.rew_evasive_turn  = args.rew_evasive_turn
    env_cfg.rew_clearance     = args.rew_clearance
    env_cfg.rew_shelf_prox    = args.rew_shelf_prox
    env_cfg.goal_shelf_margin = args.goal_shelf_margin
    env_cfg.observation_space = obr * N_ROBOTS
    env_cfg.state_space       = obr * N_ROBOTS
    env_cfg.action_space      = ACT_DIM * N_ROBOTS          # 9
    env_cfg.enable_dynamic_obstacles = args.enable_obstacles
    env_cfg.disable_strafe    = args.diff_drive
    env_cfg.strafe_curriculum = args.strafe_curriculum
    env_cfg.rew_heading       = args.rew_heading
    if args.extended_obs:
        print(f"[MAPPO] 확장 obs({obr}D) + 회피보상(evasive_turn {args.rew_evasive_turn}, clearance {args.rew_clearance})")
    if args.max_omega is not None:
        env_cfg.max_omega = args.max_omega
        print(f"[MAPPO] max_omega 상향: {args.max_omega} (회전 권한↑ — strafe 대체)")
    if args.strafe_curriculum:
        env_cfg.curriculum_success_thresh  = args.curriculum_thresh
        env_cfg.curriculum_collapse_thresh = args.curriculum_collapse
        env_cfg.curriculum_step            = args.curriculum_step
        env_cfg.curriculum_window          = args.curriculum_window
        env_cfg.rew_spin = args.rew_spin    # 전이 중엔 0 권장
        print(f"[MAPPO] diff-drive 커리큘럼: max_vy 1.0→0 적응감소(thresh {args.curriculum_thresh}/"
              f"collapse {args.curriculum_collapse}, step {args.curriculum_step}, window {args.curriculum_window}), "
              f"heading {args.rew_heading}, spin {args.rew_spin}")
    elif args.diff_drive:
        env_cfg.rew_spin = args.rew_spin
        print(f"[MAPPO] diff-drive 하드차단(레벨1): vy=0 + omega 페널티 {args.rew_spin}")
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
    if args.diff_drive:
        exp_name = "warehouse_mappo_diffdrive"   # 홀로노믹 체크포인트와 분리(덮어쓰기 방지)
    if args.strafe_curriculum:
        exp_name = "warehouse_mappo_diffdrive_curr"   # 커리큘럼 전용(하드컷과도 분리)
    if args.extended_obs:
        exp_name = "warehouse_mappo_extobs"   # 확장 obs(19D) — 기존 17D와 호환 깨짐, 경로 분리
    runner_cfg = make_mappo_runner_cfg(args.num_envs, args.max_iter, exp_name,
                                       obs_norm=not args.no_obs_norm)
    if args.no_obs_norm:
        print("[MAPPO] obs 정규화 OFF (raw obs) — warm-start 규격 일치용")
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["algorithm"]["class_name"] = "PPO"
    # 배터리: entropy_coef 0.01은 std 폭발(6.6), 0.005+clamp는 iter루프 부작용(episode 급감).
    # → 통짜 learn 유지 + entropy_coef 0.002(0.001보다 약간 높여 충전 탐색, 발산은 약함)
    _ent = 0.002 if args.enable_battery else 0.001
    if args.entropy_coef is not None:
        _ent = args.entropy_coef   # 커리큘럼: 0으로 노이즈 폭발(0.5→1.42) 억제, 도달률 회복
    cfg_dict["algorithm"]["entropy_coef"] = _ent
    # rsl_rl 3.x obs_groups (구버전 isaaclab_rl은 누락 → setdefault로 양쪽 안전)
    cfg_dict.setdefault("obs_groups", {"policy": ["policy"], "critic": ["critic"]})
    runner = OnPolicyRunner(env, cfg_dict, log_dir=f"logs/{exp_name}", device=env.device)

    if args.mappo_ckpt and not args.from_scratch:
        print(f"[MAPPO] full 로드 (이어서 훈련): {args.mappo_ckpt}")
        runner.load(args.mappo_ckpt)
        if args.reset_noise_std is not None:
            runner.alg.policy.std.data.fill_(args.reset_noise_std)
            print(f"[MAPPO] noise_std 강제 설정: {args.reset_noise_std}")
    elif args.ippo_ckpt and not args.from_scratch:
        print(f"[MAPPO] actor 로드 (제로패딩 warm-start): {args.ippo_ckpt}")
        ckpt = torch.load(args.ippo_ckpt, map_location=env.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        actor_sd = {k: v for k, v in sd.items() if k.startswith("actor.")}
        model_sd = runner.alg.policy.state_dict()
        # ── 제로패딩 입력층 이식: obs 17→19(확장) 시 입력층(actor.0.weight, 256×17→256×19)을
        #    버리지 않고 [기존 17칸 복사 + 새 칸 0]으로 키움. 새 dim은 per-robot obs '끝'에
        #    붙으므로(끝 2열) 0초기화하면 시작 시점 출력이 원본과 동일 → model_10998의 회피 능력
        #    100% 보존하면서, 학습으로 새 방향 정보를 점진 활용(선반 회피 추가 학습).
        new_sd, copied, grown, skipped = {}, [], [], []
        for k, v in actor_sd.items():
            if k not in model_sd:
                skipped.append(k); continue
            tgt = model_sd[k]
            if v.shape == tgt.shape:
                new_sd[k] = v; copied.append(k)
            elif (v.dim() == 2 and tgt.dim() == 2 and v.shape[0] == tgt.shape[0]
                  and v.shape[1] < tgt.shape[1]):
                w = tgt.clone()
                w[:, :v.shape[1]] = v          # 기존 입력 가중치 복사
                w[:, v.shape[1]:] = 0.0        # 새 입력(장애물 방향) 가중치 0 → 시작 영향 0
                new_sd[k] = w; grown.append(f"{k}{tuple(v.shape)}→{tuple(tgt.shape)}")
            else:
                skipped.append(k)
        runner.alg.policy.load_state_dict(new_sd, strict=False)
        print(f"[MAPPO] 복사 {len(copied)}층 | 제로패딩 {grown} | 건너뜀 {skipped}")
        if args.reset_noise_std is not None:
            runner.alg.policy.std.data.fill_(args.reset_noise_std)
            print(f"[MAPPO] noise_std 강제 설정: {args.reset_noise_std}")
    elif args.from_scratch:
        print("[MAPPO] 처음부터 훈련")
    else:
        print("[경고] --mappo_ckpt 또는 --ippo_ckpt 지정 권장")

    # ── std 동결 (PPO surrogate가 std를 키우는 부작용 차단) ────────────────
    # 원인: rsl_rl의 policy.std는 nn.Parameter라 entropy_coef=0이어도 surrogate
    # gradient가 std를 키울 수 있음(advantage 음수 다발 시 "탐색 늘려"로 해석).
    # 커리큘럼 동역학 변경 시 std가 0.3→0.84로 폭발해 정책이 랜덤 노이즈가 됨.
    # 해결: std 관련 모든 파라미터의 requires_grad=False — 학습에서 제외.
    if args.freeze_std:
        frozen = []
        for name, param in runner.alg.policy.named_parameters():
            if "std" in name.lower():
                param.requires_grad = False
                frozen.append((name, float(param.data.flatten()[0].item())))
        if frozen:
            print(f"[MAPPO] std 동결: {frozen} (PPO surrogate의 std 폭발 차단)")
        else:
            print("[MAPPO] 경고: 'std' 이름 파라미터 못 찾음 — freeze_std 효과 없음")

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
