"""훈련 건강 진단 테스트 — Isaac Sim 불필요.

"surrogate_loss가 왜 0인가", "entropy가 왜 지배하는가"를 수치로 검증.
훈련 로그를 받으면 어떤 지표가 문제인지 즉시 진단 가능.

실행:
  python3 -m pytest tests/test_training_health.py -v -s
  (출력 보려면 반드시 -s 필요)
"""

import math
import torch
import pytest

# ── 현재 설정 상수 ────────────────────────────────────────────────────────────
# 변경 시 여기만 수정하면 모든 테스트에 반영됨

# 훈련 스크립트
NUM_STEPS_PER_ENV  = 128
MAX_ITER           = 3000
ENTROPY_COEF       = 0.0001
INIT_NOISE_STD     = 1.0
NUM_ENVS           = 2048

# 환경
EPISODE_LENGTH_S   = 3.0
SIM_DT             = 1.0 / 120.0
DECIMATION         = 2
STEP_DT            = SIM_DT * DECIMATION   # 1/60 s
EPISODE_STEPS      = int(EPISODE_LENGTH_S / STEP_DT)  # 180

# 보상
REW_GOAL_DIST      = 2.0
REW_PLACE          = 800.0
REW_TIME           = -0.02
REW_GRASP          = 30.0
REW_APPROACH       = 0.5
GRASP_THRESHOLD    = 0.25
PLACE_THRESHOLD    = 0.12

# 관측 차원
OBS_DIM = 30

# Franka 관절 제어
ARM_DELTA_SCALE    = 0.05     # rad/step
GRIPPER_DELTA_SCALE = 0.02    # m/step
N_ARM_JOINTS       = 7

# 훈련 건강 임계값 (이 값 기준으로 pass/fail 판단)
SURROGATE_ALIVE    = 0.005    # surrogate_loss > 이 값이어야 gradient 살아있음
ENTROPY_RATIO_MAX  = 0.30     # entropy_coef / surrogate_loss < 이 값이면 안전
STD_OK_MIN         = 0.10     # std < 이면 policy 결정론 붕괴
STD_OK_MAX         = 2.50     # std > 이면 탐색 불가 (완전 랜덤)
PLACE_SIGNAL_MIN   = 5        # 배치당 place 이벤트 이 개 이상이어야 학습 가능


# ── 1. 버퍼 커버리지 ──────────────────────────────────────────────────────────

def test_buffer_coverage():
    """배치 한 번에 에피소드의 몇 %를 담는가.

    num_steps=128, episode=180 → 71% 커버
    에피소드 종료(place/timeout)를 자주 봐야 advantage 품질이 높음.
    """
    coverage = NUM_STEPS_PER_ENV / EPISODE_STEPS
    print(f"\n[버퍼 커버리지]")
    print(f"  num_steps_per_env : {NUM_STEPS_PER_ENV}")
    print(f"  episode_length    : {EPISODE_STEPS} steps ({EPISODE_LENGTH_S}s)")
    print(f"  커버리지           : {coverage*100:.1f}%")
    print(f"  판정               : {'✅ 충분 (>60%)' if coverage >= 0.6 else '❌ 부족 (<60%)'}")
    assert coverage >= 0.6, (
        f"커버리지 {coverage*100:.1f}% — episode_length를 줄이거나 num_steps를 늘려야 함"
    )


def test_episodes_per_batch():
    """배치 한 번에 완료되는 에피소드 수.

    더 많은 완료 에피소드 → 더 많은 terminal reward → 더 정확한 value target.
    """
    ep_per_env   = NUM_STEPS_PER_ENV / EPISODE_STEPS
    total_ep     = NUM_ENVS * ep_per_env
    print(f"\n[배치당 에피소드 완료 수]")
    print(f"  env당 완료         : {ep_per_env:.2f} 에피소드")
    print(f"  전체 ({NUM_ENVS} envs) : {total_ep:.0f} 에피소드/배치")
    print(f"  판정               : {'✅ 충분 (>500)' if total_ep >= 500 else '❌ 부족'}")
    assert total_ep >= 500


# ── 2. Place 이벤트 밀도 ─────────────────────────────────────────────────────

@pytest.mark.parametrize("place_rate_pct", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_place_events_per_batch(place_rate_pct):
    """place_rate별 배치당 place 이벤트 수 계산.

    이 수가 PLACE_SIGNAL_MIN(5)개 이상이어야 PPO가 transport를 학습할 수 있음.
    """
    ep_per_batch  = NUM_ENVS * (NUM_STEPS_PER_ENV / EPISODE_STEPS)
    place_events  = ep_per_batch * (place_rate_pct / 100.0)
    ok = place_events >= PLACE_SIGNAL_MIN
    print(f"  place_rate={place_rate_pct:5.1f}% → {place_events:6.1f} events/batch  "
          f"{'✅' if ok else '❌ (<5, 신호 부족)'}")
    if place_rate_pct >= 2.0:
        assert ok, f"place_rate={place_rate_pct}%에서 신호 부족 — 이건 설계 오류"


def test_minimum_place_rate_for_learning():
    """학습 가능한 최소 place_rate 계산."""
    ep_per_batch = NUM_ENVS * (NUM_STEPS_PER_ENV / EPISODE_STEPS)
    min_rate_pct = PLACE_SIGNAL_MIN / ep_per_batch * 100
    print(f"\n[최소 place_rate]")
    print(f"  배치당 에피소드  : {ep_per_batch:.0f}")
    print(f"  최소 필요 rate   : {min_rate_pct:.3f}% (= {PLACE_SIGNAL_MIN}개 이벤트)")
    print(f"  현재 실측 rate   : ~1.0%  → {1.0/min_rate_pct:.0f}x 이상 충분")
    # 설정이 말이 되는지 확인
    assert min_rate_pct < 0.5, "배치가 너무 작거나 에피소드가 너무 길다"


# ── 3. Return 분포 & Advantage 품질 ──────────────────────────────────────────

def compute_episode_return(dist_box_goal: float, place: bool,
                            n_steps_to_place: int | None = None) -> float:
    """에피소드 누적 return 계산 (discount 없이, 단순 합산).

    Args:
        dist_box_goal: grasped 동안 평균 box-goal 거리 (m)
        place: True이면 마지막에 place 달성
        n_steps_to_place: place까지 걸린 step 수 (None이면 EPISODE_STEPS)
    """
    n = n_steps_to_place if n_steps_to_place else EPISODE_STEPS
    return_val = (
        REW_GRASP                          # 잡기 보너스 (한 번)
        + (-REW_GOAL_DIST * dist_box_goal  # 거리 패널티 (매 step)
           + REW_TIME                      # 시간 패널티 (매 step)
          ) * n
        + (REW_PLACE if place else 0.0)   # place 보너스
    )
    return return_val


def test_return_timeout_vs_place():
    """timeout과 place 에피소드의 return 차이 — 이것이 advantage signal."""
    r_timeout = compute_episode_return(dist_box_goal=0.70, place=False)
    r_place_fast = compute_episode_return(dist_box_goal=0.35, place=True, n_steps_to_place=100)
    r_place_slow = compute_episode_return(dist_box_goal=0.50, place=True, n_steps_to_place=160)

    advantage_fast = r_place_fast - r_timeout
    advantage_slow = r_place_slow - r_timeout

    print(f"\n[Return 분포]")
    print(f"  timeout (0.70m, {EPISODE_STEPS}step)    : {r_timeout:+.1f}")
    print(f"  place 빠름 (0.35m avg, 100step) : {r_place_fast:+.1f}")
    print(f"  place 느림 (0.50m avg, 160step) : {r_place_slow:+.1f}")
    print(f"\n[Advantage — transport가 얼마나 더 나은가]")
    print(f"  빠른 place advantage  : {advantage_fast:+.1f}")
    print(f"  느린 place advantage  : {advantage_slow:+.1f}")
    print(f"  판정: advantage > 0이면 transport가 timeout보다 이득 → PPO가 배울 수 있음")

    assert advantage_fast > 0, "빠른 place이 timeout보다 return이 낮음 — 보상 설계 오류"
    assert advantage_slow > 0, "느린 place이 timeout보다 return이 낮음 — 보상 설계 오류"


def test_return_scale():
    """Return 절대값 범위 — VF가 추정해야 하는 값의 스케일."""
    r_worst = compute_episode_return(dist_box_goal=0.70, place=False)
    r_best  = compute_episode_return(dist_box_goal=0.10, place=True, n_steps_to_place=50)
    span    = r_best - r_worst

    print(f"\n[Return 스케일]")
    print(f"  최악 (timeout, 0.70m) : {r_worst:+.1f}")
    print(f"  최선 (place, 0.10m)   : {r_best:+.1f}")
    print(f"  범위 (span)            : {span:.1f}")
    print(f"  VF가 이 범위를 학습해야 advantage가 의미 있음")
    print(f"  span이 너무 크면({span:.0f}) VF error가 크고 advantage가 노이즈에 묻힘")

    # span이 너무 크면 VF 학습 어려움 — 2000 이하가 안정적
    assert span < 2000, f"return span {span:.0f} 너무 큼 — 보상 스케일 조정 필요"


# ── 4. Entropy 안전성 진단 ────────────────────────────────────────────────────

@pytest.mark.parametrize("surrogate_loss,std", [
    (-0.0007, 1.22),   # entropy실험 iter 142 실측값 (entropy_coef=0.003)
    (-0.0012, 1.17),   # 이전 런 실측값 (entropy_coef=0.003)
    ( 0.0050, 0.80),   # 건강한 훈련 목표값 (entropy_coef=0.0001이면 안전)
    ( 0.0200, 0.50),   # Teacher 수준
])
def test_entropy_dominance_diagnosis(surrogate_loss, std):
    """실측 surrogate_loss, std로 entropy 지배 여부 진단.

    entropy_gradient = entropy_coef × H ≈ entropy_coef × n_dim × log(std)
    policy_gradient  ≈ surrogate_loss

    entropy_gradient > policy_gradient → std 계속 성장 → 훈련 실패
    """
    n_dim = N_ARM_JOINTS + 2  # 9D action space
    entropy_per_dim = 0.5 * math.log(2 * math.pi * math.e * std ** 2)
    entropy_total   = n_dim * entropy_per_dim
    entropy_grad    = ENTROPY_COEF * entropy_total  # entropy loss 기여
    surrogate_abs   = abs(surrogate_loss)
    dominated       = entropy_grad > surrogate_abs

    print(f"\n  surrogate={surrogate_loss:+.4f}, std={std:.2f} →  "
          f"entropy_grad={entropy_grad:.4f}, ratio={entropy_grad/(surrogate_abs+1e-8):.1f}x  "
          f"{'❌ ENTROPY 지배 (std↑)' if dominated else '✅ 안전'}")

    # 건강한 훈련(surrogate≥0.005)에서는 entropy가 지배하면 안 됨
    if surrogate_loss >= SURROGATE_ALIVE:
        assert not dominated, (
            f"surrogate={surrogate_loss}에서도 entropy 지배 — entropy_coef를 낮춰야 함"
        )


def test_entropy_safety_threshold():
    """surrogate_loss가 최소 얼마여야 entropy_coef=0.003이 안전한가."""
    n_dim = N_ARM_JOINTS + 2
    # std=1.0에서의 entropy
    entropy_at_std1 = n_dim * 0.5 * math.log(2 * math.pi * math.e)
    entropy_grad    = ENTROPY_COEF * entropy_at_std1
    min_safe_surrogate = entropy_grad / ENTROPY_RATIO_MAX

    print(f"\n[Entropy 안전 임계값]")
    print(f"  entropy_coef         : {ENTROPY_COEF}")
    print(f"  entropy (std=1.0)    : {entropy_at_std1:.3f}")
    print(f"  entropy gradient     : {entropy_grad:.4f}")
    print(f"  surrogate 최소 필요값 : {min_safe_surrogate:.4f}")
    print(f"  현재 실측 surrogate  : ~0.001 → "
          f"{'❌ 부족' if 0.001 < min_safe_surrogate else '✅ 충분'}")
    print(f"  → surrogate > {min_safe_surrogate:.4f} 유지해야 entropy 억제 가능")


# ── 5. Transport 물리 가능성 ──────────────────────────────────────────────────

def test_transport_physically_possible():
    """EE가 3초(180step) 안에 0.3m 이동 가능한가.

    보수적 추정: Franka 링크 길이 L≈0.5m, Jacobian≈0.3m/rad (평균)
    EE 속도 = Jacobian × 관절 속도 ≈ 0.3 × (arm_delta_scale × std)
    """
    jacobian_approx   = 0.3          # m/rad (보수적 추정)
    joint_delta_std   = ARM_DELTA_SCALE * INIT_NOISE_STD  # 0.05 m/step
    ee_speed_per_step = jacobian_approx * joint_delta_std * N_ARM_JOINTS ** 0.5  # RMS

    # 랜덤 워크: N step에서 기대 이동 거리 = speed × sqrt(N) (diffusion)
    # 최적 이동: N step에서 최대 = speed × N (직선)
    max_transport     = ee_speed_per_step * EPISODE_STEPS   # 직선 이동 최대
    rw_transport      = ee_speed_per_step * math.sqrt(EPISODE_STEPS)  # 랜덤워크 기대

    target_dist       = 0.30  # PLACE_GOALS 기준 평균 이동 거리

    print(f"\n[Transport 물리 가능성]")
    print(f"  EE 속도 추정 (RMS)   : {ee_speed_per_step:.4f} m/step")
    print(f"  최적 이동 (직선)      : {max_transport:.2f} m  {'✅' if max_transport > target_dist else '❌'}")
    print(f"  랜덤워크 기대 이동    : {rw_transport:.3f} m  (목표 {target_dist}m 대비 "
          f"{'✅ 달성 가능' if rw_transport > target_dist else '❌ 불가 — 학습 필수'})")
    print(f"  결론: 학습 없이 랜덤워크로는 {'달성 가능' if rw_transport > target_dist else '불가'}. "
          f"PPO가 방향 학습해야 place 달성.")

    # 직선 이동은 반드시 가능해야 함 (학습 후 달성 목표)
    assert max_transport > target_dist, (
        f"최적 이동({max_transport:.2f}m) < 목표({target_dist}m) — "
        "episode_length 또는 action_scale 부족"
    )


def test_grasp_steps_estimate():
    """평균적으로 grasping까지 몇 step 걸리는가 (transport 가능 step 계산)."""
    # 실측: dist_ee_box ≈ 0.09m after grasp, approach 보상으로 빠르게 수렴
    # 현재 로그: mean dist_ee_box=0.09m → 이미 grasped 상태
    # grasp까지 보통 30-50 step (추정)
    grasp_steps_estimate = 40
    transport_steps      = EPISODE_STEPS - grasp_steps_estimate

    target_dist = 0.30
    jacobian    = 0.3
    joint_std   = ARM_DELTA_SCALE * INIT_NOISE_STD
    ee_speed    = jacobian * joint_std * N_ARM_JOINTS ** 0.5
    max_transport = ee_speed * transport_steps

    print(f"\n[Grasp 후 Transport 가능 시간]")
    print(f"  전체 에피소드        : {EPISODE_STEPS} step ({EPISODE_LENGTH_S}s)")
    print(f"  grasp 소요 (추정)    : {grasp_steps_estimate} step")
    print(f"  transport 가능       : {transport_steps} step")
    print(f"  최대 이동 가능 거리   : {max_transport:.2f} m  (목표 {target_dist}m)")
    assert max_transport > target_dist


# ── 6. Asymmetric AC 차원 검증 ───────────────────────────────────────────────

def test_obs_dims_teacher():
    """Teacher 관측 차원 = 30."""
    # box_rel(3) + box_quat(4) + box_mass(1) + gripper(1) + goal_rel(3) + jpos(9) + jvel(9)
    dims = {"box_rel": 3, "box_quat": 4, "box_mass": 1,
            "gripper": 1, "goal_rel": 3, "jpos": 9, "jvel": 9}
    total = sum(dims.values())
    print(f"\n[Teacher obs 차원]")
    for k, v in dims.items():
        print(f"  {k:12s}: {v}D")
    print(f"  합계              : {total}D  (기대: {OBS_DIM}D)")
    assert total == OBS_DIM, f"obs {total}D ≠ {OBS_DIM}D"


def test_critic_advantage_benefit():
    """노이즈 없는 critic이 advantage 추정에 얼마나 유리한가.

    Student: noisy_box_rel에 σ=1~6cm 노이즈 → VF 추정 시 box 위치 불확실
    Teacher: 정확한 box_rel → VF 추정 정확

    거리 노이즈가 VF 추정 오류로 이어지는 스케일 계산.
    """
    noise_sigma_max = 0.06   # m (최대 노이즈)
    # VF 추정 오류: noisy dist → noisy return
    # return = -rew_goal_dist * dist * N_steps
    # ΔV ≈ rew_goal_dist * σ * N_steps
    vf_error_from_noise = REW_GOAL_DIST * noise_sigma_max * EPISODE_STEPS
    place_advantage     = REW_PLACE  # transport가 주는 advantage 상한

    print(f"\n[노이즈로 인한 VF 오류 vs Advantage 크기]")
    print(f"  노이즈 σ_max         : {noise_sigma_max*100:.0f}cm")
    print(f"  VF 오류 (최대)        : {vf_error_from_noise:.1f}")
    print(f"  Place advantage       : {place_advantage:.1f}")
    print(f"  SNR                   : {place_advantage/vf_error_from_noise:.2f}x")
    print(f"  {'✅ SNR > 3: 노이즈에도 신호 명확' if place_advantage/vf_error_from_noise > 3 else '❌ SNR 낮음: 노이즈가 신호를 가림'}")
    print(f"  → Asymmetric critic으로 이 오류를 제거하면 advantage 품질 향상")

    # SNR이 1 이상이어야 학습 가능
    snr = place_advantage / (vf_error_from_noise + 1e-6)
    assert snr > 1.0, f"SNR={snr:.2f} < 1 — noisy obs로는 VF 학습 불가"


# ── 7. 통합 진단: 실측값으로 현재 훈련 상태 평가 ─────────────────────────────

@pytest.mark.parametrize("label,surrogate,std,place_rate_pct,ep_len", [
    ("이전런 2169iter (실패)",  -0.0002, 0.13, 7.59, 278.21),
    ("entropy실험 147iter (실패)", -0.0007, 1.17, 1.23, 890.09),
    ("현재런 91iter (실패)",   -0.0012, 1.24, 0.96, 177.27),
    ("목표 건강 상태",           0.0200, 0.50,15.00, 120.00),
])
def test_training_state_diagnosis(label, surrogate, std, place_rate_pct, ep_len):
    """실측 훈련 로그값으로 건강 상태 자동 진단."""
    entropy_per_dim = 0.5 * math.log(2 * math.pi * math.e * std ** 2)
    entropy_total   = (N_ARM_JOINTS + 2) * entropy_per_dim
    entropy_grad    = ENTROPY_COEF * entropy_total

    surrogate_dead    = abs(surrogate) < SURROGATE_ALIVE
    entropy_dominates = entropy_grad > abs(surrogate)
    std_collapsed     = std < STD_OK_MIN
    std_exploded      = std > STD_OK_MAX
    ep_len_ok         = ep_len <= EPISODE_STEPS * 1.05  # 5% 마진

    issues = []
    if surrogate_dead:    issues.append(f"gradient 사망(surrogate={surrogate:.4f})")
    if entropy_dominates: issues.append(f"entropy 지배({entropy_grad:.3f}>{abs(surrogate):.3f})")
    if std_collapsed:     issues.append(f"std 붕괴({std:.2f})")
    if std_exploded:      issues.append(f"std 폭발({std:.2f})")
    if not ep_len_ok:     issues.append(f"ep_len 타임아웃({ep_len:.0f}/{EPISODE_STEPS})")

    healthy = len(issues) == 0

    print(f"\n  [{label}]")
    print(f"    surrogate={surrogate:+.4f}  std={std:.2f}  "
          f"place_rate={place_rate_pct:.2f}%  ep_len={ep_len:.0f}")
    print(f"    entropy_grad={entropy_grad:.4f}")
    if healthy:
        print(f"    → ✅ 건강")
    else:
        print(f"    → ❌ 문제: {', '.join(issues)}")

    # 목표 건강 상태는 반드시 pass
    if label.startswith("목표"):
        assert healthy, f"목표 상태가 진단에서 실패: {issues}"
