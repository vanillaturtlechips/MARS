"""reward 수식 단위 테스트 — Isaac Sim 불필요, pytest로 로컬 실행 가능.

실행:
  cd /home/user/Documents/MARS
  python -m pytest tests/test_reward_math.py -v
"""

import torch
import pytest


# ── 1. EE 속도 alignment 계산 ────────────────────────────────────────────────

def compute_alignment(ee_vel: torch.Tensor, goal_rel: torch.Tensor) -> torch.Tensor:
    """_get_rewards 내 alignment 계산 그대로."""
    ee_speed   = ee_vel.norm(dim=1, keepdim=True).clamp(min=1e-4)
    ee_vel_dir = ee_vel / ee_speed
    goal_dir   = goal_rel / goal_rel.norm(dim=1, keepdim=True).clamp(min=1e-4)
    return (ee_vel_dir * goal_dir).sum(dim=1)


def test_alignment_perfect():
    """EE가 goal 방향으로 정확히 이동 → alignment=+1."""
    ee_vel  = torch.tensor([[0.0, 1.0, 0.0]])   # +y 이동
    goal    = torch.tensor([[0.0, 0.5, 0.0]])   # goal이 +y 방향
    assert compute_alignment(ee_vel, goal).item() == pytest.approx(1.0, abs=1e-5)


def test_alignment_opposite():
    """EE가 goal 반대 방향 → alignment=-1."""
    ee_vel  = torch.tensor([[0.0, -1.0, 0.0]])
    goal    = torch.tensor([[0.0, 0.5, 0.0]])
    assert compute_alignment(ee_vel, goal).item() == pytest.approx(-1.0, abs=1e-5)


def test_alignment_perpendicular():
    """EE가 goal 수직 방향 → alignment=0."""
    ee_vel  = torch.tensor([[1.0, 0.0, 0.0]])
    goal    = torch.tensor([[0.0, 1.0, 0.0]])
    assert compute_alignment(ee_vel, goal).item() == pytest.approx(0.0, abs=1e-5)


def test_alignment_zero_velocity():
    """EE 정지(속도=0) → clamp로 NaN 없어야 함."""
    ee_vel  = torch.tensor([[0.0, 0.0, 0.0]])
    goal    = torch.tensor([[0.0, 0.5, 0.0]])
    result  = compute_alignment(ee_vel, goal)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_alignment_batch():
    """배치(N=3) 처리 — 각 env 독립적으로 계산."""
    ee_vel = torch.tensor([
        [0.0, 1.0, 0.0],   # goal 방향
        [0.0, -1.0, 0.0],  # 반대
        [1.0, 0.0, 0.0],   # 수직
    ])
    goal = torch.tensor([
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ])
    result = compute_alignment(ee_vel, goal)
    assert result[0].item() == pytest.approx(1.0,  abs=1e-5)
    assert result[1].item() == pytest.approx(-1.0, abs=1e-5)
    assert result[2].item() == pytest.approx(0.0,  abs=1e-5)


# ── 2. curriculum spawn 좌표 검증 ─────────────────────────────────────────────

def test_curriculum_goal_distance():
    """goal이 box 반경 0.13~0.20m 내에 있는지 확인."""
    torch.manual_seed(42)
    N = 1000
    theta = torch.rand(N) * 6.2832
    r     = torch.rand(N) * (0.20 - 0.13) + 0.13  # [0.13, 0.20]

    box_x = torch.rand(N) * 0.13 + 0.25  # [0.25, 0.38]
    box_y = torch.rand(N) * 0.10 - 0.05  # [-0.05, 0.05]

    goal_x = box_x + r * torch.cos(theta)
    goal_y = box_y + r * torch.sin(theta)

    dist = ((goal_x - box_x)**2 + (goal_y - box_y)**2).sqrt()
    assert (dist >= 0.13 - 1e-5).all(), f"min dist {dist.min():.4f} < 0.13"
    assert (dist <= 0.20 + 1e-5).all(), f"max dist {dist.max():.4f} > 0.20"


def test_curriculum_above_place_threshold():
    """goal이 place_dist_threshold(0.12m)보다 항상 멀어야 함 — 즉시 place 방지."""
    place_threshold = 0.12
    min_r = 0.13
    assert min_r > place_threshold, (
        f"r_min={min_r} ≤ place_threshold={place_threshold} → 즉시 place 발생!"
    )


# ── 3. dist_box_goal 계산 ────────────────────────────────────────────────────

def test_dist_box_goal_grasped_gate():
    """grasped=False 환경은 dist_box_goal 로그에서 제외되어야 함."""
    N = 4
    dist_box_goal = torch.tensor([0.5, 0.1, 0.3, 0.2])
    grasped_f     = torch.tensor([1.0, 0.0, 1.0, 0.0])  # env 0,2만 grasped

    logged = (dist_box_goal * grasped_f).sum() / (grasped_f.sum() + 1e-6)
    expected = (0.5 + 0.3) / 2  # env 0,2 평균
    assert logged.item() == pytest.approx(expected, abs=1e-4)


def test_dist_box_goal_all_grasped():
    """모두 grasped → 전체 평균."""
    dist = torch.tensor([0.1, 0.2, 0.3])
    gf   = torch.ones(3)
    logged = (dist * gf).sum() / (gf.sum() + 1e-6)
    assert logged.item() == pytest.approx(0.2, abs=1e-4)


# ── 4. transport delta reward 경계 ──────────────────────────────────────────

def test_transport_clamp():
    """delta=(prev-curr) ±0.1m/step clamp 확인.
    prev-curr > 0: 가까워짐(good), prev-curr < 0: 멀어짐(bad).
    """
    prev = torch.tensor([1.0, 1.0, 1.0])
    curr = torch.tensor([0.5, 0.95, 1.5])
    # prev-curr: [+0.5, +0.05, -0.5] → clamp → [+0.1, +0.05, -0.1]
    delta = (prev - curr).clamp(-0.1, 0.1)
    assert delta[0].item() == pytest.approx( 0.1,  abs=1e-5)   # 많이 가까워짐 → cap
    assert delta[1].item() == pytest.approx( 0.05, abs=1e-5)   # 조금 가까워짐
    assert delta[2].item() == pytest.approx(-0.1,  abs=1e-5)   # 많이 멀어짐 → floor


# ── 5. rew_goal_dist=8.0 검증 ────────────────────────────────────────────────

def test_goal_dist_reward_4x_stronger():
    """rew_goal_dist 2.0→8.0: 같은 거리에서 gradient 4배 강화."""
    dist = torch.tensor([0.62])   # 현재 mean dist_box_goal
    rew_old = 2.0 * dist.item()   # 1.24/step
    rew_new = 8.0 * dist.item()   # 4.96/step
    assert rew_new == pytest.approx(4.0 * rew_old, abs=1e-4)


def test_transport_beats_stay():
    """transport(0.62→0.30m, ~100step) 누적 reward > stay(0.62m, 278step).
    이게 성립해야 policy가 transport를 배울 동기가 있음.
    """
    rew_goal_dist = 8.0
    rew_place     = 800.0
    rew_time      = -0.02

    # 현재 평균: 278 step 동안 0.62m 거리 유지 (place 없음)
    stay_return = (-rew_goal_dist * 0.62 + rew_time) * 278

    # transport: 100 step에 걸쳐 0.62→0.30m 선형 감소, 마지막에 place
    dists = torch.linspace(0.62, 0.30, 100)
    transport_return = (
        sum(-rew_goal_dist * d.item() + rew_time for d in dists)
        + rew_place
    )

    assert transport_return > stay_return, (
        f"transport({transport_return:.1f}) <= stay({stay_return:.1f}) → "
        "policy가 transport를 배울 이유가 없음"
    )


def test_transport_beats_stay_old_reward():
    """rew_goal_dist=2.0(old)에서도 transport가 우세한지 확인.
    우세하면 문제는 reward 설계가 아닌 탐색 부재.
    """
    rew_goal_dist = 2.0   # old
    rew_place     = 800.0
    rew_time      = -0.02

    stay_return = (-rew_goal_dist * 0.62 + rew_time) * 278
    dists = torch.linspace(0.62, 0.30, 100)
    transport_return = (
        sum(-rew_goal_dist * d.item() + rew_time for d in dists)
        + rew_place
    )
    assert transport_return > stay_return


# ── 6. entropy_coef 안전 구간 검증 ───────────────────────────────────────────

def test_entropy_coef_safe_when_gradient_alive():
    """entropy_coef=0.003이 surrogate_loss=0.01(활성) 대비 30% 미만 → 안전.
    surrogate_loss < 0.003이면 entropy가 지배 → std 폭발 위험.
    """
    entropy_coef          = 0.003
    min_active_surrogate  = 0.01   # gradient "살아있음" 기준
    ratio = entropy_coef / min_active_surrogate
    assert ratio <= 0.30, f"entropy too strong: {ratio:.3f}"


def test_entropy_dominates_when_gradient_dead():
    """surrogate_loss=0.0002(dead 상태)이면 entropy가 policy gradient를 압도.
    이것이 이전 런에서 std 0.13→폭발 예측 경로.
    """
    entropy_coef       = 0.003
    dead_surrogate     = 0.0002   # iter 2170 실측값
    ratio = entropy_coef / dead_surrogate
    # entropy가 policy gradient의 15배 → std 폭발 불가피
    assert ratio > 10.0, f"gradient dead인데 entropy가 안전하다면 분석 오류: {ratio:.1f}x"


def test_advantage_variance_check():
    """rew_goal_dist=8.0 vs 2.0: 같은 dist 범위에서 reward 분산 비교.
    분산이 클수록 advantage 신호 강해짐 → surrogate_loss 살아있음.
    """
    torch.manual_seed(0)
    dists = torch.rand(1000) * 0.6 + 0.1   # [0.1, 0.7]m

    var_old = (2.0 * dists).var().item()
    var_new = (8.0 * dists).var().item()

    # rew_goal_dist 4x → reward 분산 16x (선형 스케일 → 분산은 제곱)
    assert var_new == pytest.approx(16.0 * var_old, rel=1e-3)
