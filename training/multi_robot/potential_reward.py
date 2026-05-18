"""Markov Potential Game 보상 함수.

참고: Yan & Liu, arXiv:2503.22867 (2025)
  Theorem 5: rᵢ = α·rᵢ^self + β·Σⱼ≠ᵢ rᵢⱼ  이면 게임이 자동으로 MPG
  rᵢⱼ = rⱼᵢ (대칭) 조건 충족 → Nash Equilibrium 존재 및 수렴 보장

pairwise_reward: Delta Repulsion Potential (Ng et al. 1999 PBRS 형태)
  r_yield = -(P_rep^t - P_rep^{t-1}),  P_rep = 1/d
  멀어지면(양보) → 양수 보너스, 가까워지면 → 음수 패널티
  와리가리 합산 = 0 (제로섬) → reward hacking 불가
"""

from __future__ import annotations

import torch


ALPHA = 1.0   # 효율 가중치
BETA  = 1.5   # 안전 가중치
EPS   = 1e-5  # 분모 0 방지

MAX_DIST = 4.0   # goal_range (m) — 정규화 기준 반경

SAFE_DIST = 1.5   # m — 이 거리 밖에서는 pairwise 보상 없음 (1.2→1.5로 확장)
W_REP     = 2.0   # delta repulsion 스케일


def self_reward(
    pos: torch.Tensor,          # (N, 2)
    goal: torch.Tensor,         # (N, 2)
    goal_radius: float = 0.35,
    time_step: torch.Tensor | None = None,
    rew_goal: float = 3.0,
    rew_time: float = -0.01,
) -> torch.Tensor:
    dist_sq      = ((pos - goal) ** 2).sum(dim=1)
    dist_sq_norm = dist_sq / (MAX_DIST ** 2)
    reached      = (dist_sq.sqrt() < goal_radius).float()

    r = -dist_sq_norm + rew_goal * reached
    if time_step is not None:
        r = r + rew_time
    return r


def pairwise_reward(
    pos_i:      torch.Tensor,   # (N, 2) 현재 i 위치
    pos_j:      torch.Tensor,   # (N, 2) 현재 j 위치
    prev_pos_i: torch.Tensor,   # (N, 2) 이전 i 위치
    prev_pos_j: torch.Tensor,   # (N, 2) 이전 j 위치
    eps: float = EPS,
) -> torch.Tensor:
    """Delta Repulsion Potential — 양보하면 보너스, 접근하면 패널티.

    r = -(1/d_t - 1/d_{t-1}) * W_REP
      멀어질 때: d_t > d_{t-1} → 1/d_t < 1/d_{t-1} → r > 0 (보너스)
      가까울 때: d_t < d_{t-1} → 1/d_t > 1/d_{t-1} → r < 0 (패널티)
    SAFE_DIST 밖은 0으로 마스킹.
    """
    dist_t    = ((pos_i - pos_j)     ** 2).sum(dim=1).add(eps).sqrt()   # (N,)
    dist_prev = ((prev_pos_i - prev_pos_j) ** 2).sum(dim=1).add(eps).sqrt()

    pot_t    = 1.0 / dist_t
    pot_prev = 1.0 / dist_prev

    delta = -(pot_t - pot_prev) * W_REP   # 양보 → +, 접근 → -

    # 현재 또는 이전 스텝 중 하나라도 SAFE_DIST 이내면 작동
    in_danger = ((dist_t < SAFE_DIST) | (dist_prev < SAFE_DIST)).float()
    return delta * in_danger


def mpg_reward(
    positions:      torch.Tensor,   # (N, n_robots, 2)
    goals:          torch.Tensor,   # (N, n_robots, 2)
    prev_positions: torch.Tensor,   # (N, n_robots, 2)
    robot_idx: int,
    alpha: float = ALPHA,
    beta: float = BETA,
    goal_radius: float = 0.35,
    time_step: torch.Tensor | None = None,
    eps: float = EPS,
    rew_goal: float = 3.0,
) -> torch.Tensor:
    pos_i      = positions[:, robot_idx]
    goal_i     = goals[:, robot_idx]
    prev_pos_i = prev_positions[:, robot_idx]

    r_self = self_reward(pos_i, goal_i, goal_radius, time_step, rew_goal=rew_goal)

    n_robots = positions.shape[1]
    r_pair = torch.zeros_like(r_self)
    for j in range(n_robots):
        if j == robot_idx:
            continue
        r_pair = r_pair + pairwise_reward(
            pos_i, positions[:, j],
            prev_pos_i, prev_positions[:, j],
            eps,
        )

    return alpha * r_self + beta * r_pair


def all_robots_mpg_reward(
    positions:      torch.Tensor,   # (N, n_robots, 2)
    goals:          torch.Tensor,   # (N, n_robots, 2)
    prev_positions: torch.Tensor,   # (N, n_robots, 2)
    alpha: float = ALPHA,
    beta: float = BETA,
    goal_radius: float = 0.35,
    time_step: torch.Tensor | None = None,
    rew_goal: float = 3.0,
    eps: float = EPS,
) -> torch.Tensor:
    """모든 로봇의 MPG 보상. Returns (N, n_robots)."""
    n_robots = positions.shape[1]
    rewards = torch.stack([
        mpg_reward(positions, goals, prev_positions, i,
                   alpha, beta, goal_radius, time_step, eps, rew_goal=rew_goal)
        for i in range(n_robots)
    ], dim=1)
    return rewards
