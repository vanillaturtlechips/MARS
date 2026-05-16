"""Markov Potential Game 보상 함수.

참고: Yan & Liu, arXiv:2503.22867 (2025)
  Theorem 5: rᵢ = α·rᵢ^self + β·Σⱼ≠ᵢ rᵢⱼ  이면 게임이 자동으로 MPG
  rᵢⱼ = rⱼᵢ (대칭) 조건 충족 → Nash Equilibrium 존재 및 수렴 보장
"""

from __future__ import annotations

import torch


# 기본 가중치 (PROJECT_DESIGN.md 기준)
ALPHA = 1.0   # 효율 가중치
BETA  = 1.5   # 안전 가중치
EPS   = 1e-5  # 분모 0 방지


MAX_DIST = 4.0   # goal_range (m) — 정규화 기준 반경


def self_reward(
    pos: torch.Tensor,          # (N, 2) 현재 위치
    goal: torch.Tensor,         # (N, 2) 목표 위치
    goal_radius: float = 0.35,
    time_step: torch.Tensor | None = None,  # (N,) 경과 스텝
    rew_goal: float = 10.0,
    rew_time: float = -0.01,
) -> torch.Tensor:
    """rᵢ^self — 개별 목표 추적 보상 (Theorem 3).

    논문 공식 rᵢ^self = -||posᵢ - goalᵢ||² 를 MAX_DIST²로 정규화.
    dist_sq_norm ∈ [0, 1] → Theorem 5 Nash Equilibrium 보장 유지.
    """
    dist_sq      = ((pos - goal) ** 2).sum(dim=1)          # (N,)
    dist_sq_norm = dist_sq / (MAX_DIST ** 2)               # [0, 1]
    reached      = (dist_sq.sqrt() < goal_radius).float()

    r = -dist_sq_norm + rew_goal * reached
    if time_step is not None:
        r = r + rew_time * time_step.float()
    return r                                                # (N,)


SAFE_DIST = 1.2   # m — 이 거리 밖에서는 pairwise 페널티 없음


def pairwise_reward(
    pos_i: torch.Tensor,   # (N, 2) 로봇 i 위치
    pos_j: torch.Tensor,   # (N, 2) 로봇 j 위치
    eps: float = EPS,
) -> torch.Tensor:
    """rᵢⱼ — 충돌 회피 쌍별 보상 (Danger Zone 마스킹 적용).

    SAFE_DIST 이내일 때만 반발력 작동. 멀리 있으면 0으로 마스킹.
    → 로봇이 먼 거리에서는 pairwise 노이즈 없이 goal만 따라감.
    APF(Artificial Potential Field)의 표준 cutoff 기법.
    """
    dist_sq = ((pos_i - pos_j) ** 2).sum(dim=1)          # (N,)
    dist    = (dist_sq + eps).sqrt()                      # (N,)
    raw     = -1.0 / dist                                 # (N,)
    in_danger = (dist < SAFE_DIST).float()
    return (raw * in_danger).clamp(min=-5.0)              # 충돌 시 -∞ 방지


def mpg_reward(
    positions: torch.Tensor,    # (N, n_robots, 2) 모든 로봇 위치
    goals: torch.Tensor,        # (N, n_robots, 2) 모든 로봇 목표
    robot_idx: int,             # 이 로봇의 인덱스
    alpha: float = ALPHA,
    beta: float = BETA,
    goal_radius: float = 0.35,
    time_step: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """rᵢ = α·rᵢ^self + β·Σⱼ≠ᵢ rᵢⱼ — 전체 MPG 보상.

    Args:
        positions: 모든 env의 모든 로봇 위치
        goals:     모든 env의 모든 로봇 목표
        robot_idx: 보상 계산 대상 로봇 인덱스

    Returns:
        (N,) 텐서 — 각 env에서 robot_idx 로봇의 MPG 보상
    """
    pos_i  = positions[:, robot_idx]   # (N, 2)
    goal_i = goals[:, robot_idx]       # (N, 2)

    r_self = self_reward(pos_i, goal_i, goal_radius, time_step)

    n_robots = positions.shape[1]
    r_pair = torch.zeros_like(r_self)
    for j in range(n_robots):
        if j == robot_idx:
            continue
        r_pair = r_pair + pairwise_reward(pos_i, positions[:, j], eps)

    return alpha * r_self + beta * r_pair   # (N,)


def all_robots_mpg_reward(
    positions: torch.Tensor,    # (N, n_robots, 2)
    goals: torch.Tensor,        # (N, n_robots, 2)
    alpha: float = ALPHA,
    beta: float = BETA,
    goal_radius: float = 0.35,
    time_step: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """모든 로봇의 MPG 보상을 한 번에 계산.

    Returns:
        (N, n_robots) 텐서
    """
    n_robots = positions.shape[1]
    rewards = torch.stack([
        mpg_reward(positions, goals, i, alpha, beta, goal_radius, time_step, eps)
        for i in range(n_robots)
    ], dim=1)
    return rewards
