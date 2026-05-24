"""경로 검증 에이전트 — Propose-then-Commit 패턴의 검증 단계.

로봇 목표 배정 전 경로 충돌을 사전 검사한다.
A* 경로는 추후 구현; 현재는 직선 경로 충돌 체크 + 선반 침범 검사.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 창고 물리 상수
ROBOT_RADIUS    = 0.3   # m
SHELF_WIDTH     = 1.0   # m (y 방향)
SHELF_DEPTH     = 0.5   # m (x 방향)
SAFE_SEPARATION = 1.5   # m (로봇 간 안전 거리)

# 선반 AABB (x_center, y_center) → 실제 창고 기준
SHELF_POSITIONS: list[tuple[float, float]] = [
    (-3.5,  2.0), (-3.5,  0.0), (-3.5, -2.0),
    ( 0.0,  2.0), ( 0.0,  0.0), ( 0.0, -2.0),
    ( 3.5,  2.0), ( 3.5,  0.0), ( 3.5, -2.0),
]


@dataclass
class PathCheckResult:
    valid: bool
    conflicts: list[str]
    warnings: list[str]


def _segment_min_dist(
    p1: tuple[float, float], p2: tuple[float, float],
    q1: tuple[float, float], q2: tuple[float, float],
) -> float:
    """두 선분 사이 최소 거리 (2D)."""
    def _pt_seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        ab = (b[0] - a[0], b[1] - a[1])
        ap = (p[0] - a[0], p[1] - a[1])
        t = max(0.0, min(1.0, (ap[0]*ab[0] + ap[1]*ab[1]) / (ab[0]**2 + ab[1]**2 + 1e-9)))
        proj = (a[0] + t*ab[0], a[1] + t*ab[1])
        return math.hypot(p[0] - proj[0], p[1] - proj[1])

    candidates = [
        _pt_seg_dist(p1, q1, q2),
        _pt_seg_dist(p2, q1, q2),
        _pt_seg_dist(q1, p1, p2),
        _pt_seg_dist(q2, p1, p2),
    ]
    return min(candidates)


def _check_shelf_collision(
    start: tuple[float, float],
    goal:  tuple[float, float],
    robot_id: str,
) -> list[str]:
    """직선 경로가 선반 AABB를 통과하는지 검사."""
    warnings = []
    half_x = SHELF_DEPTH / 2 + ROBOT_RADIUS
    half_y = SHELF_WIDTH / 2 + ROBOT_RADIUS

    for sx, sy in SHELF_POSITIONS:
        # 선반 중심과 직선 경로 사이 최소 거리 (점-선분)
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        t = max(0.0, min(1.0, ((sx - start[0])*dx + (sy - start[1])*dy) / (length**2)))
        cx = start[0] + t*dx - sx
        cy = start[1] + t*dy - sy
        if abs(cx) < half_x and abs(cy) < half_y:
            warnings.append(
                f"{robot_id} 경로가 선반 ({sx:.1f},{sy:.1f}) 근접 통과 "
                f"(Δx={abs(cx):.2f}m < {half_x:.2f}m)"
            )
    return warnings


def validate_assignments(
    assignments: list[dict],  # {"robot_id": str, "start_x", "start_y", "goal_x", "goal_y"}
) -> PathCheckResult:
    """목표 배정 유효성 검사 (충돌, 선반 침범)."""
    conflicts: list[str] = []
    warnings:  list[str] = []

    n = len(assignments)
    for i in range(n):
        a = assignments[i]
        start_i = (a["start_x"], a["start_y"])
        goal_i  = (a["goal_x"],  a["goal_y"])

        # 선반 침범 검사
        warnings += _check_shelf_collision(start_i, goal_i, a["robot_id"])

        # 목표 중복 검사
        for j in range(i + 1, n):
            b = assignments[j]
            goal_j = (b["goal_x"], b["goal_y"])
            if math.hypot(goal_i[0] - goal_j[0], goal_i[1] - goal_j[1]) < SAFE_SEPARATION:
                conflicts.append(
                    f"{a['robot_id']}({goal_i[0]:.1f},{goal_i[1]:.1f})와 "
                    f"{b['robot_id']}({goal_j[0]:.1f},{goal_j[1]:.1f}) 목표 근접"
                )

            # 직선 경로 교차 검사
            start_j = (b["start_x"], b["start_y"])
            d = _segment_min_dist(start_i, goal_i, start_j, goal_j)
            if d < SAFE_SEPARATION:
                warnings.append(
                    f"{a['robot_id']}↔{b['robot_id']} 경로 근접 (최소 {d:.2f}m < {SAFE_SEPARATION}m)"
                )

    return PathCheckResult(
        valid    = len(conflicts) == 0,
        conflicts= conflicts,
        warnings = warnings,
    )


def print_check_result(result: PathCheckResult) -> None:
    status = "✅ 통과" if result.valid else "❌ 충돌 감지"
    print(f"\n[PathAgent] 경로 검증: {status}")
    for c in result.conflicts:
        print(f"  [충돌] {c}")
    for w in result.warnings:
        print(f"  [경고] {w}")
    if not result.conflicts and not result.warnings:
        print("  모든 경로 안전")
