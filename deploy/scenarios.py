"""S1~S6 multi-robot scenarios — single source of truth.

The same spawn/goal coordinates drive BOTH:
  - Isaac spawn  (deploy/isaac/isaac_multi_robot_ros2.py --scenario S1)
  - Nav2 goals   (deploy/nav2/scenario_goals.py S1)

so the robot that spawns at a scenario's spawn point is sent that scenario's
goal. Originally these scenarios were evaluated with a pure-RL policy in a small
±4 m arena centred on the origin (training/multi_robot/eval_scenarios.py). Here
we run the SAME interaction geometry on sensor-equipped iw_hub robots driven by
Nav2 (lidar costmap avoidance) — the "improved" version.

Arena coords below are in the original ±4 m frame. world_xy() shifts them into a
clear, open patch of the warehouse (south floor, no shelves) so Nav2 has room.
Tune ARENA_ORIGIN if the patch overlaps a shelf in your map.
"""
from __future__ import annotations

# Centre of the scenario arena in warehouse world coords. South open floor
# (shelves start at y~8.5); ±4 m arena -> x[-11,-3], y[-2,6], clear of shelves.
ARENA_ORIGIN = (-7.0, 2.0)

# name -> (description, spawns[(x,y)...], goals[(x,y)...], extra)
# robot i spawns at spawns[i] and is sent to goals[i]. len(spawns)==len(goals).
SCENARIOS: dict[str, dict] = {
    "S1": {
        "desc": "정면 교행 — 좁은 복도 양방향 진입",
        "spawns": [(-3.0, 0.0), (3.0, 0.0), (0.0, 3.5)],
        "goals":  [(3.0, 0.0), (-3.0, 0.0), (0.0, -3.5)],
    },
    "S2": {
        "desc": "3-way 교착 — 삼각 대치",
        "spawns": [(0.0, 2.0), (-1.73, -1.0), (1.73, -1.0)],
        "goals":  [(0.0, -2.0), (1.73, 1.0), (-1.73, 1.0)],
    },
    "S3": {
        "desc": "배터리 우선순위 — 낮은 배터리 로봇 우선 통과",
        "spawns": [(-2.5, 0.0), (-2.5, 0.8), (3.0, 0.8)],
        "goals":  [(3.0, 0.0), (3.0, 0.8), (-2.5, 0.8)],
        "priority": [0, 1, 2],   # lower index = higher priority (battery low)
    },
    "S4": {
        "desc": "동일 목표 경쟁 — 두 로봇이 같은 지점 도달 경쟁",
        "spawns": [(-2.0, -1.5), (-2.0, 1.5), (3.0, 0.0)],
        "goals":  [(2.0, 0.0), (2.0, 0.0), (-3.0, 0.0)],
        # robots 0,1 share goal (2,0): physically one point. Driver detects the
        # conflict and reassigns/queues the loser (see scenario_goals.py).
        "shared_goal": [0, 1],
    },
    "S5": {
        "desc": "혼잡 통로 — y=0 중앙 통로 3방향 교차",
        "spawns": [(-4.0, 0.0), (4.0, 0.3), (0.0, 4.0)],
        "goals":  [(4.0, 0.0), (-4.0, 0.3), (0.0, -4.0)],
    },
    "S6": {
        "desc": "동적 장애물 회피 — 평행 직진 중 장애물 출현, 우회",
        "spawns": [(-4.0, -2.0), (-4.0, 0.0), (-4.0, 2.0)],
        "goals":  [(4.0, -2.0), (4.0, 0.0), (4.0, 2.0)],
        "obstacles": [(0.0, -2.0), (0.0, 0.0)],   # spawned in front of robots 0,1
    },
}

# Robot names match the multi-robot Isaac scene (R1..R3).
ROBOT_NAMES = ["R1", "R2", "R3"]


def world_xy(ax: float, ay: float) -> tuple[float, float]:
    """Arena (±4 m) coords -> warehouse world coords."""
    return (ARENA_ORIGIN[0] + ax, ARENA_ORIGIN[1] + ay)


def robots_for(scenario: str) -> list[tuple[str, float, float]]:
    """[(name, world_x, world_y), ...] spawn list for the Isaac scene."""
    s = SCENARIOS[scenario]
    out = []
    for i, (ax, ay) in enumerate(s["spawns"]):
        wx, wy = world_xy(ax, ay)
        out.append((ROBOT_NAMES[i], wx, wy))
    return out


def goals_for(scenario: str) -> list[tuple[str, float, float]]:
    """[(name, world_x, world_y), ...] goal list for the Nav2 driver."""
    s = SCENARIOS[scenario]
    out = []
    for i, (ax, ay) in enumerate(s["goals"]):
        wx, wy = world_xy(ax, ay)
        out.append((ROBOT_NAMES[i], wx, wy))
    return out


def obstacles_for(scenario: str) -> list[tuple[float, float]]:
    """World-coord obstacle positions (S6), or []."""
    s = SCENARIOS[scenario]
    return [world_xy(ox, oy) for ox, oy in s.get("obstacles", [])]


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else None
    for k, v in SCENARIOS.items():
        if name and k != name:
            continue
        print(f"\n{k}: {v['desc']}")
        for nm, x, y in robots_for(k):
            print(f"  spawn {nm}: ({x:.2f}, {y:.2f})")
        for nm, x, y in goals_for(k):
            print(f"  goal  {nm}: ({x:.2f}, {y:.2f})")
        if v.get("obstacles"):
            print(f"  obstacles: {obstacles_for(k)}")
