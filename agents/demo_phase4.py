"""Phase 4 통합 데모 — Propose-then-Commit 전체 파이프라인.

자연어 임무 → Claude API 계획 → 경로 검증 → 로봇 배정

실행:
  export ANTHROPIC_API_KEY=sk-ant-...
  python agents/demo_phase4.py
  python agents/demo_phase4.py --task "A구역 재고를 출고 게이트로 이동"
  python agents/demo_phase4.py --demo      # 시나리오 3종 자동 실행
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from agents.orchestrator.orchestrator import WarehouseOrchestrator, WAREHOUSE_MAP
from agents.sub_agents.path_agent import validate_assignments, print_check_result
from agents.sub_agents.robot_agent import RobotFleet


def run_mission(orch: WarehouseOrchestrator, fleet: RobotFleet, task: str) -> bool:
    """단일 임무 실행 — Propose-then-Commit."""
    print(f"\n{'━'*60}")
    print(f"  임무: {task}")
    print(f"{'━'*60}")

    # ── 1. Claude API → 계획 생성 (Propose) ──────────────────────────
    plan = orch.plan_mission(task)
    orch.print_plan(plan)

    # ── 2. 경로 검증 (Validate) ───────────────────────────────────────
    robot_map = {r["id"]: r for r in WAREHOUSE_MAP["robots"]}
    check_input = [
        {
            "robot_id": a.robot_id,
            "start_x":  robot_map.get(a.robot_id, {}).get("x", 0.0),
            "start_y":  robot_map.get(a.robot_id, {}).get("y", 0.0),
            "goal_x":   a.goal_x,
            "goal_y":   a.goal_y,
        }
        for a in plan.assignments
    ]
    result = validate_assignments(check_input)
    print_check_result(result)

    if not result.valid:
        print(f"\n  [Orchestrator] 경로 충돌 감지 → 재계획 필요")
        return False

    # ── 3. 로봇 배정 (Commit) ─────────────────────────────────────────
    print(f"\n[Fleet] 목표 배정 실행...")
    dispatch_input = [
        {
            "robot_id": a.robot_id,
            "goal_x":   a.goal_x,
            "goal_y":   a.goal_y,
            "task":     a.task,
            "priority": a.priority,
        }
        for a in plan.assignments
    ]
    dispatch_results = fleet.dispatch(dispatch_input)

    success_count = sum(1 for ok in dispatch_results.values() if ok)
    print(f"\n  배정 완료: {success_count}/{len(dispatch_results)}대")
    print(f"\n{fleet.status_summary()}")

    return success_count == len(dispatch_results)


DEMO_SCENARIOS = [
    (
        "시나리오 1 — 일반 출고",
        "C구역 반도체칩과 센서모듈을 G1 출고 게이트로 이동해줘",
    ),
    (
        "시나리오 2 — 배터리 우선순위",
        "배터리 낮은 로봇을 먼저 충전시키고, 나머지는 B구역 완제품을 G1으로 출고해",
    ),
    (
        "시나리오 3 — 긴급 입고",
        "긴급 입고: 로봇 3대 모두 G2 입고 게이트로 즉시 집결",
    ),
]


def main():
    parser = argparse.ArgumentParser(description="MARS Phase 4 데모")
    parser.add_argument("--task",        type=str, default=None)
    parser.add_argument("--demo",        action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MARS Phase 4 — Claude API 오케스트레이터 데모")
    print("  Propose-then-Commit 경로 검증 파이프라인")
    print("=" * 60)

    orch  = WarehouseOrchestrator()
    fleet = RobotFleet()

    if args.interactive:
        print("\n임무를 입력하세요 (종료: q)\n")
        while True:
            try:
                task = input("임무> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task.lower() in ("q", "quit", "종료"):
                break
            if task:
                run_mission(orch, fleet, task)

    elif args.demo:
        for title, task in DEMO_SCENARIOS:
            print(f"\n\n{'#'*60}")
            print(f"  {title}")
            print(f"{'#'*60}")
            ok = run_mission(orch, fleet, task)
            status = "✅ 성공" if ok else "❌ 재계획 필요"
            print(f"\n  결과: {status}")
            if title != DEMO_SCENARIOS[-1][0]:
                time.sleep(2)

        print(f"\n\n{'='*60}")
        print(f"  데모 완료 | API 사용량: {orch.governor.usage_summary}")
        print(f"{'='*60}")

    elif args.task:
        run_mission(orch, fleet, args.task)

    else:
        run_mission(orch, fleet, DEMO_SCENARIOS[0][1])


if __name__ == "__main__":
    main()
