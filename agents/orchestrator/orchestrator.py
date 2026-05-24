"""Phase 4 — Claude API 오케스트레이터.

자연어 임무를 받아 창고 상태를 파악하고 로봇 3대에게 목표를 배정한다.

실행:
  python agents/orchestrator/orchestrator.py
  python agents/orchestrator/orchestrator.py --task "A구역 박스 3개를 출고 게이트로 이동"
  python agents/orchestrator/orchestrator.py --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import anthropic

# ── 창고 상태 (데모용 하드코딩 — 실배포 시 pgvector로 대체) ──────────────

WAREHOUSE_MAP = {
    "shelves": [
        {"id": "A1", "zone": "A", "x": -3.5, "y":  2.0, "product": "전자부품",    "qty": 45},
        {"id": "A2", "zone": "A", "x": -3.5, "y":  0.0, "product": "케이블",      "qty": 12},
        {"id": "A3", "zone": "A", "x": -3.5, "y": -2.0, "product": "배터리팩",    "qty": 30},
        {"id": "B1", "zone": "B", "x":  0.0, "y":  2.0, "product": "포장재",      "qty": 80},
        {"id": "B2", "zone": "B", "x":  0.0, "y":  0.0, "product": "완제품 A",    "qty": 20},
        {"id": "B3", "zone": "B", "x":  0.0, "y": -2.0, "product": "완제품 B",    "qty": 15},
        {"id": "C1", "zone": "C", "x":  3.5, "y":  2.0, "product": "반도체칩",    "qty":  8},
        {"id": "C2", "zone": "C", "x":  3.5, "y":  0.0, "product": "센서모듈",    "qty": 25},
        {"id": "C3", "zone": "C", "x":  3.5, "y": -2.0, "product": "모터부품",    "qty": 18},
    ],
    "gates": [
        {"id": "G1", "x":  4.5, "y":  0.0, "type": "출고"},
        {"id": "G2", "x": -4.5, "y":  0.0, "type": "입고"},
        {"id": "G3", "x":  0.0, "y":  4.5, "type": "충전"},
    ],
    "robots": [
        {"id": "R1", "x": -3.0, "y": -3.5, "battery": 87, "status": "idle"},
        {"id": "R2", "x":  0.0, "y": -3.5, "battery": 64, "status": "idle"},
        {"id": "R3", "x":  3.0, "y": -3.5, "battery": 92, "status": "idle"},
    ],
}

SYSTEM_PROMPT = """당신은 물류 창고 자율 로봇 시스템(MARS)의 AI 오케스트레이터입니다.

창고에는 로봇 3대(R1, R2, R3)가 있으며, 각 로봇은 RL 정책으로 자율 이동합니다.

임무가 주어지면:
1. 창고 상태를 분석하여 어떤 로봇이 어떤 목표 지점으로 이동해야 하는지 결정합니다
2. 경로 충돌을 최소화하도록 목표를 배정합니다
3. 배정 근거와 함께 JSON으로 응답합니다

응답 형식 (반드시 이 JSON만 출력):
{
  "plan_summary": "임무 요약 (1-2문장)",
  "assignments": [
    {
      "robot_id": "R1",
      "goal_x": 3.5,
      "goal_y": 2.0,
      "task": "C1 선반 → G1 출고 게이트",
      "priority": 1
    }
  ],
  "safety_notes": "충돌 회피 / 우선순위 근거",
  "estimated_completion_sec": 45
}

중요: 로봇 간 목표가 겹치지 않도록 배정하세요. 배터리가 낮은 로봇(< 30%)은 충전 게이트(G3)로 먼저 배정합니다."""


@dataclass
class RobotAssignment:
    robot_id: str
    goal_x: float
    goal_y: float
    task: str
    priority: int


@dataclass
class MissionPlan:
    plan_summary: str
    assignments: list[RobotAssignment]
    safety_notes: str
    estimated_completion_sec: int
    raw_response: str = ""


class CostGovernor:
    """API 호출 비용 제한."""

    def __init__(self, max_cost_usd: float = 2.0, max_calls: int = 50):
        self.max_cost_usd = max_cost_usd
        self.max_calls = max_calls
        self._total_tokens = 0
        self._calls = 0

    def check(self) -> None:
        if self._calls >= self.max_calls:
            raise RuntimeError(f"[CostGovernor] 세션 호출 한도 {self.max_calls}회 초과")
        estimated_cost = self._total_tokens / 1_000_000 * 3.0  # ~$3/M tokens
        if estimated_cost >= self.max_cost_usd:
            raise RuntimeError(f"[CostGovernor] 예산 한도 ${self.max_cost_usd:.2f} 초과")

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self._total_tokens += input_tokens + output_tokens
        self._calls += 1

    @property
    def usage_summary(self) -> str:
        cost = self._total_tokens / 1_000_000 * 3.0
        return f"{self._calls}회 호출, ~{self._total_tokens:,} 토큰, ~${cost:.4f}"


class WarehouseOrchestrator:
    """Claude API 기반 창고 임무 오케스트레이터."""

    MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY 환경변수를 설정하세요.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-..."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.governor = CostGovernor()
        self._mission_log: list[dict] = []

    # ── 창고 상태 직렬화 ──────────────────────────────────────────────

    def _warehouse_context(self) -> str:
        state = WAREHOUSE_MAP
        lines = ["=== 창고 현재 상태 ===\n"]

        lines.append("[ 선반 현황 ]")
        for s in state["shelves"]:
            lines.append(
                f"  {s['id']} (Zone {s['zone']}): {s['product']} {s['qty']}개 "
                f"@ ({s['x']:.1f}, {s['y']:.1f})"
            )

        lines.append("\n[ 출입구 ]")
        for g in state["gates"]:
            lines.append(f"  {g['id']}: {g['type']} @ ({g['x']:.1f}, {g['y']:.1f})")

        lines.append("\n[ 로봇 상태 ]")
        for r in state["robots"]:
            bat_warn = " ⚠️ 배터리 부족" if r["battery"] < 30 else ""
            lines.append(
                f"  {r['id']}: 배터리 {r['battery']}%, {r['status']} "
                f"@ ({r['x']:.1f}, {r['y']:.1f}){bat_warn}"
            )

        return "\n".join(lines)

    # ── Claude API 호출 ───────────────────────────────────────────────

    def plan_mission(self, task: str) -> MissionPlan:
        """자연어 임무를 받아 로봇 목표 배정 계획을 반환."""
        self.governor.check()

        user_content = f"{self._warehouse_context()}\n\n=== 임무 요청 ===\n{task}"

        print(f"\n[Orchestrator] Claude API 호출 중...")
        t0 = time.time()

        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        elapsed = time.time() - t0
        self.governor.record(response.usage.input_tokens, response.usage.output_tokens)

        raw = response.content[0].text.strip()
        print(f"[Orchestrator] 응답 수신 ({elapsed:.2f}s, "
              f"in={response.usage.input_tokens} out={response.usage.output_tokens} tok)")

        plan = self._parse_response(raw, task)
        self._mission_log.append({
            "task": task,
            "plan": asdict(plan) if not plan.raw_response else {**asdict(plan), "raw_response": "..."},
            "latency_s": round(elapsed, 2),
        })
        return plan

    def _parse_response(self, raw: str, task: str) -> MissionPlan:
        # JSON 블록 추출
        text = raw
        if "```" in text:
            start = text.find("```")
            end   = text.rfind("```")
            block = text[start:end].strip()
            block = block.lstrip("`").lstrip("json").strip()
            text  = block

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"[경고] JSON 파싱 실패: {e}\n원본:\n{raw[:400]}")
            return self._fallback_plan(task, raw)

        assignments = [
            RobotAssignment(
                robot_id = a["robot_id"],
                goal_x   = float(a["goal_x"]),
                goal_y   = float(a["goal_y"]),
                task     = a.get("task", ""),
                priority = int(a.get("priority", 1)),
            )
            for a in data.get("assignments", [])
        ]

        return MissionPlan(
            plan_summary            = data.get("plan_summary", ""),
            assignments             = assignments,
            safety_notes            = data.get("safety_notes", ""),
            estimated_completion_sec= int(data.get("estimated_completion_sec", 60)),
            raw_response            = raw,
        )

    def _fallback_plan(self, task: str, raw: str) -> MissionPlan:
        """JSON 파싱 실패 시 기본 계획."""
        robots = WAREHOUSE_MAP["robots"]
        gates  = WAREHOUSE_MAP["gates"]
        return MissionPlan(
            plan_summary=f"기본 배정 (파싱 실패): {task}",
            assignments=[
                RobotAssignment(
                    robot_id=r["id"],
                    goal_x=gates[0]["x"],
                    goal_y=gates[0]["y"] + i * 0.5,
                    task=f"{r['id']} → G1",
                    priority=i + 1,
                )
                for i, r in enumerate(robots)
            ],
            safety_notes="파싱 실패로 기본 배정 적용",
            estimated_completion_sec=60,
            raw_response=raw,
        )

    # ── 출력 ─────────────────────────────────────────────────────────

    def print_plan(self, plan: MissionPlan) -> None:
        print("\n" + "=" * 60)
        print(f"  임무 계획")
        print("=" * 60)
        print(f"  {plan.plan_summary}")
        print()
        print(f"  {'로봇':<5} {'목표 (x, y)':<20} {'임무'}")
        print(f"  {'-' * 55}")
        for a in sorted(plan.assignments, key=lambda x: x.priority):
            print(f"  {a.robot_id:<5} ({a.goal_x:+.1f}, {a.goal_y:+.1f}){'':<10} {a.task}")
        print()
        print(f"  안전: {plan.safety_notes}")
        print(f"  예상 완료: {plan.estimated_completion_sec}초")
        print(f"  API 사용량: {self.governor.usage_summary}")
        print("=" * 60)

    def get_goal_list(self, plan: MissionPlan) -> list[tuple[float, float]]:
        """[R1 goal, R2 goal, R3 goal] 순서로 (x, y) 반환 — 시뮬 연동용."""
        goal_map = {a.robot_id: (a.goal_x, a.goal_y) for a in plan.assignments}
        return [
            goal_map.get("R1", (0.0, 0.0)),
            goal_map.get("R2", (0.0, 0.0)),
            goal_map.get("R3", (0.0, 0.0)),
        ]


# ── CLI ──────────────────────────────────────────────────────────────────

DEFAULT_TASKS = [
    "C구역 전체 재고를 G1 출고 게이트로 이동해줘",
    "배터리가 부족한 로봇을 충전소로 보내고, 나머지는 A구역 재고를 B구역으로 이동시켜",
    "긴급 주문: 반도체칩과 센서모듈을 최대한 빨리 G1 게이트로 출고해",
]


def main():
    parser = argparse.ArgumentParser(description="MARS Phase 4 오케스트레이터 데모")
    parser.add_argument("--task",        type=str, default=None, help="임무 입력")
    parser.add_argument("--interactive", action="store_true",    help="대화형 모드")
    parser.add_argument("--demo",        action="store_true",    help="데모 임무 3개 순차 실행")
    args = parser.parse_args()

    orch = WarehouseOrchestrator()

    if args.interactive:
        print("\n[MARS 오케스트레이터] 임무를 입력하세요 (종료: q)")
        while True:
            try:
                task = input("\n임무> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task.lower() in ("q", "quit", "exit", "종료"):
                break
            if not task:
                continue
            plan = orch.plan_mission(task)
            orch.print_plan(plan)

    elif args.demo:
        print("\n[MARS 오케스트레이터] 데모 임무 3종 실행\n")
        for i, task in enumerate(DEFAULT_TASKS, 1):
            print(f"\n{'─'*60}")
            print(f"  데모 임무 {i}/{len(DEFAULT_TASKS)}: {task}")
            plan = orch.plan_mission(task)
            orch.print_plan(plan)
            if i < len(DEFAULT_TASKS):
                time.sleep(1)

    elif args.task:
        plan = orch.plan_mission(args.task)
        orch.print_plan(plan)

    else:
        # 기본: 첫 번째 데모 임무
        task = DEFAULT_TASKS[0]
        print(f"\n임무: {task}")
        plan = orch.plan_mission(task)
        orch.print_plan(plan)


if __name__ == "__main__":
    main()
