"""
DEMO 3 — the multi-agent CONVERSATION (agent-to-agent), no Isaac/ROS.

Runs the REAL supervisory pipeline on a zone-wide failure and surfaces the
message flow between the specialist agents so an audience can SEE the agents
talk and reach a fleet decision — the part demo1 actuates on the robots but
hides in the logs:

  fleet incident (3 robots abort in one zone)
    -> FailureAnalysisAgent (Haiku, ReAct): queries the blackboard with its
       read-only tools, reasons over multiple real failures -> diagnosis
    -> DecisionValidator: grounding / confidence gate
    -> OperationsStrategyAgent (Haiku): reads the diagnosis -> recommends a
       fleet policy from the whitelist (avoid_zone)
    -> Guardrail -> PolicyManager activates it (-> KeepoutService -> Nav2 in demo1)

Everything below the banners is the agents' own INFO logging — the genuine
A2A trace, not a script. The blackboard (shared DB) is how one agent's output
reaches the next.

Run (system py3.10, env_ros2 sourced, Postgres up, agents/mars/.env with
ANTHROPIC key), from agents/mars/:
    python3 -m tools.demo_agent_conversation
"""
from __future__ import annotations

import logging

# Clean, readable formatting so the agent messages read like a transcript.
logging.basicConfig(level=logging.INFO, format="    %(name)-22s %(message)s")
log = logging.getLogger("demo3")

import mars.blackboard.queries as Q                       # noqa: E402
from mars.blackboard.db import connect, apply_migrations  # noqa: E402
from mars.orchestrator import demo as demopkg             # noqa: E402
from mars.llm.client import get_llm_client                # noqa: E402

ZONE = "receiving_dock"
ROBOTS = ["R1", "R2", "R3"]


def _failure_event(robot: str, spread: int) -> dict:
    """A robot whose nav aborted because the zone is physically blocked (same
    shape demo1's bridge records). spread = how many distinct robots have now
    failed in this zone -> the investigator sees a zone obstruction, not a glitch."""
    return {
        "event_type": "navigation.aborted", "robot_id": robot,
        "mission_id": f"{robot}-dock", "goal_id": f"{robot}-goal", "zone": ZONE,
        "nav_outcome": "aborted", "goal_status": 6,
        "failure_reason": "path blocked by obstacle in zone; controller could not make progress",
        "health_at_failure": {"battery_pct": 80, "estop_active": False, "fault_codes": ["NAV_PATH_BLOCKED"]},
        "fault_flag": "path_blocked",
        "distribution": {"per_robot_zone_spread": 1, "per_zone_robot_spread": spread},
        "failures_for_this_mission": 1,
    }


def banner(text: str) -> None:
    print("\n" + "=" * 74 + f"\n  {text}\n" + "=" * 74, flush=True)


def main() -> None:
    conn = connect()
    apply_migrations(conn)
    demopkg.seed_world(conn)   # zones/robots only — NO seeded failures/diagnoses

    hot_state = demopkg._make_hot_state()
    embedder = demopkg._make_embedder()
    llm = get_llm_client("anthropic")
    components = demopkg.build_components(connect, hot_state, llm, embedder)
    orchestrator = components["orchestrator"]
    policy_manager = components["policy_manager"]

    banner(f"FLEET INCIDENT — robots aborting in '{ZONE}'")
    # First two robots fail and are recorded on the blackboard (the shared memory
    # the agents read). The third triggers the supervisory brain.
    for r in ROBOTS[:2]:
        c = connect()
        Q.write_failure(c, _failure_event(r, 1)); c.commit(); c.close()
        print(f"    {r}: navigation.aborted in {ZONE}  (NAV_PATH_BLOCKED)", flush=True)
    print(f"    {ROBOTS[2]}: navigation.aborted in {ZONE}  (NAV_PATH_BLOCKED)", flush=True)

    banner("SUPERVISORY BRAIN — multi-agent A2A (live Haiku, blackboard-mediated)")
    print("  (everything below is the agents' own log — a real conversation)\n", flush=True)
    result = orchestrator.handle_failure(_failure_event(ROBOTS[2], spread=3), conn)
    conn.commit()

    banner("OUTCOME — fleet decision the agents reached")
    print(f"    orchestrator path : {result.get('path')}   "
          f"(slow = full agent pipeline ran)", flush=True)
    print(f"    decision validator: {result.get('dv_result')}   "
          f"(grounding/confidence gate on the diagnosis)", flush=True)
    print(f"    disposition       : {result.get('disposition')}", flush=True)
    print(f"    diagnosis_id      : {result.get('diagnosis_id')}   "
          f"(written to the blackboard for the next agent + future RAG)", flush=True)
    active = policy_manager.is_policy_active_for_zone(ZONE, "avoid_zone")
    print(f"    avoid_zone active : {active}   (-> KeepoutService -> Nav2 keepout in demo1)", flush=True)
    print(f"\n  Single-robot RL can react to its own obstacle; only this supervisory\n"
          f"  layer turns 3 separate failures into one fleet-wide 'avoid {ZONE}'.\n", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
