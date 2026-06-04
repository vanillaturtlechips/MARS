# MARS — Multi-Agent Robot Supervisory System

Supervisory AI layer for a warehouse robot fleet.  Agents reason; services
execute; nothing an agent produces reaches a robot without a deterministic
validator.

---

## Quick start

### Prerequisites

- Docker + Docker Compose
- Python 3.10+ (with `.venv` already set up)
- ROS2 Humble (only needed for M4 Isaac Sim adapter; the mock sim runs without it)

### 1. Start Postgres + Redis

```bash
docker-compose up -d
# wait for healthy status
docker-compose ps
```

The Postgres container automatically applies `mars/blackboard/migrations/0001_initial.sql`
(mounted as `docker-entrypoint-initdb.d`).  On an existing volume this is a no-op.

To apply the migration manually against an external Postgres:

```bash
psql $DB_DSN -f mars/blackboard/migrations/0001_initial.sql
```

### 2. Install Python dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and OPENAI_API_KEY
```

---

## M1 demo — end-to-end vertical slice

The M1 demo runs the full pipeline without ROS2 or Isaac Sim:

```
fault injected via FaultInjector
  → MockSim generates NavGoalStatus(ABORTED)
  → Aggregator enriches event (health snapshot + distribution)
  → Router selects SLOW path (4 robots, same zone)
  → Failure Analysis Agent diagnoses zone_wide congestion
  → Decision Validator: PASS
  → Strategy Trigger Rules: TRIGGER
  → Operations Strategy Agent recommends avoid_zone
  → Policy Guardrail: ACCEPT
  → Policy Manager activates policy
  → Scheduler skips missions routing through that zone
  → Outcome Evaluator watches for 15 min, labels result
```

Run the demo script (to be added in M1):

```bash
python -m mars.orchestrator.demo
```

---

## Running tests

```bash
pytest tests/ -v
```

Unit tests have no external dependencies (no Postgres, no Redis, no LLM calls).
The LLM is mocked with canned outputs from `tests/conftest.py`.

---

## Repository layout

```
mars/
  mars_msgs/      ROS2 interface package — RobotHealth.msg (+ CMakeLists.txt)
  blackboard/     Postgres + pgvector schema, migrations, query layer, Redis hot state
  ros/            Abstract ROS boundary interfaces (no rclpy import here)
  sim/            Mock sim node + fault injector (no GPU, no Isaac Sim needed)
  aggregator/     Health snapshot + distribution enrichment (§1a)
  orchestrator/   Workflow coordination + strategy trigger rules
  router/         Deterministic fast/slow path selection (§2)
  agents/         failure_analysis, operations_strategy, fleet_state
  validators/     retrieval_validator (§4), decision_validator (§5)
  guardrail/      Policy guardrail — schema, refs, feasibility, conflict, bounds (§6)
  policy/         Policy manager (lifecycle, expiry, consumer notifications)
  services/       scheduling, charging, ros_executor (stubs per §6a/§6b)
  outcome/        Outcome evaluator (closes RAG loop — §7)
  llm/            Provider-agnostic LLM client + OpenAI embedder
  config.py       All env-var-driven settings
tests/
DECISIONS.md      All open-parameter defaults with rationale
docker-compose.yml
```

## Building mars_msgs (ROS2 Humble)

```bash
mkdir -p ~/ros2_ws/src
ln -s $(pwd)/mars/mars_msgs ~/ros2_ws/src/mars_msgs
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mars_msgs
source install/setup.bash
```

Once built, `from mars_msgs.msg import RobotHealth` works in ROS2 code.
Without a ROS2 build, use the Python shim at `mars/mars_msgs/__init__.py`.

---

## Milestones

| Milestone | Status |
|-----------|--------|
| **M0** — Scaffold (schema, infra, LLM client, mock sim) | ✅ Done — confirm schema before M1 |
| **M1** — Vertical slice (full pipeline, observable) | 🔲 Next |
| **M2** — Breadth (charging, scheduling, fleet agent) | 🔲 Pending |
| **M3** — Depth (validators, guardrail, outcome RAG) | 🔲 Pending |
| **M4** — Isaac Sim adapter | 🔲 Pending |
