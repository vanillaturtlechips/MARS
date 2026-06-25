# MARS — Custom ROS 2 Interfaces (`mars_msgs`) & Sensor Sources

Implementation, build, and verification guide for the **Phase 4 / M4** ROS 2 layer.
Covers architecture doc **§1a** (topic → enriched-event mapping) and **§1b**
(custom ROS 2 interfaces).

> Scope: this document is about the *ROS 2 boundary* only — the custom message,
> the topics the Aggregator consumes, and the self-published sim sources. The
> supervisory reasoning above the Aggregator is unchanged.

---

## 1. Status (what is actually built & verified)

| Item | Status | Evidence |
|------|--------|----------|
| `mars_msgs` colcon package (`RobotHealth.msg`) | ✅ builds | `colcon build` finishes; `ros2 interface show mars_msgs/msg/RobotHealth` prints the definition |
| Self-publish node (`/<robot>/robot_health`, `/<robot>/battery_state`) | ✅ runs | topics appear; `ros2 topic echo` returns messages |
| Fault injection (`/mars/fault_cmd`) | ✅ works | injecting `R1 ERROR MOTOR_FAULT` flips `level: 0 → 2`, `fault_codes: [] → [MOTOR_FAULT]` |
| Adapter nav-result path (`_on_result`) | ⚠️ fixed, not yet verified | needs a real (or mock) `NavigateToPose` abort to exercise |
| Isaac Sim + Nav2 bringup | ⬜ not started | Step 4 |

Verified environment: **Ubuntu 22.04 (Jammy), ROS 2 Jazzy, Python 3.10**.

---

## 2. Why a custom message at all

Almost everything uses standard ROS 2 messages. We build exactly **one** custom
message; the rest are optional (§1b).

`RobotHealth` exists because **nothing in an Isaac Sim + Nav2 stack publishes robot
health or e-stop/fault state** — those are supervisor concerns, not simulator
physics. The Aggregator combines `RobotHealth` with the standard
`sensor_msgs/BatteryState` to compute the deterministic `fault_flag` on a failure
event. Publishing a fault here is also the **primary trigger for the
Failure-Analysis demo** (the robot-internal scenario).

Battery deliberately stays in standard `sensor_msgs/BatteryState`
(`percentage` ∈ [0,1], `power_supply_status`: 1=CHARGING, 2=DISCHARGING, 4=FULL).
It is **not** duplicated into `RobotHealth`.

### `mars_msgs/msg/RobotHealth`

```text
std_msgs/Header header

uint8 LEVEL_OK=0
uint8 LEVEL_WARN=1
uint8 LEVEL_ERROR=2

string    robot_id
uint8     level          # overall health level (constants above)
bool      estop_active
string[]  fault_codes    # e.g. ["MOTOR_FAULT","LIDAR_TIMEOUT"]
```

`fault_flag` is derived by the Aggregator (deterministic, no inference):
`robot_health.level == LEVEL_ERROR` **OR** `estop_active` **OR**
`battery_pct < BATTERY_CRITICAL_PCT`.

---

## 3. Topic → enriched-event field mapping (Isaac Sim + Nav2 Jazzy)

All subscriptions are per-robot namespaced (`/<robot_id>/...`). Three sources:
**Nav2** (navigation), **Isaac Sim** (sim state/sensors), and **self-published**
sim nodes (battery, health — nothing else produces these).

| Subscribed topic | Type | Source | → Field |
|------------------|------|--------|---------|
| `…/navigate_to_pose/_action/status` | `action_msgs/GoalStatusArray` | Nav2 | `event_type`, `nav_outcome`, `goal_status` (**trigger**) |
| `…/navigate_to_pose/_action/feedback` | `nav2_msgs/.../NavigateToPose_FeedbackMessage` | Nav2 | struggle signal: `number_of_recoveries`, `distance_remaining` |
| `/tf` (map → base_link) | `tf2_msgs/TFMessage` | Isaac Sim | robot pose → zone, distribution |
| `…/odom` | `nav_msgs/Odometry` | Isaac Sim | velocity, backup pose |
| `…/battery_state` | `sensor_msgs/BatteryState` | **self** | `health_at_failure.battery_pct` |
| `…/robot_health` | `mars_msgs/RobotHealth` (**custom**) | **self** | `estop` / `fault_codes` → `fault_flag` |
| `/clock` | `rosgraph_msgs/Clock` | Isaac Sim | `use_sim_time = true` |

**Outcome source (Jazzy-specific):** `nav_outcome` is read from the action
**GoalStatus** (`4=SUCCEEDED, 5=CANCELED, 6=ABORTED`), **not** a result error
code — on Jazzy the `NavigateToPose` *result* payload is empty. The adapter
drives outcomes from the result-wrapper's `status` field (see §6).

---

## 4. Package layout

```text
agents/mars/
├── ros2_ws/                         # colcon workspace (build here)
│   └── src/mars_msgs/
│       ├── package.xml              # ament_cmake, depends std_msgs + geometry_msgs
│       ├── CMakeLists.txt           # rosidl_generate_interfaces(RobotHealth.msg)
│       └── msg/RobotHealth.msg
├── mars/
│   ├── mars_msgs/__init__.py        # PURE-PYTHON SHIM (fallback only, see note)
│   ├── ros/
│   │   ├── interfaces.py            # ABCs: Navigation/Sensor (no rclpy import)
│   │   ├── isaac_sim_adapter.py     # rclpy adapter (real Isaac Sim + Nav2)
│   │   ├── ros2_node.py             # entry point: adapter → Aggregator
│   │   └── zone_resolver.py
│   └── sim/
│       └── ros_health_publisher.py  # self-publishes robot_health + battery_state
```

> **Shim vs package.** `mars/mars_msgs/__init__.py` is a plain-`dataclass` shim so
> the supervisory code can `import` `RobotHealth` *without* a compiled ROS 2
> workspace (e.g. for the MockSim demo / unit tests). The **real, on-the-wire**
> message comes from the colcon package in `ros2_ws/`. The adapter prefers the
> compiled `mars_msgs` and falls back to the shim only when rclpy/`mars_msgs`
> isn't available.

---

## 5. Build & verify

ROS 2 work runs under **system Python 3.10** with ROS sourced — **not** inside the
Isaac Lab venv (which is Python 3.12). Deactivate any venv first.

### 5.1 Environment (one-time, RunPod)

```bash
# installs ROS 2 Jazzy + Nav2 + colcon and aligns python3 → 3.10
bash deploy/runpod/setup_ros2.sh
```

> Why python3.12: ROS 2 Jazzy is built for 3.10; if `/usr/bin/python3` points to
> 3.11, `import rclpy` and `apt_pkg` both fail. The script sets
> `update-alternatives --set python3 /usr/bin/python3.12`. The RL venv is
> unaffected (it has its own 3.11). Revert with `--set python3 .../python3.12`.

### 5.2 Build `mars_msgs`

```bash
cd agents/mars/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select mars_msgs
source install/setup.bash
ros2 interface show mars_msgs/msg/RobotHealth   # should print the definition
```

### 5.3 Run the self-publish node + inject a fault

```bash
cd agents/mars
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

python3 -m mars.sim.ros_health_publisher > /tmp/health.log 2>&1 &
sleep 3

ros2 topic echo /R1/robot_health --once          # expect level: 0

ros2 topic pub --once /mars/fault_cmd std_msgs/msg/String '{data: "R1 ERROR MOTOR_FAULT"}'
sleep 2
ros2 topic echo /R1/robot_health --once          # expect level: 2, fault_codes: [MOTOR_FAULT]
```

Stop the node with `pkill -f ros_health_publisher`.

> **Gotcha:** the node runs with `use_sim_time=False` by default. With
> `use_sim_time=True` and no `/clock` (i.e. no Isaac Sim running) the 1 Hz timer
> never fires and nothing is published. When running *with* Isaac Sim, override:
> `python3 -m mars.sim.ros_health_publisher --ros-args -p use_sim_time:=true`.

### Fault-injection commands (`/mars/fault_cmd`, `std_msgs/String`)

| Command | Effect |
|---------|--------|
| `R3 ERROR MOTOR_FAULT` | `level=ERROR`, `fault_codes=[MOTOR_FAULT]` → `fault_flag` |
| `R3 ESTOP` | `estop_active=True`, `level=ERROR` → `fault_flag` |
| `R3 BATTERY 12` | battery 12 % (→ `fault_flag` if below CRITICAL) |
| `R3 WARN LIDAR_TIMEOUT` | `level=WARN` (no `fault_flag`) |
| `R3 CLEAR` | back to healthy |

---

## 6. Adapter nav-result fix (important)

`isaac_sim_adapter.py` previously fired nav-status callbacks only from
`/_action/status` (`GoalStatusArray`), matching our dispatch `goal_id` against
Nav2's UUID — which **never matches**, so aborts never reached the Aggregator.

The fix drives outcomes from **`get_result_async()`'s result wrapper**: on Jazzy
the result payload is empty but `result.status` still carries the terminal
`GoalStatus` (4/5/6), and the callback already has the correct registered
`goal_id`. This path is now the source of truth; the status-array subscription is
redundant. **Still needs a real/mock `NavigateToPose` abort to verify end-to-end.**

---

## 7. Optional interfaces (not built yet — §1b)

Build only if a clean supervisory mission abstraction over Nav2 is wanted;
otherwise the ROS Executor dispatches a plain `NavigateToPose` goal and mission
state lives in the blackboard.

- `mars_msgs/msg/MissionCommand` — ROS Executor → robot (assign/cancel).
- `mars_msgs/msg/MissionStatus` — supervisor-published mission lifecycle mirror.
- `mars_msgs/srv/MissionFeasible` — pre-dispatch energy/reachability query
  (honors Principle 4: the robot owns routing/energy). Fallback is a flat
  battery-% threshold.

---

## 8. Remaining work (Steps 4–6)

1. **Step 4** — Isaac Sim + Nav2 bringup: a navigable robot publishing `/tf`,
   `/odom`, `/scan`, `/clock`, accepting `NavigateToPose`. (Heaviest step — the
   Isaac Sim ROS 2 bridge is the main unknown.)
2. **Step 5** — trigger a real Nav2 abort (block the path) → confirm the adapter
   emits an enriched event to the Aggregator (verifies §6).
3. **Step 6** — full pipeline: real abort → diagnosis → `avoid_zone` policy.
