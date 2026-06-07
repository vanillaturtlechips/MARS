# Phase 4-B — Supervisory agents → Nav2 keepout in Isaac Sim

LLM supervisory agents that detect robot failures, decide an `avoid_zone`
policy, and reroute real robots in Isaac Sim via a Nav2 KeepoutFilter.

Status (2026-06-06): **core loop verified end-to-end on RunPod.** Remaining items
are visual/recording and a multi-robot reroute capture — see [Remaining](#remaining-work).

---

## 1. Two layers (don't conflate)

- **RL layer (S1–S6):** per-robot navigation / obstacle avoidance / deadlock —
  trained policies (model_10998 etc.), Isaac Lab Gym. **Not this work.**
- **Supervisory layer (this work):** global decisions on top — "this zone is
  bad, route the fleet around it." LLM agents + Nav2 keepout.

The differentiator: pure RL can't make a fleet-wide "avoid this dock" call; the
agents can.

## 2. The loop

```
robot Nav2 abort  ->  Aggregator/bridge  ->  FailureAnalysisAgent (Haiku, ReAct)
  -> diagnosis (zone_wide)  -> DecisionValidator (grounding/confidence)
  -> OperationsStrategyAgent (Haiku) -> avoid_zone policy
  -> DecisionValidator + Guardrail -> PolicyManager.activate
       -> KeepoutService: zone polygon -> OccupancyGrid mask
       -> /keepout_filter_mask  (nav_msgs/OccupancyGrid, latched)
  -> Nav2 KeepoutFilter (global costmap) -> masked cells lethal
  -> planner routes the robot around the zone
```

### Agents (LLM, Haiku) — only 3
- **FailureAnalysisAgent** (`mars/agents/failure_analysis.py`) — ReAct investigator,
  read-only tools (`InvestigatorTools`): query_failures, get_zone_state,
  get_robot_history, search_incidents, get_active_policies. Output: diagnosis
  {cause, scope, persistence, affected_zone, confidence, evidence, relied_on_precedents}.
- **OperationsStrategyAgent** (`operations_strategy.py`) — recommends policies from
  `POLICY_WHITELIST` (avoid_zone, delay_low_priority_missions,
  reserve_chargers_for_critical, lower_target_charge_level, pre_charge_for_demand_spike).
- **FleetStateAgent** (`fleet_state.py`) — periodic fleet health / charging pressure.

Everything else (Aggregator, Scheduler, Navigation/Nav2, validators, guardrail,
PolicyManager) is deterministic — **not agents.**

## 3. Components built

**Supervisor (policy → keepout mask), pure Python, unit-tested:**
- `mars/ros/keepout.py` — rasterize avoid-zone polygons → `nav_msgs/OccupancyGrid`
  dict (100 = keepout). `MapMeta.covering()` auto-sizing. (`tests/test_keepout.py`)
- `mars/services/keepout_service.py` — PolicyManager consumer; on avoid_zone
  activate/deactivate, fetch zone polygons → union mask → `nav.publish_keepout_mask`.
  (`tests/test_keepout_service.py`)
- `mars/ros/interfaces.py` — `publish_keepout_mask` no-op default; mock_sim records it.
- `mars/blackboard/queries.py` — `get_zone_polygons`, upsert_zone now upserts polygon.

**Real ROS2 publish:**
- `mars/ros/isaac_sim_adapter.py` — `ROS2SimAdapter.publish_keepout_mask` →
  latched OccupancyGrid on `/keepout_filter_mask`.
- `mars/ros/ros2_keepout_publisher.py` — standalone `Ros2KeepoutPublisher`
  (used by demo.py `--ros2-keepout` and the bridges).

**Nav2 (RunPod):** `deploy/nav2/`
- `make_empty_map.py` — 40×40 m free occupancy map (numpy-free).
- `nav2_keepout_demo.params.yaml` — planner(NavFn)/controller(RPP)/costmaps
  (static + inflation + **keepout_filter**, no obstacle layer / no lidar),
  global_frame=map, no AMCL. `static_layer map_topic:/map`.
- `keepout_costmap_filter_info.yaml` + `keepout_filter.launch.py` — costmap_filter_info_server.
- `bringup_keepout.launch.py` — single-robot all-in-one.
- `bringup_global.launch.py` — shared map_server + costmap_filter_info_server.
- `bringup_robot_ns.launch.py namespace:=R*` — per-robot namespaced Nav2
  (RewrittenYaml root_key=ns, robot_base_frame=<ns>/base_link).
- `global_costmap_keepout_snippet.yaml`, `README_keepout.md`.

**Isaac Sim scenes (RunPod, `deploy/isaac/`):**
- `env_isaac.sh` / `env_ros2.sh` — sourceable env (internal ROS2 + UDP FastDDS).
- `fastdds_udp_only.xml` — UDPv4-only DDS profile (Isaac↔system ROS2 interop).
- `isaac_ros2_smoke.py` — 1a: /clock smoke test.
- `isaac_warehouse_ros2.py` — single iw_hub: /odom, /tf, /cmd_vel→diff drive
  (`--obstacle`, `--warehouse`).
- `isaac_multi_robot_ros2.py` — R1/R2/R3 namespaced, dock blocking box.

**Brain↔ROS2 bridges (RunPod, `agents/mars/tools/`):**
- `keepout_publish_test.py` — standalone mask publisher (Nav2 plumbing test).
- `isaac_failure_bridge.py` — single robot: real abort → brain → mask.
- `isaac_multi_failure_bridge.py` — multi-robot: ≥2 distinct real aborts →
  zone_wide → avoid_zone → mask (no seeding).

**Setup scripts:** `deploy/runpod/setup_phase_b.sh`, `setup_postgres.sh`,
`install_isaac_pip.sh`.

## 4. Verified

| Step | What | Status |
|------|------|--------|
| Phase A | brain → avoid_zone + fleet policies (real Haiku), 139 tests | ✅ local + RunPod |
| keepout core | polygon → OccupancyGrid, KeepoutService | ✅ unit tests |
| 1a | Isaac↔ROS2 bridge `/clock` | ✅ RunPod |
| 1b | iw_hub `/odom` + `/tf` | ✅ |
| 1c | `/cmd_vel` → diff drive (drove 0→19 m) | ✅ |
| 2a | Nav2 drives iw_hub to a goal | ✅ |
| 2b | keepout mask → planner reroute → robot detours (y≈2.5 around x=4) | ✅ |
| full loop (single) | `demo.py --ros2-keepout`: live Haiku avoid_zone → mask → reroute | ✅ |
| gap#1 (single, real failure) | iw_hub real dock abort → brain → avoid_zone → reroute | ✅ |
| **multi-robot real zone_wide** | R2+R3 real dock aborts (no seeding) → Haiku zone_wide 0.72 PASS → avoid_zone → mask | ✅ |

## 5. Run procedure (multi-robot, RunPod)

Each long process in its own terminal; **never Ctrl+Z** (suspends). Kill stale
first: `pkill -9 -f deploy/isaac; pkill -9 -f /opt/ros/humble/lib/nav2; pkill -9 -f bridge`.

1. **Isaac** (`source deploy/isaac/env_isaac.sh`):
   `python deploy/isaac/isaac_multi_robot_ros2.py`
2. **Global Nav2** (`source deploy/isaac/env_ros2.sh`):
   `ros2 launch deploy/nav2/bringup_global.launch.py`
3. **Per-robot Nav2** (3 terminals, env_ros2):
   `ros2 launch deploy/nav2/bringup_robot_ns.launch.py namespace:=R1` (R2, R3)
4. **Bridge** (env_ros2, Postgres up, `.env` with ANTHROPIC key, from `agents/mars`):
   `python3 -m tools.isaac_multi_failure_bridge`
5. **Trigger** — drive ≥2 robots past the dock (so they can't reach → sustained abort):
   `ros2 action send_goal /R2/navigate_to_pose ... {x:8,y:0}` and `/R3 ... {x:8,y:-2}`
   → bridge logs 2 REAL aborts → `running brain` → `avoid_zone active = True` + mask.

Single-robot full loop: `demo.py --ros2-keepout` (see git history / README_keepout.md).

## 6. Known gotchas (all hit + solved)

- **Python:** Isaac Sim 5.1 wheels are **cp311** only → separate `python3.11` venv
  (`/workspace/isaac_venv311`); ROS2/supervisor run in system **py3.10** (rclpy).
- **DDS:** Isaac's bundled FastDDS ≠ system FastDDS → topic visible but `echo`
  hangs over SHM. Force **UDPv4-only** via `FASTRTPS_DEFAULT_PROFILES_FILE`
  (`fastdds_udp_only.xml`) on **both** sides.
- **Isaac ROS2 bridge:** strip system ROS2 from the Isaac shell (`env_isaac.sh`)
  so it loads its **internal** rclpy; `LD_LIBRARY_PATH` → ext `humble/lib`.
- **OmniGraph:** must `timeline.play()` + `world.step(render=True)` or
  `OnPlaybackTick` never fires (topic advertised, 0 messages).
- **Repeated `kill -9` Isaac → Vulkan/NGX startup hang** (cache clear doesn't fix;
  Pod restart does). Always `pkill -9 -f deploy/isaac` before relaunch; prefer
  background + log over Ctrl+Z.
- **Namespaced Nav2:** `static_layer map_topic:/map` (absolute) or costmaps don't
  get the shared map → "Robot is out of bounds".
- **Isaac restart resets sim clock** → restart Nav2 too or tf "jump back in time".
- **Abort trigger:** send goals *past* the dock (x=8), not the edge (4,±2) — edge
  goals land within `xy_goal_tolerance` of the stuck pose → false SUCCEEDED.
- **Manual `/cmd_vel`:** publish a zero twist to stop, or the diff controller
  holds the last velocity (robot runs away off the map).
- **relied_on_precedents:** with the mock embedder, force it to `[]` (stale
  incident_embeddings make search_incidents non-empty → hallucinated precedent
  reliance → DV DEGRADE).
- **inflation_radius 0.25** (not 0.55) so a failure at the zone boundary doesn't
  trap the robot inside the inflated keepout.

## 7. Remaining work

- [ ] **Multi-robot reroute visual (TODO):** after the multi-robot avoid_zone
  activates, show R1/R2/R3 physically detouring around the dock. Today the
  failing robots are stuck at the dock box (failure goal) so the post-keepout
  detour wasn't captured. Mechanism is proven (2b single-robot, y≈2.5);
  needs clean repositioning of the robots, then re-issue cross-dock goals.
- [ ] **Warehouse visual + recording (item 3):** run scenes with `--warehouse`
  (full_warehouse.usd) and wire **offline** Isaac camera capture → mp4
  (recording, NOT livestream). Demos are recorded, not streamed.
- [ ] **Natural failure trigger:** failures are induced (dock box + goal).
  A more organic scenario is future work.
- [ ] **M1 strategy grounding:** strategy occasionally cites nested
  `incident_analysis.*` refs that don't resolve (non-blocking; avoid_zone still
  activates via the validated path). Prompt enumerates valid sub-fields now.
- [ ] **Pod persistence:** community pod loses `/workspace` on delete — env is
  scriptable (setup_phase_b/postgres/install_isaac_pip), not persistent.

## 8. Rebuild on a fresh pod

```bash
cd /workspace && git clone https://github.com/vanillaturtlechips/MARS.git
cd MARS && git checkout feature/mars-msgs-interfaces
bash deploy/runpod/setup_phase_b.sh        # ROS2+Nav2+mars_msgs + py3.11 Isaac (~20-30 min)
bash deploy/runpod/setup_postgres.sh       # only for the brain (Phase A / bridges)
python deploy/nav2/make_empty_map.py
```
Then the run procedure (§5). Code is all on GitHub; nothing is lost by stopping
the pod — only the running processes + the ephemeral `/workspace`.
