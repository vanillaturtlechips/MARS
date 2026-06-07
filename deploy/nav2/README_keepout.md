# Agent-driven Nav2 keepout (avoid_zone → costmap filter)

How the MARS supervisory agents make robots physically route around a zone.

```
FailureAnalysisAgent (Haiku, ReAct)  →  zone_wide diagnosis
  → OperationsStrategyAgent           →  avoid_zone policy
  → DecisionValidator + Guardrail     →  ACCEPT
  → PolicyManager.activate            →  fires consumers
      ├─ SchedulingService            →  stops dispatching missions into the zone
      └─ KeepoutService               →  rasterizes the zone polygon to an
                                          OccupancyGrid and calls
                                          nav.publish_keepout_mask(grid)
  → ROS2SimAdapter.publish_keepout_mask
      → /keepout_filter_mask  (nav_msgs/OccupancyGrid, latched)
  → Nav2 KeepoutFilter (global_costmap)  →  masked cells become lethal
  → global planner routes around the zone
```

The mask is **dynamic**: the supervisor publishes a fresh mask on every
avoid_zone activate/deactivate (mask is cleared to all-free when no zone is
avoided). No static mask map_server is needed.

## Files

| File | Role |
|------|------|
| `keepout_costmap_filter_info.yaml` | `costmap_filter_info_server` params (type=0 keepout, mask_topic=`/keepout_filter_mask`, filter_info_topic=`/costmap_filter_info`) |
| `keepout_filter.launch.py` | launches `costmap_filter_info_server` + lifecycle manager |
| `global_costmap_keepout_snippet.yaml` | add-in: `keepout_filter` plugin for each robot's `global_costmap` |

The mask publisher itself lives in the supervisor:
`agents/mars/mars/ros/isaac_sim_adapter.py` → `ROS2SimAdapter.publish_keepout_mask`
(topic constant `KEEPOUT_MASK_TOPIC = "/keepout_filter_mask"`).

## Bring-up (RunPod, ROS2 Humble + Nav2)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash          # mars_msgs

# 1) Isaac Sim warehouse scene publishing /tf (map -> <robot>/base_link),
#    /clock, /<robot>/odom, /<robot>/battery_state, /<robot>/robot_health,
#    and accepting /<robot>/navigate_to_pose. (use_sim_time=True everywhere)

# 2) Nav2 bringup per robot, with the keepout_filter merged into each
#    global_costmap (see global_costmap_keepout_snippet.yaml).

# 3) Keepout filter info server
ros2 launch deploy/nav2/keepout_filter.launch.py

# 4) The MARS supervisor (mask publisher + the agent brain)
cd agents/mars && python -m mars.ros.ros2_node
```

## Verify

1. **Filter info is alive:**
   ```bash
   ros2 topic echo /costmap_filter_info --once     # CostmapFilterInfo, type=0
   ```
2. **Mask publishes when an agent decides avoid_zone.** Trigger a zone-wide
   failure (Isaac scene or fault injection), then:
   ```bash
   ros2 topic echo /keepout_filter_mask --once     # OccupancyGrid, 100s in the zone
   ```
   Supervisor log shows: `[ros2_adapter] published keepout mask: NxM, K keepout cells`.
3. **Costmap shows the keepout.** In RViz add the global costmap; the avoided
   zone turns lethal (the keepout footprint appears) after activation.
4. **Planner routes around it.** Request a path through the zone before vs after
   activation:
   ```bash
   ros2 action send_goal /<robot>/compute_path_to_pose nav2_msgs/action/ComputePathToPose "{...}"
   ```
   The post-activation path bends around the keepout. With a moving robot the
   NavigateToPose goal follows the detour.

## Notes / gotchas

- `use_sim_time: true` must be consistent across Isaac, Nav2, filter server, and
  the supervisor, or the latched mask timestamp will look stale.
- Costmaps may use only `static_layer` + `keepout_filter` + `inflation_layer`
  (no `obstacle_layer`) — keepout needs **no laser**. Dynamic obstacle avoidance
  is handled by the RL policy, not Nav2.
- The mask covers only the avoided polygons + margin; cells outside the mask are
  treated as free (no keepout), which is correct.
- **Not yet verified on hardware/sim** — these are the RunPod integration
  artifacts. The policy→mask core is unit-tested locally (tests/test_keepout*.py)
  and exercised end-to-end with real Haiku in `mars.orchestrator.demo`.
