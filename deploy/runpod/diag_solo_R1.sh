#!/usr/bin/env bash
# DIAGNOSTIC (not a demo): bring up ONLY R1's Nav2 stack against the same 3-robot
# Isaac sim, send R1 one goal straight up aisle x=-8, and stream R1's odom.
#
# Splits the two live hypotheses for "robots abort follow_path and never move"
# (seen with 3 stacks: follow_path ActionServer aborting every ~1s, bt_navigator
# timing out waiting for follow_path to ACK, costmap clear services timing out,
# R3 lifecycle_manager dying -9 — all on an idle 48-core box = a middleware/DDS
# problem, not compute):
#   * R1 odom CLIMBS (y 0 -> several m) => single stack is healthy => the 3-stack
#     DDS endpoint load over the UDP-only FastDDS profile is what kills the demo.
#   * R1 STILL aborts with odom ~0 => it's the single-stack costmap/TF/layout
#     (e.g. inflation_radius 0.25 < inscribed 0.353, footprint, or map->base_link tf).
# No bridge, no keepout, no recording — pure "can ONE robot drive" test.
set -u
REPO=/workspace/MARS
RENV="$REPO/deploy/isaac/env_ros2.sh"
IENV="$REPO/deploy/isaac/env_isaac.sh"
L=/tmp/solo; mkdir -p "$L"
kill_(){ pkill -9 -f deploy/isaac; pkill -9 -f /opt/ros/humble/lib/nav2; pkill -9 -f static_transform_publisher; pkill -9 -f isaac_multi_failure_bridge; pkill -9 -f "topic echo"; pkill -9 -f "action send_goal"; }

kill_; sleep 3; rm -f /tmp/keepout_isaac_ready
echo "[solo] starting Isaac (3 robots in sim, but we nav only R1)..."
bash -c "source $IENV && cd $REPO && exec stdbuf -oL -eL python -u deploy/isaac/isaac_multi_robot_ros2.py --warehouse" > "$L/isaac.log" 2>&1 &
until [ -f /tmp/keepout_isaac_ready ]; do sleep 3; done
echo "[solo] Isaac up."

bash -c "source $RENV; exec ros2 run tf2_ros static_transform_publisher --x -8 --y 8 --z 0 --frame-id map --child-frame-id R1/odom" > "$L/stf.log" 2>&1 &
sleep 5
bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_global.launch.py" > "$L/global.log" 2>&1 &
echo "[solo] waiting global nav2 active..."; until grep -q "Managed nodes are active" "$L/global.log" 2>/dev/null; do sleep 2; done
bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_robot_ns.launch.py namespace:=R1" > "$L/R1.log" 2>&1 &
echo "[solo] waiting R1 nav2 active..."; until grep -q "Managed nodes are active" "$L/R1.log" 2>/dev/null; do sleep 2; done
sleep 3

bash -c "source $RENV; exec stdbuf -oL ros2 topic echo /R1/odom --field pose.pose.position" > "$L/odom_R1.log" 2>&1 &
echo "[solo] sending R1 goal up aisle x=-8 to (-8,18) — blocks until SUCCEEDED/ABORTED ..."
bash -c "source $RENV; ros2 action send_goal /R1/navigate_to_pose nav2_msgs/action/NavigateToPose '{pose: {header: {frame_id: map}, pose: {position: {x: -8.0, y: 18.0}, orientation: {w: 1.0}}}}'" 2>&1 | tail -8

echo ""; echo "===== R1 odom (start vs end — climbing = R1 drives fine) ====="
head -5 "$L/odom_R1.log"; echo "   ... (end) ..."; tail -6 "$L/odom_R1.log"
echo ""; echo "===== R1 controller/bt errors (sans tick-rate noise) ====="
grep -aiE "abort|collision|fail|unable|tolerance|progress|timed out|in collision|recovery" "$L/R1.log" | grep -vi "tick rate" | tail -25
kill_
echo ""; echo "[solo] done. full logs in $L/"
