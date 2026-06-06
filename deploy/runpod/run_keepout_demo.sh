#!/usr/bin/env bash
# One-command Phase 4-B demo: robots fail at the dock -> the supervisory AGENTS
# (FailureAnalysis/OperationsStrategy/validators/guardrail/PolicyManager via the
# brain bridge) decide an avoid_zone -> Nav2 keepout -> a robot reroutes around
# it. Headless Isaac camera records it all to mp4.
#
#   bash deploy/runpod/run_keepout_demo.sh [out.mp4]
#
# Watch live:  tail -f /tmp/keepdemo/bridge.log   (the agent pipeline)
set -u
REPO=/workspace/MARS
OUT="${1:-/workspace/keepout_demo.mp4}"
IENV="$REPO/deploy/isaac/env_isaac.sh"
RENV="$REPO/deploy/isaac/env_ros2.sh"
L=/tmp/keepdemo; mkdir -p "$L"; rm -f "$OUT"

say(){ echo -e "\n=== $* ==="; }
waitfor(){ # <logfile> <pattern> <timeout_s>
  local f="$1" pat="$2" to="${3:-60}" i=0
  while ! grep -q "$pat" "$f" 2>/dev/null; do
    sleep 2; i=$((i+2))
    [ $i -ge "$to" ] && { echo "[TIMEOUT after ${to}s] '$pat'  (see $f)"; return 1; }
  done
  echo "[ok] $pat"
}
goal(){ # <ns> <x> <y>
  bash -c "source $RENV && ros2 action send_goal /$1/navigate_to_pose nav2_msgs/action/NavigateToPose \"{pose: {header: {frame_id: map}, pose: {position: {x: $2, y: $3}, orientation: {w: 1.0}}}}\"" >> "$L/goals.log" 2>&1 &
}

say "0. kill stale + deps"
pkill -9 -f deploy/isaac; pkill -9 -f /opt/ros/humble/lib/nav2
pkill -9 -f isaac_multi_failure_bridge; pkill -9 -f static_transform_publisher
pkill -9 -f "action send_goal"; pkill -9 -f keepout_publish_test
rm -rf /tmp/keepout_frames    # clear old frames so the readiness-wait only sees this run's
sleep 3
command -v ffmpeg >/dev/null || apt-get install -y ffmpeg >/dev/null 2>&1 || true
service postgresql start >/dev/null 2>&1 || true

say "1. Isaac (+record) — wait for play (heavy USD, up to ~3 min)"
bash -c "source $IENV && cd $REPO && exec stdbuf -oL -eL python -u deploy/isaac/isaac_multi_robot_ros2.py --warehouse --record '$OUT'" > "$L/isaac.log" 2>&1 &
ISAAC_PID=$!
# Isaac buffers its log, so grepping it for readiness is unreliable. Wait on the
# camera's frame files instead (they appear once it's up AND recording).
echo "waiting for Isaac to come up + record (cold pod can be ~2-3 min)..."
i=0; until ls /tmp/keepout_frames/frame_*.png >/dev/null 2>&1; do
  sleep 3; i=$((i+3))
  kill -0 "$ISAAC_PID" 2>/dev/null || { echo "Isaac died; see $L/isaac.log"; exit 1; }
  [ $i -ge 600 ] && { echo "Isaac timeout; see $L/isaac.log"; exit 1; }
done
echo "[ok] Isaac up + recording"
sleep 3

say "2. static tf (spawn offsets map->R*/odom)"
bash -c "source $RENV && \
  ros2 run tf2_ros static_transform_publisher --x -3 --y 5 --z 0 --frame-id map --child-frame-id R1/odom & \
  ros2 run tf2_ros static_transform_publisher --x 0 --y 5 --z 0 --frame-id map --child-frame-id R2/odom & \
  ros2 run tf2_ros static_transform_publisher --x 3 --y 5 --z 0 --frame-id map --child-frame-id R3/odom & wait" > "$L/stf.log" 2>&1 &
sleep 3

say "3. global Nav2 (map_server first)"
bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_global.launch.py" > "$L/nav2_global.log" 2>&1 &
waitfor "$L/nav2_global.log" "Managed nodes are active" 90 || exit 1

say "4. per-robot Nav2 (R1/R2/R3)"
for r in R1 R2 R3; do
  bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_robot_ns.launch.py namespace:=$r" > "$L/nav2_$r.log" 2>&1 &
  waitfor "$L/nav2_$r.log" "Managed nodes are active" 150 || exit 1
done

say "5. brain bridge (ALL agents)"
bash -c "source $RENV && cd $REPO/agents/mars && exec stdbuf -oL -eL python3 -u -m tools.isaac_multi_failure_bridge" > "$L/bridge.log" 2>&1 &
waitfor "$L/bridge.log" "listening for aborts" 60 || exit 1
sleep 3

say "6. TRIGGER — send R2,R3 into the dock (0,0) so they FAIL"
goal R2 0 -8; sleep 5; goal R3 0 -8
echo "   waiting for the agents to decide avoid_zone (Haiku ~15-30s)..."
waitfor "$L/bridge.log" "avoid_zone active for receiving_dock = True" 150 \
  || echo "[warn] avoid_zone not confirmed — check $L/bridge.log (continuing)"

say "7. REROUTE — send R1 across the dock; it detours around the agents' keepout"
sleep 3; goal R1 3 -8
echo "   letting R1 detour + arrive (40s)..."
sleep 40

say "8. finalize -> encode mp4 (runner does it; Isaac intercepts SIGINT so we don't rely on its finally)"
kill -INT "$ISAAC_PID" 2>/dev/null; sleep 8
FR=/tmp/keepout_frames
N=$(ls "$FR"/frame_*.png 2>/dev/null | wc -l)
echo "captured $N frames in $FR"
[ "$N" -gt 0 ] && ffmpeg -y -framerate 20 -i "$FR/frame_%06d.png" -pix_fmt yuv420p "$OUT" >/dev/null 2>&1

say "DONE"
if [ -f "$OUT" ]; then echo "VIDEO: $OUT  ($(du -h "$OUT" | cut -f1))"; else echo "[!] no mp4 — see $L/isaac.log"; fi
echo
echo "Agent pipeline (the differentiator) — every agent's role:"
grep -aE "REAL abort|running brain|investigator|decision_validator|strategy_trigger|ops_strategy|retrieval_validator|guardrail|policy_manager|keepout|avoid_zone" "$L/bridge.log" | tail -40
echo
echo "logs: $L/  (isaac.log bridge.log nav2_*.log goals.log)"
