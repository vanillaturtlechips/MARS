#!/usr/bin/env bash
# Three agent-capability demos, each INDEPENDENT (one failing never blocks the
# others) and each recorded to its own mp4 in /workspace.
#   1) keepout  : 2 robots fail at a dock -> the AGENTS flag the zone (avoid_zone)
#                 -> a 3rd robot reroutes around a zone it never entered.
#   2) charging : agent reserves a charger for the critical robot + delays a
#                 low-priority one -> critical charges, low-priority waits.
#   3) priority : 2 robots want the same spot -> agent orders them (no deadlock).
#
#   bash deploy/runpod/run_all_demos.sh          # all three
#   bash deploy/runpod/run_all_demos.sh 2        # just demo 2 (1/2/3)
# Output: /workspace/demo1_keepout.mp4  demo2_charging.mp4  demo3_priority.mp4
set -u
REPO=/workspace/MARS
RENV="$REPO/deploy/isaac/env_ros2.sh"
IENV="$REPO/deploy/isaac/env_isaac.sh"
L=/tmp/alldemo; mkdir -p "$L"
FR=/tmp/keepout_frames
WANT="${1:-all}"

say(){ echo -e "\n########## $* ##########"; }
grepwait(){ local f="$1" pat="$2" to="${3:-300}" i=0; while ! grep -q "$pat" "$f" 2>/dev/null; do sleep 2; i=$((i+2)); [ $i -ge "$to" ] && { echo "[TIMEOUT] $pat ($f)"; return 1; }; done; echo "[ok] $pat"; }
goal(){ bash -c "source $RENV; ros2 action send_goal /$1/navigate_to_pose nav2_msgs/action/NavigateToPose \"{pose: {header: {frame_id: map}, pose: {position: {x: $2, y: $3}, orientation: {w: 1.0}}}}\"" >> "$L/goals.log" 2>&1 & }
killall_(){ pkill -9 -f deploy/isaac; pkill -9 -f /opt/ros/humble/lib/nav2; pkill -9 -f isaac_multi_failure_bridge; pkill -9 -f static_transform_publisher; pkill -9 -f "action send_goal"; sleep 3; }

# bringup <isaac extra args> ; sets ISAAC_PID. returns 1 on failure.
bringup(){
  killall_; rm -rf "$FR"
  bash -c "source $IENV && cd $REPO && exec stdbuf -oL -eL python -u deploy/isaac/isaac_multi_robot_ros2.py --warehouse $1" > "$L/isaac.log" 2>&1 &
  ISAAC_PID=$!
  echo "  waiting for Isaac+record (cold ~2-3min)..."; local i=0
  until ls "$FR"/frame_*.png >/dev/null 2>&1; do sleep 3; i=$((i+3)); kill -0 "$ISAAC_PID" 2>/dev/null || { echo "  Isaac died ($L/isaac.log)"; return 1; }; [ $i -ge 600 ] && { echo "  Isaac timeout"; return 1; }; done
  bash -c "source $RENV; \
    ros2 run tf2_ros static_transform_publisher --x -3 --y 5 --z 0 --frame-id map --child-frame-id R1/odom & \
    ros2 run tf2_ros static_transform_publisher --x 0 --y 5 --z 0 --frame-id map --child-frame-id R2/odom & \
    ros2 run tf2_ros static_transform_publisher --x 3 --y 5 --z 0 --frame-id map --child-frame-id R3/odom & wait" > "$L/stf.log" 2>&1 &
  sleep 5
  bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_global.launch.py" > "$L/nav2_global.log" 2>&1 &
  grepwait "$L/nav2_global.log" "Managed nodes are active" 120 || return 1
  for r in R1 R2 R3; do
    bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_robot_ns.launch.py namespace:=$r" > "$L/nav2_$r.log" 2>&1 &
    grepwait "$L/nav2_$r.log" "Managed nodes are active" 300 || return 1
    sleep 5
  done
}
enc(){ pkill -9 -f "action send_goal"; sleep 2; local n; n=$(ls "$FR"/frame_*.png 2>/dev/null | wc -l); echo "  $n frames -> $1"; [ "$n" -gt 0 ] && ffmpeg -y -framerate 20 -i "$FR/frame_%06d.png" -pix_fmt yuv420p "$1" >/dev/null 2>&1; }

demo1(){
  say "DEMO 1 — keepout (agents flag a failing zone, fleet avoids it)"
  su - postgres -c "psql -d warehouse -c 'TRUNCATE incident_embeddings, failures, diagnoses, outcomes RESTART IDENTITY CASCADE;'" >/dev/null 2>&1 || true
  bringup "--record /workspace/demo1_keepout.mp4" || { echo "  DEMO1 skipped (bringup failed)"; return 1; }
  bash -c "source $RENV && cd $REPO/agents/mars && exec stdbuf -oL -eL python3 -u -m tools.isaac_multi_failure_bridge" > "$L/bridge.log" 2>&1 &
  grepwait "$L/bridge.log" "listening for aborts" 60 || true
  sleep 3
  echo "  trigger: R2,R3 into dock (0,0) -> fail"
  goal R2 0 -8; sleep 5; goal R3 0 -8
  grepwait "$L/bridge.log" "avoid_zone active for receiving_dock = True" 150 || echo "  [warn] avoid_zone not confirmed"
  echo "  reroute: R1 across the dock"; sleep 3; goal R1 3 -8; sleep 40
  kill -INT "$ISAAC_PID" 2>/dev/null; sleep 6
  enc /workspace/demo1_keepout.mp4
}
demo2(){
  say "DEMO 2 — charging (reserve charger for critical, delay low-priority)"
  bringup "--no-obstacle --record /workspace/demo2_charging.mp4" || { echo "  DEMO2 skipped (bringup failed)"; return 1; }
  echo "  reserve (-3,-8) for critical R1; R3 -> (3,-8); R2 delayed"
  goal R1 -3 -8; sleep 2; goal R3 3 -8; sleep 18; goal R2 -3 -8; sleep 35
  kill -INT "$ISAAC_PID" 2>/dev/null; sleep 6
  enc /workspace/demo2_charging.mp4
}
demo3(){
  say "DEMO 3 — priority (two robots want the same spot; agent orders them)"
  bringup "--no-obstacle --record /workspace/demo3_priority.mp4" || { echo "  DEMO3 skipped (bringup failed)"; return 1; }
  echo "  R2 priority to (0,-6); R3 yields then goes"
  goal R2 0 -6; sleep 14; goal R2 0 -8; sleep 3; goal R3 0 -6; sleep 35
  kill -INT "$ISAAC_PID" 2>/dev/null; sleep 6
  enc /workspace/demo3_priority.mp4
}

command -v ffmpeg >/dev/null || apt-get install -y ffmpeg >/dev/null 2>&1 || true
service postgresql start >/dev/null 2>&1 || true

# each is independent: a failure in one does NOT stop the others
case "$WANT" in
  1) demo1 || true ;;
  2) demo2 || true ;;
  3) demo3 || true ;;
  *) demo1 || true; demo2 || true; demo3 || true ;;
esac

killall_
say "DONE"
ls -lh /workspace/demo1_keepout.mp4 /workspace/demo2_charging.mp4 /workspace/demo3_priority.mp4 2>/dev/null
echo "agent pipeline (demo1): grep -aE 'investigator|decision_validator|strategy|guardrail|policy_manager|keepout|avoid_zone' $L/bridge.log"
