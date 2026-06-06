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
  killall_; rm -rf "$FR" /tmp/keepout_isaac_ready /tmp/keepout_record_go /tmp/keepout_zone_go
  bash -c "source $IENV && cd $REPO && exec stdbuf -oL -eL python -u deploy/isaac/isaac_multi_robot_ros2.py --warehouse $1" > "$L/isaac.log" 2>&1 &
  ISAAC_PID=$!
  echo "  waiting for Isaac up (cold ~2-3min)..."; local i=0
  until [ -f /tmp/keepout_isaac_ready ]; do sleep 3; i=$((i+3)); kill -0 "$ISAAC_PID" 2>/dev/null || { echo "  Isaac died ($L/isaac.log)"; return 1; }; [ $i -ge 600 ] && { echo "  Isaac timeout"; return 1; }; done
  bash -c "source $RENV; \
    ros2 run tf2_ros static_transform_publisher --x 8 --y 5 --z 0 --frame-id map --child-frame-id R1/odom & \
    ros2 run tf2_ros static_transform_publisher --x 11 --y 5 --z 0 --frame-id map --child-frame-id R2/odom & \
    ros2 run tf2_ros static_transform_publisher --x 5 --y 5 --z 0 --frame-id map --child-frame-id R3/odom & wait" > "$L/stf.log" 2>&1 &
  sleep 5
  bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_global.launch.py" > "$L/nav2_global.log" 2>&1 &
  grepwait "$L/nav2_global.log" "Managed nodes are active" 120 || return 1
  for r in R1 R2 R3; do
    local ok=0
    for attempt in 1 2; do
      bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_robot_ns.launch.py namespace:=$r" > "$L/nav2_$r.log" 2>&1 &
      if grepwait "$L/nav2_$r.log" "Managed nodes are active" 200; then ok=1; break; fi
      echo "  $r nav2 attempt $attempt timed out (contention); retrying"
      pkill -9 -f "namespace:=$r"; pkill -9 -f "__ns:=/$r"; sleep 4
    done
    [ "$ok" = 1 ] || return 1
    sleep 5
  done
}
enc(){ pkill -9 -f "action send_goal"; sleep 2; local n; n=$(ls "$FR"/frame_*.png 2>/dev/null | wc -l); echo "  $n frames -> $1"; [ "$n" -gt 0 ] && ffmpeg -y -framerate 20 -i "$FR/frame_%06d.png" -pix_fmt yuv420p "$1" >/dev/null 2>&1; }

demo1(){
  say "DEMO 1 — keepout (agents flag a failing zone, fleet avoids it)"
  killall_   # kill stale bridge FIRST so nothing holds a lock on the tables
  # simple form (worked in run_keepout); killall above prevents the lock hang, timeout is a safety net
  timeout 25 su - postgres -c "psql -d warehouse -c 'TRUNCATE incident_embeddings, failures, diagnoses, outcomes RESTART IDENTITY CASCADE;'" >/dev/null 2>&1 || true
  bringup "--record /workspace/demo1_keepout.mp4" || { echo "  DEMO1 skipped (bringup failed)"; return 1; }
  bash -c "source $RENV && cd $REPO/agents/mars && exec stdbuf -oL -eL python3 -u -m tools.isaac_multi_failure_bridge" > "$L/bridge.log" 2>&1 &
  grepwait "$L/bridge.log" "listening for aborts" 60 || true
  sleep 3
  touch /tmp/keepout_record_go    # start camera capture now that Nav2 is up (no bringup contention)
  echo "  trigger: R1 drives UP aisle-8 into the box at (8,13) -> stuck between the shelves"
  goal R1 8 16                       # R1 up aisle-8; rams the box at (8,13), aborts (stuck)
  grepwait "$L/bridge.log" "avoid_zone active for receiving_dock = True" 150 || echo "  [warn] avoid_zone not confirmed"
  touch /tmp/keepout_zone_go    # agent banned aisle-8 -> reveal the red keepout slab on camera
  echo "  reroute: R2,R3 take the clear aisle-13 instead (agent banned aisle-8)"
  sleep 3
  goal R2 13 16          # R2 rerouted up the clear aisle-13
  sleep 4
  goal R3 13 11          # R3 also rerouted to aisle-13 (shallower so they don't overlap)
  sleep 45
  kill -INT "$ISAAC_PID" 2>/dev/null; sleep 6
  enc /workspace/demo1_keepout.mp4
}
# charge_demo <scenario> <mp4> : bring up, then let the REAL charging bridge
# decide the serve order and drive the robots. The bridge runs in the foreground
# so we only stop Isaac once its sequence is done.
charge_demo(){
  local scn="$1" out="$2"
  killall_
  timeout 25 su - postgres -c "psql -d warehouse -c 'TRUNCATE incident_embeddings, failures, diagnoses, outcomes RESTART IDENTITY CASCADE;'" >/dev/null 2>&1 || true
  # view centered on the charging station (x≈12) + the robots in the open band
  bringup "--record $out --cam-eye 10,-12,9 --cam-target 10,4,0.6" || { echo "  bringup failed for $scn"; return 1; }
  touch /tmp/keepout_record_go
  echo "  running real charging arbitration bridge (scenario=$scn)"
  bash -c "source $RENV && cd $REPO/agents/mars && exec stdbuf -oL -eL python3 -u -m tools.isaac_charging_bridge --scenario $scn" > "$L/charge_$scn.log" 2>&1
  kill -INT "$ISAAC_PID" 2>/dev/null; sleep 6
  enc "$out"
}
demo2(){
  say "DEMO 2 — charging (agent reserves the one charger for the CRITICAL robot, delays the low one)"
  charge_demo charging /workspace/demo2_charging.mp4
}
demo3(){
  say "DEMO 3 — priority (two robots want the same charger; the queue serializes them, no deadlock)"
  charge_demo priority /workspace/demo3_priority.mp4
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
echo "agent arbitration (demo2/3): grep -aE 'charging-svc|serve order|granted|fleet-llm' $L/charge_charging.log $L/charge_priority.log"
