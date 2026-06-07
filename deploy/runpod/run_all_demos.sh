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
killall_(){ pkill -9 -f deploy/isaac; pkill -9 -f /opt/ros/humble/lib/nav2; pkill -9 -f isaac_multi_failure_bridge; pkill -9 -f static_transform_publisher; pkill -9 -f "action send_goal"; pkill -9 -f "topic echo"; pkill -9 -f follow_waypoints; sleep 3; }
# background odom logger: prove whether each robot PHYSICALLY moves (position in the
# R*/odom frame, zeroed at spawn -> grows as the robot drives). Files: $L/odom_R*.log
odomlog(){ for r in R1 R2 R3; do bash -c "source $RENV; exec stdbuf -oL ros2 topic echo /$r/odom --field pose.pose.position" > "$L/odom_$r.log" 2>&1 & done; }
# follow <ns> <wp_odom...> : drive a NON-Nav2 robot along odom-frame waypoints via
# direct cmd_vel (closed loop on /<ns>/odom). Lets demo1 run only ONE Nav2 stack
# (the hero R1) so the multi-stack FastDDS discovery contention can't kill robots.
follow(){ local ns="$1"; shift; bash -c "source $RENV && cd $REPO && exec python3 -u deploy/isaac/follow_waypoints.py $ns $*" > "$L/follow_$ns.log" 2>&1 & }
# block until R1 has PHYSICALLY driven up to the box (odom y in the R1/odom frame,
# spawn -8,11 -> box at world y=15 is odom y~4; wait for >2.8 = world ~13.8). Gates the
# whole reroute on the real collision so it can NEVER look preemptive. Needs odomlog running.
wait_at_box(){ local i=0 y; echo "  waiting for R1 to reach the box (odom)..."; while :; do
    y=$(grep '^y:' "$L/odom_R1.log" 2>/dev/null | tail -1 | awk '{print $2+0}')
    awk "BEGIN{exit !(${y:-0} > 2.8)}" && { echo "  R1 reached the box (odom y=$y)"; return 0; }
    sleep 2; i=$((i+2)); [ $i -ge 180 ] && { echo "  [warn] R1-at-box wait timed out"; return 0; }
  done; }

# bringup <isaac extra args> ; sets ISAAC_PID. returns 1 on failure.
bringup(){
  killall_; rm -rf "$FR" /tmp/keepout_isaac_ready /tmp/keepout_record_go /tmp/keepout_zone_go
  bash -c "source $IENV && cd $REPO && exec stdbuf -oL -eL python -u deploy/isaac/isaac_multi_robot_ros2.py --warehouse $1" > "$L/isaac.log" 2>&1 &
  ISAAC_PID=$!
  echo "  waiting for Isaac up (cold ~2-3min)..."; local i=0
  until [ -f /tmp/keepout_isaac_ready ]; do sleep 3; i=$((i+3)); kill -0 "$ISAAC_PID" 2>/dev/null || { echo "  Isaac died ($L/isaac.log)"; return 1; }; [ $i -ge 600 ] && { echo "  Isaac timeout"; return 1; }; done
  # map->R*/odom offsets MUST match the sim spawns. Defaults = demo1 aisle spawns;
  # charge_demo overrides via SPAWN_R* to the charging layout (south of the pad).
  local s1="${SPAWN_R1:--8 11}" s2="${SPAWN_R2:--7 8}" s3="${SPAWN_R3:--9 8}"
  bash -c "source $RENV; \
    ros2 run tf2_ros static_transform_publisher --x ${s1%% *} --y ${s1##* } --z 0 --frame-id map --child-frame-id R1/odom & \
    ros2 run tf2_ros static_transform_publisher --x ${s2%% *} --y ${s2##* } --z 0 --frame-id map --child-frame-id R2/odom & \
    ros2 run tf2_ros static_transform_publisher --x ${s3%% *} --y ${s3##* } --z 0 --frame-id map --child-frame-id R3/odom & wait" > "$L/stf.log" 2>&1 &
  sleep 5
  bash -c "source $RENV && cd $REPO && exec stdbuf -oL -eL ros2 launch deploy/nav2/bringup_global.launch.py" > "$L/nav2_global.log" 2>&1 &
  grepwait "$L/nav2_global.log" "Managed nodes are active" 120 || return 1
  # which robots get a full Nav2 stack. demo1 sets this to just "R1" (the hero whose
  # real failure drives avoid_zone); R2/R3 are follower-driven so we don't pay the
  # multi-stack DDS contention. charge demos leave the default (all three).
  for r in ${NAV_ROBOTS:-R1 R2 R3}; do
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
  # ONLY R1 gets a Nav2 stack (its real failure drives avoid_zone); R2/R3 are
  # follower-driven. One Nav2 stack = no multi-stack FastDDS discovery contention.
  NAV_ROBOTS="R1" bringup "--record /workspace/demo1_keepout.mp4" || { echo "  DEMO1 skipped (bringup failed)"; return 1; }
  bash -c "source $RENV && cd $REPO/agents/mars && exec stdbuf -oL -eL python3 -u -m tools.isaac_multi_failure_bridge" > "$L/bridge.log" 2>&1 &
  grepwait "$L/bridge.log" "listening for aborts" 60 || true
  sleep 3
  touch /tmp/keepout_record_go    # start camera capture now that Nav2 is up (no bringup contention)
  odomlog                          # diag: record each robot's odom position for the whole demo
  echo "  R1 drives up aisle x=-8 to the box; R2,R3 HOLD at the aisle mouth (don't move yet)"
  # ONLY R1 moves first. Everything else is gated on R1 PHYSICALLY reaching the box,
  # so the avoidance is unambiguously REACTIVE (after the crash), never preemptive.
  goal R1 -8 17                 # R1 alone drives up to the box at (-8,15) and rams it
  wait_at_box                   # <-- block until R1 has actually driven up and hit the box
  grepwait "$L/bridge.log" "avoid_zone active for receiving_dock = True" 90 || echo "  [warn] avoid_zone not confirmed"
  touch /tmp/keepout_zone_go    # R1 hit the box -> agent declared avoid_zone -> red slab appears
  sleep 5                       # hold on the crash + red zone (R2/R3 still parked) so it reads as the trigger
  echo "  NOW R2,R3 react -> CONVOY left to aisle x=-13 (R3 leads, R2 follows behind)"
  # CONVOY (not parallel) so they can't collide: R3 goes first all the way up the left
  # aisle; ~14s later R2 takes the SAME lane and stops below R3. Odom wp = world - spawn
  # (R3 spawn -9,8 ; R2 spawn -7,8). Lane: south to y=7, west to x=-13, then up.
  follow R3 0,-1 -4,-1 -4,7     # R3: world (-9,7)->(-13,7)->(-13,15)
  sleep 14                      # let R3 clear the lane before R2 enters it
  follow R2 -2,-1 -6,-1 -6,3    # R2: world (-9,7)->(-13,7)->(-13,11)  (joins R3's lane, stops lower)
  sleep 60                      # closed-loop followers; generous time for the (slow-sim) reroute
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
  # view centered on the REAL charging area: station at (0,3), dock (0,5), robots
  # approach from the aisle (x=-8) and park along y=3. (Old cam aimed at x=10 = empty.)
  # Charging layout: robots spawn SPREAD just south of the pad (--charge) so they don't
  # bump and dock quickly; static TF offsets must match those spawns.
  export SPAWN_R1="0 2.5" SPAWN_R2="3 2" SPAWN_R3="-4 2"
  # Camera INSIDE the building (y=-2; charge_5/charge_3 at y=-7/-10 were OUTSIDE the
  # south wall -> grey). Wide overview pitched down onto the charging band.
  bringup "--record $out --charge --cam-eye=0,-2,7 --cam-target=0,4,0.4" || { echo "  bringup failed for $scn"; return 1; }
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
# raise UDP socket buffer ceiling so FastDDS can actually take the 16MB buffers
# from fastdds_udp_only.xml (else the kernel clamps it and 3 Nav2 stacks drop
# large-costmap fragments -> follow_path/costmap/bond timeouts; single stack OK).
sysctl -w net.core.rmem_max=16777216 net.core.wmem_max=16777216 >/dev/null 2>&1 || true

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
