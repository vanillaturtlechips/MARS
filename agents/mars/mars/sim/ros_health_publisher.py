"""
ROS2 self-publish node — the "SELF" sources from the §1a topic table.

Isaac Sim + Nav2 give you /tf, /odom, /scan and the NavigateToPose action, but
NOTHING publishes battery or health — those are warehouse-supervisor concerns,
not physics.  This node fills that gap and doubles as the fault injector for the
robot-internal demo scenario.

Publishes (per robot, 1 Hz):
    /<robot_id>/robot_health   mars_msgs/RobotHealth
    /<robot_id>/battery_state  sensor_msgs/BatteryState   (percentage 0-1)

Fault injection (subscribe, std_msgs/String on /mars/fault_cmd):
    "<robot_id> ERROR <CODE>"   -> level=ERROR, fault_codes=[CODE]   (-> fault_flag)
    "<robot_id> ESTOP"          -> estop_active=True, level=ERROR     (-> fault_flag)
    "<robot_id> BATTERY <pct>"  -> set battery percent (e.g. 12)      (-> fault_flag if < CRITICAL)
    "<robot_id> WARN <CODE>"    -> level=WARN (no fault_flag)
    "<robot_id> CLEAR"          -> back to healthy

Trigger from the CLI, e.g.:
    ros2 topic pub --once /mars/fault_cmd std_msgs/String '{data: "R3 ERROR MOTOR_FAULT"}'

Run (after sourcing ROS2 + the mars_msgs workspace):
    python -m mars.sim.ros_health_publisher --ros-args -p robot_ids:="[R1,R2,R3,R4,R5]"
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_DEFAULT_ROBOTS = ["R1", "R2", "R3", "R4", "R5"]
# Idle discharge per tick (% of full).  Demo-scale; battery is a knob, not physics.
_DRAIN_PER_TICK = 0.0


class _RobotHealthState:
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.level = 0                 # 0=OK 1=WARN 2=ERROR
        self.estop_active = False
        self.fault_codes: list[str] = []
        self.battery_pct = 100.0       # 0-100 here; published as 0-1
        self.charging = False

    def clear(self) -> None:
        self.level = 0
        self.estop_active = False
        self.fault_codes = []


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from sensor_msgs.msg import BatteryState
    except ImportError:
        log.error("rclpy/std_msgs/sensor_msgs not available — source ROS2 Humble first.")
        raise SystemExit(1)

    try:
        from mars_msgs.msg import RobotHealth
    except ImportError:
        log.error(
            "mars_msgs not on the path.  Build + source it first:\n"
            "  cd ros2_ws && colcon build --packages-select mars_msgs\n"
            "  source install/setup.bash"
        )
        raise SystemExit(1)

    rclpy.init()
    # use_sim_time는 기본 False(wall clock). True면 /clock 이 있어야만 타이머가 도는데,
    # Isaac Sim 없이 standalone 실행 시 /clock 이 없어 1Hz 타이머가 영영 안 돈다(발행 0건).
    # Isaac Sim(+/clock)과 함께 돌릴 때만 override:
    #   python3 -m mars.sim.ros_health_publisher --ros-args -p use_sim_time:=true
    node = rclpy.create_node("mars_sim_health")
    node.declare_parameter("robot_ids", _DEFAULT_ROBOTS)
    robot_ids = list(node.get_parameter("robot_ids").value) or _DEFAULT_ROBOTS

    states = {rid: _RobotHealthState(rid) for rid in robot_ids}
    health_pubs = {
        rid: node.create_publisher(RobotHealth, f"/{rid}/robot_health", 10)
        for rid in robot_ids
    }
    battery_pubs = {
        rid: node.create_publisher(BatteryState, f"/{rid}/battery_state", 10)
        for rid in robot_ids
    }

    def on_fault_cmd(msg) -> None:
        parts = msg.data.split()
        if not parts:
            return
        rid = parts[0]
        st = states.get(rid)
        if st is None:
            node.get_logger().warn(f"fault_cmd: unknown robot {rid!r}")
            return
        cmd = parts[1].upper() if len(parts) > 1 else "CLEAR"

        if cmd == "ERROR":
            st.level = 2
            if len(parts) > 2:
                st.fault_codes = [parts[2]]
        elif cmd == "WARN":
            st.level = 1
            if len(parts) > 2:
                st.fault_codes = [parts[2]]
        elif cmd == "ESTOP":
            st.estop_active = True
            st.level = 2
        elif cmd == "BATTERY":
            try:
                st.battery_pct = float(parts[2])
            except (IndexError, ValueError):
                node.get_logger().warn("fault_cmd BATTERY needs a percent, e.g. 'R3 BATTERY 12'")
        elif cmd == "CLEAR":
            st.clear()
        else:
            node.get_logger().warn(f"fault_cmd: unknown command {cmd!r}")
            return
        node.get_logger().info(
            f"[inject] {rid}: level={st.level} estop={st.estop_active} "
            f"codes={st.fault_codes} batt={st.battery_pct:.0f}"
        )

    node.create_subscription(String, "/mars/fault_cmd", on_fault_cmd, 10)

    def tick() -> None:
        now = node.get_clock().now().to_msg()
        for rid, st in states.items():
            if _DRAIN_PER_TICK and not st.charging:
                st.battery_pct = max(0.0, st.battery_pct - _DRAIN_PER_TICK)

            h = RobotHealth()
            h.header.stamp = now
            h.header.frame_id = rid
            h.robot_id = rid
            h.level = st.level
            h.estop_active = st.estop_active
            h.fault_codes = list(st.fault_codes)
            health_pubs[rid].publish(h)

            b = BatteryState()
            b.header.stamp = now
            b.header.frame_id = rid
            b.percentage = st.battery_pct / 100.0
            b.power_supply_status = 1 if st.charging else 2  # 1=CHARGING 2=DISCHARGING
            battery_pubs[rid].publish(b)

    node.create_timer(1.0, tick)
    node.get_logger().info(
        f"mars_sim_health publishing for {robot_ids}; inject via /mars/fault_cmd"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
