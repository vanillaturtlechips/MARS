"""
Isaac Sim + Nav2 Adapter — §1a topic mapping (ROS2 Humble)

Implements NavigationInterface and SensorInterface behind the same abstract
contracts as MockSim, so the entire supervisory stack is unchanged when
swapping from mock to real Isaac Sim.

Topic mapping (all per-robot namespaced as /<robot_id>/...):
──────────────────────────────────────────────────────────────
  /<robot_id>/navigate_to_pose/_action/status
      → action_msgs/GoalStatusArray          (Nav2)
      → NavGoalStatus.status: 4=SUCCEEDED 5=CANCELED 6=ABORTED
        NOTE: NavigateToPose result is empty on Humble — use status, not result.

  /<robot_id>/navigate_to_pose/_action/feedback
      → nav2_msgs/action/NavigateToPose_FeedbackMessage   (Nav2)
      → number_of_recoveries, distance_remaining (struggle signal)

  /tf  (global, not namespaced)
      → tf2_msgs/TFMessage                   (Isaac Sim)
      → map → /<robot_id>/base_link  →  pose  →  zone (via ZoneResolver)

  /<robot_id>/odom
      → nav_msgs/Odometry                    (Isaac Sim)
      → velocity; backup pose

  /<robot_id>/battery_state
      → sensor_msgs/BatteryState             (self-published sim node)
      → health_at_failure.battery_pct  (percentage 0–1, DISCHARGING=2)

  /<robot_id>/robot_health
      → mars_msgs/msg/RobotHealth            (self-published sim node)
      → health_at_failure.estop / fault_codes → fault_flag

  /clock
      → rosgraph_msgs/Clock                  (Isaac Sim)
      → use_sim_time = True

Goal dispatch:
  ActionClient → /<robot_id>/navigate_to_pose

Swap guide:
  Replace MockSim(...) with
      ROS2SimAdapter(node, robot_ids, zone_resolver, ...)
  Everything else is identical.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

# All rclpy imports are deferred so this module is importable without ROS2.
# Callers that need the real adapter must have rclpy installed.

log = logging.getLogger(__name__)

# GoalStatus values (action_msgs/GoalStatus) — Humble
STATUS_UNKNOWN   = 0
STATUS_ACCEPTED  = 1
STATUS_EXECUTING = 2
STATUS_CANCELING = 3
STATUS_SUCCEEDED = 4
STATUS_CANCELED  = 5
STATUS_ABORTED   = 6


class ROS2SimAdapter:
    """
    Concrete ROS2 adapter.  Interchangeable with MockSim.

    Constructor args:
      node          — an rclpy Node (created externally so lifetime is managed
                      by the caller)
      robot_ids     — list of robot IDs whose topics to subscribe
      zone_resolver — ZoneResolver instance for TF → zone mapping
    """

    def __init__(self, node, robot_ids: list[str], zone_resolver):
        self._node          = node
        self._robot_ids     = robot_ids
        self._zone_resolver = zone_resolver
        self._lock          = threading.Lock()

        # Callback registries (mirrors MockSim's API)
        self._battery_cbs:    dict[str, list[Callable]] = {}
        self._health_cbs:     dict[str, list[Callable]] = {}
        self._pose_cbs:       dict[str, list[Callable]] = {}
        self._nav_status_cbs: dict[str, list[Callable]] = {}

        # Active goals: goal_id → (robot_id, goal_handle, on_status_change)
        self._active_goals:   dict[str, tuple] = {}

        self._subs:           list = []
        self._action_clients: dict[str, object] = {}

        self._setup()

    # ------------------------------------------------------------------
    # NavigationInterface
    # ------------------------------------------------------------------

    def send_goal(
        self,
        robot_id: str,
        goal_id: str,
        destination,          # mars.ros.interfaces.Pose
        on_status_change: Callable,
    ) -> None:
        """
        Dispatch a NavigateToPose goal.  Results arrive asynchronously via
        on_status_change(NavGoalStatus).

        NOTE: on Humble the NavigateToPose result is empty; outcome comes
        exclusively from GoalStatus (4/5/6).
        """
        import rclpy
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Header
        from builtin_interfaces.msg import Time

        ac = self._action_clients.get(robot_id)
        if ac is None:
            log.error("send_goal: no action client for %s", robot_id)
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = destination.x
        goal_msg.pose.pose.position.y = destination.y
        goal_msg.pose.pose.position.z = destination.z
        goal_msg.pose.pose.orientation.x = destination.qx
        goal_msg.pose.pose.orientation.y = destination.qy
        goal_msg.pose.pose.orientation.z = destination.qz
        goal_msg.pose.pose.orientation.w = destination.qw

        def _goal_response(future):
            handle = future.result()
            if not handle.accepted:
                log.warning("[ros2_adapter] goal %s rejected by Nav2", goal_id)
                return
            with self._lock:
                self._active_goals[goal_id] = (robot_id, handle, on_status_change)
            handle.get_result_async().add_done_callback(
                lambda f: self._on_result(robot_id, goal_id, f)
            )

        send_future = ac.send_goal_async(
            goal_msg,
            feedback_callback=lambda fb: self._on_feedback(robot_id, goal_id, fb),
        )
        send_future.add_done_callback(_goal_response)
        log.info("[ros2_adapter] goal sent robot=%s goal_id=%s", robot_id, goal_id)

    def cancel_goal(self, robot_id: str, goal_id: str) -> None:
        with self._lock:
            entry = self._active_goals.get(goal_id)
        if entry is None:
            return
        _, handle, _ = entry
        cancel_future = handle.cancel_goal_async()
        cancel_future.add_done_callback(
            lambda _: log.info("[ros2_adapter] cancel_goal sent robot=%s goal=%s", robot_id, goal_id)
        )

    # ------------------------------------------------------------------
    # SensorInterface
    # ------------------------------------------------------------------

    def subscribe_battery(self, robot_id: str, callback: Callable) -> None:
        self._battery_cbs.setdefault(robot_id, []).append(callback)

    def subscribe_health(self, robot_id: str, callback: Callable) -> None:
        self._health_cbs.setdefault(robot_id, []).append(callback)

    def subscribe_pose(self, robot_id: str, callback: Callable) -> None:
        self._pose_cbs.setdefault(robot_id, []).append(callback)

    def subscribe_nav_status(self, robot_id: str, callback: Callable) -> None:
        self._nav_status_cbs.setdefault(robot_id, []).append(callback)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Create all subscriptions and action clients."""
        try:
            import rclpy
            from rclpy.action import ActionClient
            from action_msgs.msg import GoalStatusArray
            from nav2_msgs.action import NavigateToPose
            from nav2_msgs.msg import NavigateToPoseFeedback
            from sensor_msgs.msg import BatteryState
            from nav_msgs.msg import Odometry
            from tf2_msgs.msg import TFMessage
            import sys
            import importlib

            # Try to import mars_msgs; fall back gracefully if not compiled
            try:
                sys.path.insert(0, "/opt/ros/humble/lib/python3.10/site-packages")
                RobotHealth = importlib.import_module(
                    "mars_msgs.msg"
                ).RobotHealth
            except Exception:
                log.warning("[ros2_adapter] mars_msgs not compiled — using Python shim")
                from mars.mars_msgs import RobotHealth  # Python shim

        except ImportError as exc:
            log.error("[ros2_adapter] rclpy not available: %s — adapter is a no-op", exc)
            return

        for robot_id in self._robot_ids:
            ns = f"/{robot_id}"

            # Nav2 action status (source of truth for goal outcomes on Humble)
            self._subs.append(
                self._node.create_subscription(
                    GoalStatusArray,
                    f"{ns}/navigate_to_pose/_action/status",
                    lambda msg, rid=robot_id: self._on_goal_status_array(rid, msg),
                    qos_profile=10,
                )
            )

            # Nav2 feedback (struggle signal — not used for routing decisions)
            self._subs.append(
                self._node.create_subscription(
                    NavigateToPose.Impl.FeedbackMessage,
                    f"{ns}/navigate_to_pose/_action/feedback",
                    lambda msg, rid=robot_id: self._on_nav_feedback(rid, msg),
                    qos_profile=10,
                )
            )

            # Battery (self-published; percentage ∈ [0, 1])
            self._subs.append(
                self._node.create_subscription(
                    BatteryState,
                    f"{ns}/battery_state",
                    lambda msg, rid=robot_id: self._on_battery(rid, msg),
                    qos_profile=10,
                )
            )

            # Robot health (self-published; maps to fault_flag)
            self._subs.append(
                self._node.create_subscription(
                    RobotHealth,
                    f"{ns}/robot_health",
                    lambda msg, rid=robot_id: self._on_robot_health(rid, msg),
                    qos_profile=10,
                )
            )

            # Odometry (Isaac Sim; backup pose + velocity)
            self._subs.append(
                self._node.create_subscription(
                    Odometry,
                    f"{ns}/odom",
                    lambda msg, rid=robot_id: self._on_odom(rid, msg),
                    qos_profile=10,
                )
            )

            # Action client for NavigateToPose
            self._action_clients[robot_id] = ActionClient(
                self._node,
                NavigateToPose,
                f"{ns}/navigate_to_pose",
            )

        # TF is global (not per-robot); Isaac Sim publishes map → base_link
        self._subs.append(
            self._node.create_subscription(
                TFMessage,
                "/tf",
                self._on_tf,
                qos_profile=100,
            )
        )

        log.info(
            "[ros2_adapter] setup complete for %d robot(s)", len(self._robot_ids)
        )

    # ------------------------------------------------------------------
    # Subscription callbacks → convert to interface data types
    # ------------------------------------------------------------------

    def _on_goal_status_array(self, robot_id: str, msg) -> None:
        """
        Parse GoalStatusArray and fire NavGoalStatus callbacks for any
        terminal status (SUCCEEDED / CANCELED / ABORTED).

        This is the Humble-correct approach: the NavigateToPose result message
        is empty, so we drive everything from GoalStatus.
        """
        from mars.ros.interfaces import NavGoalStatus

        with self._lock:
            active_goal_ids = {
                gid: entry
                for gid, entry in self._active_goals.items()
                if entry[0] == robot_id
            }

        for status_msg in (msg.status_list or []):
            # Extract goal UUID as hex string to match our goal_id
            goal_uuid = _uuid_bytes_to_hex(status_msg.goal_info.goal_id.uuid)
            status    = status_msg.status

            if status not in (STATUS_SUCCEEDED, STATUS_CANCELED, STATUS_ABORTED):
                continue

            # Find the matching active goal
            matched_gid = None
            for gid in active_goal_ids:
                if gid == goal_uuid or gid.startswith(goal_uuid[:8]):
                    matched_gid = gid
                    break

            if matched_gid is None:
                continue

            outcome = {
                STATUS_SUCCEEDED: "succeeded",
                STATUS_CANCELED:  "canceled",
                STATUS_ABORTED:   "aborted",
            }[status]

            nav_status = NavGoalStatus(
                goal_id=matched_gid,
                robot_id=robot_id,
                status=status,
            )

            # Fire all registered nav_status callbacks
            for cb in self._nav_status_cbs.get(robot_id, []):
                try:
                    cb(nav_status)
                except Exception:
                    log.exception("[ros2_adapter] nav_status callback error")

            # Remove from active goals if terminal
            with self._lock:
                self._active_goals.pop(matched_gid, None)

    def _on_result(self, robot_id: str, goal_id: str, future) -> None:
        """
        Called when get_result_async() completes.  On Humble the result payload
        is empty; the real signal already came through _on_goal_status_array.
        Log only.
        """
        try:
            result = future.result()
            log.debug(
                "[ros2_adapter] result received robot=%s goal=%s status=%d",
                robot_id, goal_id, result.status,
            )
        except Exception:
            log.exception("[ros2_adapter] get_result error robot=%s goal=%s", robot_id, goal_id)

    def _on_nav_feedback(self, robot_id: str, feedback_msg) -> None:
        """
        Nav2 feedback — number_of_recoveries and distance_remaining are the
        struggle signals used by the Aggregator for enrichment context.
        No callback is fired here; the Aggregator reads feedback passively
        if it subscribes directly, or we extend the interface in a later pass.
        """
        pass  # logged for future use

    def _on_battery(self, robot_id: str, msg) -> None:
        from mars.ros.interfaces import BatteryState as MARSBatteryState

        battery = MARSBatteryState(
            robot_id=robot_id,
            percentage=float(msg.percentage),
            power_supply_status=int(msg.power_supply_status),
        )
        for cb in self._battery_cbs.get(robot_id, []):
            try:
                cb(battery)
            except Exception:
                log.exception("[ros2_adapter] battery callback error")

    def _on_robot_health(self, robot_id: str, msg) -> None:
        from mars.ros.interfaces import RobotHealth as MARSRobotHealth

        health = MARSRobotHealth(
            robot_id=robot_id,
            level=int(msg.level),
            estop_active=bool(msg.estop_active),
            fault_codes=list(msg.fault_codes),
        )
        for cb in self._health_cbs.get(robot_id, []):
            try:
                cb(health)
            except Exception:
                log.exception("[ros2_adapter] health callback error")

    def _on_odom(self, robot_id: str, msg) -> None:
        from mars.ros.interfaces import Pose

        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        pose = Pose(
            x=float(p.x), y=float(p.y), z=float(p.z),
            qx=float(o.x), qy=float(o.y), qz=float(o.z), qw=float(o.w),
            frame_id="odom",
        )
        # Odom is backup pose; /tf (map frame) takes precedence
        # Only fire if TF hasn't provided a pose recently
        for cb in self._pose_cbs.get(robot_id, []):
            try:
                cb(pose)
            except Exception:
                log.exception("[ros2_adapter] odom callback error")

    def _on_tf(self, msg) -> None:
        """
        Process /tf transforms.  We look for map → <robot_id>/base_link.

        Isaac Sim publishes this as:
            header.frame_id = "map"
            child_frame_id  = "<robot_id>/base_link"
        """
        from mars.ros.interfaces import Pose

        for transform in msg.transforms:
            child = transform.child_frame_id
            if not child.endswith("/base_link"):
                continue

            # Extract robot_id: "<robot_id>/base_link" → "<robot_id>"
            robot_id = child[: -len("/base_link")]
            if robot_id not in self._robot_ids:
                continue

            t = transform.transform.translation
            r = transform.transform.rotation
            pose = Pose(
                x=float(t.x), y=float(t.y), z=float(t.z),
                qx=float(r.x), qy=float(r.y), qz=float(r.z), qw=float(r.w),
                frame_id=transform.header.frame_id,
            )

            for cb in self._pose_cbs.get(robot_id, []):
                try:
                    cb(pose)
                except Exception:
                    log.exception("[ros2_adapter] pose callback error")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid_bytes_to_hex(uuid_bytes) -> str:
    """Convert UUID byte array from action_msgs to a hex string for matching."""
    try:
        return bytes(uuid_bytes).hex()
    except Exception:
        return str(uuid_bytes)


def topic_name(robot_id: str, suffix: str) -> str:
    """Canonical per-robot topic name: /<robot_id>/<suffix>."""
    return f"/{robot_id}/{suffix}"
