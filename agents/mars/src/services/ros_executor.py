"""
ROS Executor — stub (§6a)

Translates a scheduled mission (robot_id + destination) into a NavigateToPose
goal dispatch.  Records goal_id → mission_id in the dispatch ledger.

In the real system this is a thin ROS2 node.  Here it's a plain class that
delegates to the NavigationInterface (mock or Isaac Sim adapter).
"""
from __future__ import annotations

import logging
import uuid

from mars.ros.interfaces import NavigationInterface, NavGoalStatus, Pose
from mars.blackboard.queries import record_dispatch

log = logging.getLogger(__name__)


class ROSExecutor:
    def __init__(self, nav: NavigationInterface, conn_factory):
        self._nav = nav
        self._conn = conn_factory
        # goal_id → mission_id for correlating outcomes
        self._pending: dict[str, str] = {}

    def dispatch(self, mission: dict) -> str:
        """
        Dispatch a NavigateToPose goal for mission.
        Returns goal_id.
        """
        goal_id = str(uuid.uuid4())
        robot_id = mission["robot_id"]
        mission_id = mission["mission_id"]

        dest_raw = mission.get("destination_pose") or {}
        destination = Pose(
            x=dest_raw.get("x", 0.0),
            y=dest_raw.get("y", 0.0),
            z=dest_raw.get("z", 0.0),
        )

        # Record in dispatch ledger before sending the goal
        conn = self._conn()
        record_dispatch(conn, goal_id, mission_id, robot_id)
        conn.commit()

        self._pending[goal_id] = mission_id

        self._nav.send_goal(
            robot_id=robot_id,
            goal_id=goal_id,
            destination=destination,
            on_status_change=lambda status: self._on_nav_status(status),
        )

        log.info("[ros_executor] dispatched goal=%s mission=%s robot=%s",
                 goal_id, mission_id, robot_id)
        return goal_id

    def cancel(self, robot_id: str, goal_id: str) -> None:
        self._nav.cancel_goal(robot_id, goal_id)

    def _on_nav_status(self, status: NavGoalStatus) -> None:
        log.info("[ros_executor] goal=%s status=%d", status.goal_id, status.status)
        # Aggregator picks this up via its own subscription
