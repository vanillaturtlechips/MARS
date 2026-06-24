#!/usr/bin/env python3
"""Drive ONE robot along (x,y) waypoints in its odom frame via /<ns>/cmd_vel,
closing the loop on /<ns>/odom. NO Nav2.

Why: 3 full Nav2 stacks overload this environment's constrained FastDDS discovery
(Isaac-interop UDP-only, container can't raise net.core.*mem_max), so the later
stacks' action/lifecycle endpoint matching is flaky and 1-2 of the 3 robots' nav
dies at random (1 stack = rock solid). So only the "hero" robot (R1) keeps Nav2 —
its real failure triggers the agent's avoid_zone + keepout — and the follower
robots execute the (already scripted) reroute by direct cmd_vel here. The agent
decision and R1's keepout stay real; this just removes the multi-stack DDS load.

Waypoints are in the robot's odom frame (= world minus its spawn, map-aligned,
since the spawn map->odom static tf has no rotation):
    python3 follow_waypoints.py R2 0,-1 -5,-1 -6,2
Holds a zero twist when done / on SIGINT|SIGTERM (the diff drive latches the last
velocity, so a robot left un-zeroed runs off the map)."""
import sys, math, signal
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

NS = sys.argv[1]
WPS = [tuple(float(v) for v in a.split(",")) for a in sys.argv[2:]]

# Single-robot warehouse run uses un-namespaced /cmd_vel and /odom. Pass "-"
# (or "" / "/") as the namespace for that case.
_BARE = NS in ("", "-", "/")
_CMD = "/cmd_vel" if _BARE else f"/{NS}/cmd_vel"
_ODOM = "/odom" if _BARE else f"/{NS}/odom"
_NODE = "follower" if _BARE else f"{NS}_follower"


class Follower(Node):
    def __init__(self):
        super().__init__(_NODE)
        self.pub = self.create_publisher(Twist, _CMD, 10)
        self.create_subscription(Odometry, _ODOM, self._odom, 10)
        self.x = self.y = self.yaw = 0.0
        self.have = False
        self.i = 0
        self.VLIN, self.VANG = 0.4, 0.8
        self.K_ANG = 1.5          # proportional heading gain
        self.POS_TOL = 0.3
        self.create_timer(0.1, self._tick)

    def _odom(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                              1 - 2 * (q.y * q.y + q.z * q.z))
        self.have = True

    def _tick(self):
        t = Twist()
        if not self.have or self.i >= len(WPS):
            self.pub.publish(t)            # zero -> hold position
            return
        gx, gy = WPS[self.i]
        dx, dy = gx - self.x, gy - self.y
        if math.hypot(dx, dy) < self.POS_TOL:
            self.i += 1
            self.pub.publish(t)
            if self.i >= len(WPS):
                self.get_logger().info(f"{NS} reached final waypoint")
            return
        err = math.atan2(math.sin(math.atan2(dy, dx) - self.yaw),
                         math.cos(math.atan2(dy, dx) - self.yaw))
        # smooth proportional control: steer toward the goal and drive forward only as
        # much as we already face it (cos) -> gentle curves, no bang-bang zig-zag/dance.
        t.angular.z = max(-self.VANG, min(self.VANG, self.K_ANG * err))
        t.linear.x = self.VLIN * max(0.0, math.cos(err))
        self.pub.publish(t)


def main():
    rclpy.init()
    n = Follower()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            n.pub.publish(Twist())         # stop before exit
        except Exception:
            pass
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
