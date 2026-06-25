"""Headless logger for S1~S6 runs — record trajectories + inter-robot distance.

Subscribes to each robot's /{name}/odom, logs (t, x, y) and the minimum pairwise
robot-robot distance. Prints a live one-line dashboard (positions + closest pair)
and, on exit, dumps a JSON the plotter turns into trajectory/distance PNGs.

Run alongside scenario_goals.py:
    source deploy/isaac/env_ros2.sh
    python3 deploy/nav2/scenario_logger.py S1 --out /workspace/s1_log.json

Then plot (anywhere with matplotlib):
    python3 deploy/nav2/plot_scenario.py /workspace/s1_log.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # deploy/
import scenarios as SC  # noqa: E402

COLLISION_DIST = 0.6   # robot-robot distance below this = near-collision flag


class ScenarioLogger(Node):
    def __init__(self, scenario: str, out: str):
        super().__init__("scenario_logger")
        self.scenario = scenario
        self.out = out
        self.t0 = time.time()
        self.names = [n for n, _, _ in SC.robots_for(scenario)]
        self.tracks: dict[str, list] = {n: [] for n in self.names}
        self.pos: dict[str, tuple] = {}
        self.min_dist_seen = float("inf")
        # /Rx/odom is zeroed at each robot's spawn (per-robot frame) -> useless for
        # inter-robot distance. Read map->Rx/base_link from TF instead (common frame).
        self.tfbuf = Buffer()
        self.tfl = TransformListener(self.tfbuf, self)
        self.create_timer(0.2, self._sample)
        self.create_timer(1.0, self._dash)
        self.get_logger().info(f"[{scenario}] logging {self.names} (TF map->base_link) -> {out}")

    def _sample(self):
        from rclpy.time import Time
        t = time.time() - self.t0
        for name in self.names:
            try:
                tr = self.tfbuf.lookup_transform("map", f"{name}/base_link", Time())
            except Exception:
                continue
            x = tr.transform.translation.x
            y = tr.transform.translation.y
            self.pos[name] = (x, y)
            self.tracks[name].append([round(t, 2), round(x, 3), round(y, 3)])

    def _min_pair(self) -> float:
        d = float("inf")
        for a, b in combinations(self.names, 2):
            if a in self.pos and b in self.pos:
                d = min(d, math.dist(self.pos[a], self.pos[b]))
        return d

    def _dash(self):
        if not self.pos:
            return
        d = self._min_pair()
        self.min_dist_seen = min(self.min_dist_seen, d)
        ps = "  ".join(f"{n}({self.pos[n][0]:.1f},{self.pos[n][1]:.1f})"
                       for n in self.names if n in self.pos)
        flag = "  <!-- NEAR-COLLISION" if d < COLLISION_DIST else ""
        self.get_logger().info(f"{ps}  | closest pair {d:.2f} m{flag}")

    def dump(self):
        data = {
            "scenario": self.scenario,
            "desc": SC.SCENARIOS[self.scenario]["desc"],
            "names": self.names,
            "spawns": [[n, x, y] for n, x, y in SC.robots_for(self.scenario)],
            "goals": [[n, x, y] for n, x, y in SC.goals_for(self.scenario)],
            "obstacles": SC.obstacles_for(self.scenario),
            "collision_dist": COLLISION_DIST,
            "min_dist_seen": round(self.min_dist_seen, 3),
            "tracks": self.tracks,
        }
        Path(self.out).write_text(json.dumps(data, indent=2))
        self.get_logger().info(
            f"saved {self.out}  (min robot-robot dist seen: {self.min_dist_seen:.2f} m)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or f"/workspace/{a.scenario.lower()}_log.json"
    rclpy.init()
    node = ScenarioLogger(a.scenario, out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.dump()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
