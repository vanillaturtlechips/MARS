"""Plot an S1~S6 scenario log into trajectory + distance PNGs (headless).

    python3 deploy/nav2/plot_scenario.py /workspace/s1_log.json

Writes <log>_traj.png (top-down robot paths, spawns o, goals x, obstacles ▪)
and <log>_dist.png (closest robot-robot distance over time + per-robot
distance-to-goal). Uses the Agg backend so it runs with no display.
"""
from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"R1": "tab:red", "R2": "tab:blue", "R3": "tab:green"}


def main():
    if len(sys.argv) < 2:
        print("usage: plot_scenario.py <log.json>")
        sys.exit(1)
    path = Path(sys.argv[1])
    d = json.loads(path.read_text())
    names = d["names"]
    tracks = d["tracks"]
    goals = {n: (x, y) for n, x, y in d["goals"]}
    spawns = {n: (x, y) for n, x, y in d["spawns"]}

    # ---- trajectory plot ----
    fig, ax = plt.subplots(figsize=(7, 7))
    for n in names:
        pts = tracks.get(n, [])
        if not pts:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        c = COLORS.get(n, "gray")
        ax.plot(xs, ys, "-", color=c, lw=1.5, label=n)
        sx, sy = spawns[n]
        gx, gy = goals[n]
        ax.plot(sx, sy, "o", color=c, ms=10, mfc="white", mew=2)   # spawn
        ax.plot(gx, gy, "x", color=c, ms=12, mew=3)                # goal
    for ox, oy in d.get("obstacles", []):
        ax.plot(ox, oy, "s", color="black", ms=12)                 # obstacle
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # desc is Korean; default matplotlib font can't render it (boxes). Keep the
    # title ASCII (scenario id) — full desc is in the JSON / console.
    ax.set_title(f"{d['scenario']}  (o=spawn  x=goal  s=obstacle)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend()
    traj = path.with_name(path.stem + "_traj.png")
    fig.savefig(traj, dpi=120, bbox_inches="tight")
    print(f"wrote {traj}")

    # ---- distance-over-time plot ----
    fig2, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    # closest robot-robot distance over time (resample onto union of timestamps)
    times = sorted({p[0] for n in names for p in tracks.get(n, [])})

    def pos_at(n, t):
        pts = tracks.get(n, [])
        best = None
        for p in pts:
            if p[0] <= t:
                best = p
            else:
                break
        return (best[1], best[2]) if best else None

    mind = []
    for t in times:
        dd = float("inf")
        for a, b in combinations(names, 2):
            pa, pb = pos_at(a, t), pos_at(b, t)
            if pa and pb:
                dd = min(dd, math.dist(pa, pb))
        mind.append(dd if dd != float("inf") else None)
    a1.plot(times, mind, "k-")
    a1.axhline(d.get("collision_dist", 0.6), color="red", ls="--", label="collision threshold")
    a1.set_ylabel("closest pair [m]"); a1.grid(True, alpha=0.3); a1.legend()
    a1.set_title(f"{d['scenario']} - robot-robot distance (min seen {d.get('min_dist_seen')} m)")

    # per-robot distance to its goal over time
    for n in names:
        pts = tracks.get(n, [])
        if not pts:
            continue
        gx, gy = goals[n]
        ts = [p[0] for p in pts]
        dg = [math.dist((p[1], p[2]), (gx, gy)) for p in pts]
        a2.plot(ts, dg, "-", color=COLORS.get(n, "gray"), label=n)
    a2.set_ylabel("dist to goal [m]"); a2.set_xlabel("t [s]")
    a2.grid(True, alpha=0.3); a2.legend()
    dist = path.with_name(path.stem + "_dist.png")
    fig2.savefig(dist, dpi=120, bbox_inches="tight")
    print(f"wrote {dist}")


if __name__ == "__main__":
    main()
