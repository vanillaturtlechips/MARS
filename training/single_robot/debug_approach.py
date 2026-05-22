"""EE 도달 범위 및 환경 설계 전면 검증.

테스트:
  1. reach_pose에서 EE 실제 위치
  2. 박스 위치별 EE 접근 가능성 (grasp_dist=0.11m 기준)
  3. 새 PLACE_GOALS (y=±0.25) transport 가능성 — debug_transport 방식

실행:
  python training/single_robot/debug_approach.py --headless
"""
import argparse, sys
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv, WarehouseManipulationEnvCfg, PLACE_GOALS
)

GRASP_DIST  = 0.11
PLACE_DIST  = 0.17
MAX_STEPS   = 600

APPROACH_TARGETS = [
    (0.32, 0.00, 0.53),
    (0.35, 0.00, 0.53),
    (0.38, 0.00, 0.53),
    (0.40, 0.00, 0.53),
    (0.45, 0.00, 0.53),
    (0.50, 0.00, 0.53),
]


def get_ee(env):
    return env.robot.data.body_pos_w[:, env._ee_body_idx]


def main():
    # ── 1. reach_pose에서 EE 실제 위치 ──────────────────────────
    print("\n" + "="*65)
    print("[테스트 1] reach_pose에서 EE 실제 위치")
    print("="*65)
    env = WarehouseManipulationEnv(WarehouseManipulationEnvCfg())
    env.reset()
    ee     = get_ee(env)
    origin = env.scene.env_origins[0]
    local  = ee[0] - origin
    print(f"  EE 로컬 위치: x={local[0]:.4f}  y={local[1]:.4f}  z={local[2]:.4f}")
    ee_local = local.clone()
    env.close()

    # ── 2. 박스 위치별 EE 접근 가능성 ────────────────────────────
    print("\n" + "="*65)
    print(f"[테스트 2] 박스 위치별 EE 접근 (grasp_dist={GRASP_DIST}m)")
    print(f"  EE 시작: x={ee_local[0]:.4f}  y={ee_local[1]:.4f}  z={ee_local[2]:.4f}")
    print("="*65)
    print(f"  {'박스 위치':>25}  {'초기dist':>8}  {'최소dist':>8}  {'도달':>4}  {'스텝':>5}")
    print("  " + "-"*58)

    n = len(APPROACH_TARGETS)
    cfg = WarehouseManipulationEnvCfg()
    cfg.scene.num_envs = n
    cfg.grasp_dist_threshold = 999.0   # 자동 grasp 차단
    env = WarehouseManipulationEnv(cfg)
    env.reset()
    origin = env.scene.env_origins

    targets_w = torch.tensor(APPROACH_TARGETS, device=env.device) + origin[:n]
    env.reset()

    ee0      = get_ee(env)
    init_dist = (ee0 - targets_w).norm(dim=1)
    min_dist  = init_dist.clone()
    reach_step = [None] * n

    for step in range(MAX_STEPS):
        ee_pos = get_ee(env)
        diff   = targets_w - ee_pos
        dist   = diff.norm(dim=1)
        norm   = dist.unsqueeze(1).clamp(min=1e-6)

        for i in range(n):
            if dist[i] < min_dist[i]:
                min_dist[i] = dist[i]
            if reach_step[i] is None and dist[i].item() < GRASP_DIST:
                reach_step[i] = step

        action = torch.zeros(n, 4, device=env.device)
        action[:, :3] = diff / norm
        action[:, 3]  = -1.0
        env.step(action)

    for i, tgt in enumerate(APPROACH_TARGETS):
        rs   = str(reach_step[i]) if reach_step[i] is not None else "—"
        ok   = "O" if reach_step[i] is not None else "X"
        print(f"  ({tgt[0]:.2f}, {tgt[1]:+.2f}, {tgt[2]:.2f})  "
              f"{init_dist[i].item():>8.4f}  "
              f"{min_dist[i].item():>8.4f}  "
              f"{ok:>4}  {rs:>5}")
    env.close()

    # ── 3. PLACE_GOALS transport 가능성 ────────────────────────
    print("\n" + "="*65)
    print(f"[테스트 3] PLACE_GOALS transport 가능성 (place_dist={PLACE_DIST}m)")
    print(f"  현재 PLACE_GOALS:")
    for i, g in enumerate(PLACE_GOALS):
        print(f"    goal{i}: {g}")
    print("="*65)
    print(f"  {'goal 위치':>28}  {'최소dist':>8}  {'도달':>4}  {'스텝':>5}")
    print("  " + "-"*52)

    ng = len(PLACE_GOALS)
    cfg2 = WarehouseManipulationEnvCfg()
    cfg2.scene.num_envs = ng
    cfg2.grasp_dist_threshold = 999.0   # 첫 스텝에 전부 auto-grasp
    cfg2.place_dist_threshold = PLACE_DIST
    env2 = WarehouseManipulationEnv(cfg2)

    obs_dict, _ = env2.reset()
    goals_w = torch.tensor(PLACE_GOALS, device=env2.device) + env2.scene.env_origins[:ng]
    env2._goal_pos_w[:] = goals_w
    obs_dict, _ = env2.reset()
    env2._goal_pos_w[:] = goals_w
    obs = obs_dict["policy"]

    min_dist2  = torch.full((ng,), 9999.0, device=env2.device)
    place_step = [None] * ng

    for step in range(MAX_STEPS):
        # obs[9:12] = goal - box_pos_carried (grasped 이후 올바른 direction)
        direction = obs[:, 9:12]
        norm      = direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        action    = torch.zeros(ng, 4, device=env2.device)
        action[:, :3] = direction / norm
        action[:, 3]  = -1.0

        obs_dict, _, _, _, _ = env2.step(action)
        obs = obs_dict["policy"]

        ee_pos     = get_ee(env2)
        box_carried = ee_pos + env2._grasp_ee_offset
        dists      = (box_carried - env2._goal_pos_w).norm(dim=1)

        for i in range(ng):
            if dists[i] < min_dist2[i]:
                min_dist2[i] = dists[i]
            if place_step[i] is None and dists[i].item() < PLACE_DIST:
                place_step[i] = step

        if all(s is not None for s in place_step):
            break

    for i, g in enumerate(PLACE_GOALS):
        ps  = str(place_step[i]) if place_step[i] is not None else "—"
        ok  = "O" if place_step[i] is not None else "X"
        print(f"  ({g[0]:.2f}, {g[1]:+.2f}, {g[2]:.2f})  "
              f"{min_dist2[i].item():>8.4f}  "
              f"{ok:>4}  {ps:>5}")
    env2.close()

    # ── 판정 ────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("[판정]")
    print(f"  EE 실제 위치: x={ee_local[0]:.4f}")
    print(f"\n  접근 가능 박스 위치 ({GRASP_DIST}m 기준):")
    for i, tgt in enumerate(APPROACH_TARGETS):
        tag = "가능" if reach_step[i] is not None else "불가"
        print(f"    x={tgt[0]:.2f}: {tag}  (최소dist={min_dist[i].item():.4f})")
    print(f"\n  goal 도달 가능:")
    for i, g in enumerate(PLACE_GOALS):
        tag = "가능" if place_step[i] is not None else "불가"
        print(f"    ({g[0]:.2f}, {g[1]:+.2f}): {tag}  (최소dist={min_dist2[i].item():.4f})")

    all_approach = all(reach_step[i] is not None for i in range(3))  # 새 스폰 범위 3개
    all_goals    = all(s is not None for s in place_step)
    print(f"\n  새 스폰 범위(x≤0.38) 접근: {'전부 가능' if all_approach else '일부 불가'}")
    print(f"  새 goal 전체 도달:          {'전부 가능' if all_goals else '일부 불가'}")
    if all_approach and all_goals:
        print("\n  → 훈련 시작 가능")
    else:
        print("\n  → 환경 재설계 필요")
    print("="*65)


if __name__ == "__main__":
    main()
    simulation_app.close()
