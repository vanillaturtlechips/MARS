"""EE 도달 범위 및 환경 설계 전면 검증.

테스트:
  1. reach_pose에서 EE 실제 위치
  2. 박스 위치별 EE 접근 가능성 (grasp_dist=0.10m 기준)
  3. 새 goal 위치 (y=±0.25) transport 가능성

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
    WarehouseManipulationEnv, WarehouseManipulationEnvCfg
)

# ── 테스트 파라미터 ──────────────────────────────────────────────
GRASP_DIST   = 0.10   # 현재 설정값
PLACE_DIST   = 0.17
MAX_STEPS    = 600

# 테스트 1: 박스 위치별 EE 접근 가능 여부
APPROACH_TARGETS = [
    (0.32, 0.00, 0.53),  # 새 스폰 범위 (가까운 쪽)
    (0.35, 0.00, 0.53),  # 새 스폰 범위 (중간)
    (0.38, 0.00, 0.53),  # 새 스폰 범위 (먼 쪽)
    (0.40, 0.00, 0.53),  # 경계
    (0.45, 0.00, 0.53),  # 기존 스폰 범위 (가까운 쪽)
    (0.50, 0.00, 0.53),  # 기존 스폰 범위 (먼 쪽)
]

# 테스트 2: goal 위치별 transport 가능 여부 (grasp된 상태 가정)
TRANSPORT_GOALS = [
    (0.36, -0.12, 0.53),  # 기존 goal (검증됨)
    (0.36,  0.12, 0.53),  # 기존 goal (검증됨)
    (0.33, -0.25, 0.53),  # 새 goal 후보
    (0.33,  0.25, 0.53),  # 새 goal 후보
    (0.36, -0.25, 0.53),  # 새 goal 후보
    (0.36,  0.25, 0.53),  # 새 goal 후보
]


def make_env(num_envs: int):
    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.grasp_dist_threshold = 999.0   # grasp 자동 발동 차단
    env_cfg.place_dist_threshold = PLACE_DIST
    return WarehouseManipulationEnv(env_cfg)


def get_ee_pos(env):
    return env.robot.data.body_pos_w[:, env._ee_body_idx]


def main():
    # ── 1. reach_pose에서 EE 실제 위치 확인 ──────────────────────
    print("\n" + "="*65)
    print("[테스트 1] reach_pose에서 EE 실제 위치")
    print("="*65)

    env = make_env(1)
    env.reset()
    ee = get_ee_pos(env)
    origin = env.scene.env_origins[0]
    ee_local = ee[0] - origin
    print(f"  EE 절대 위치 (world): {ee[0].tolist()}")
    print(f"  EE 로컬 위치 (env 기준): x={ee_local[0]:.4f}, y={ee_local[1]:.4f}, z={ee_local[2]:.4f}")
    ee_reach = ee_local.clone()
    env.close()

    # ── 2. 박스 위치별 EE 접근 가능성 ────────────────────────────
    print("\n" + "="*65)
    print(f"[테스트 2] 박스 위치별 EE 접근 가능성 (grasp_dist={GRASP_DIST}m)")
    print(f"  EE 시작: x={ee_reach[0]:.4f}, y={ee_reach[1]:.4f}, z={ee_reach[2]:.4f}")
    print("="*65)
    print(f"{'박스 위치':>28}  {'최소dist':>9}  {'도달':>4}  {'스텝':>5}")
    print("-"*55)

    n_approach = len(APPROACH_TARGETS)
    env = make_env(n_approach)
    env.reset()
    origin = env.scene.env_origins  # (N, 3)

    targets_w = torch.tensor(APPROACH_TARGETS, device=env.device) + origin[:n_approach]
    obs_dict, _ = env.reset()

    min_dist = torch.full((n_approach,), 9999.0, device=env.device)
    reach_step = [None] * n_approach

    for step in range(MAX_STEPS):
        ee_pos = get_ee_pos(env)
        direction = targets_w - ee_pos
        dist = direction.norm(dim=1)
        norm = dist.unsqueeze(1).clamp(min=1e-6)

        for i in range(n_approach):
            if dist[i] < min_dist[i]:
                min_dist[i] = dist[i]
            if reach_step[i] is None and dist[i].item() < GRASP_DIST:
                reach_step[i] = step

        action = torch.zeros(n_approach, 4, device=env.device)
        action[:, :3] = direction / norm
        action[:, 3] = -1.0
        env.step(action)

    for i, tgt in enumerate(APPROACH_TARGETS):
        reached = reach_step[i] is not None
        step_str = str(reach_step[i]) if reached else "—"
        print(f"  ({tgt[0]:.2f}, {tgt[1]:+.2f}, {tgt[2]:.2f})  "
              f"{min_dist[i].item():>9.4f}  "
              f"{'O' if reached else 'X':>4}  {step_str:>5}")
    env.close()

    # ── 3. goal 위치별 transport 가능성 ──────────────────────────
    print("\n" + "="*65)
    print(f"[테스트 3] goal 위치별 transport 가능성 (place_dist={PLACE_DIST}m)")
    print("  (박스를 EE에 붙인 채 goal로 이동 — 기존 debug_transport 방식)")
    print("="*65)
    print(f"{'goal 위치':>28}  {'최소dist':>9}  {'도달':>4}  {'스텝':>5}")
    print("-"*55)

    n_goals = len(TRANSPORT_GOALS)
    env_cfg2 = WarehouseManipulationEnvCfg()
    env_cfg2.scene.num_envs = n_goals
    env_cfg2.grasp_dist_threshold = 999.0
    env_cfg2.place_dist_threshold = PLACE_DIST
    env2 = WarehouseManipulationEnv(env_cfg2)

    obs_dict, _ = env2.reset()
    origin2 = env2.scene.env_origins[:n_goals]
    goals_w = torch.tensor(TRANSPORT_GOALS, device=env2.device) + origin2

    env2._goal_pos_w[:] = goals_w
    obs_dict, _ = env2.reset()
    env2._goal_pos_w[:] = goals_w

    # grasp 강제 설정
    env2._grasped[:] = True
    ee0 = get_ee_pos(env2)
    env2._grasp_ee_offset[:] = torch.tensor([0.0, 0.0, 0.04], device=env2.device)

    obs = obs_dict["policy"]
    min_dist2 = torch.full((n_goals,), 9999.0, device=env2.device)
    place_step = [None] * n_goals

    for step in range(MAX_STEPS):
        direction = obs[:, 9:12]
        norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        action = torch.zeros(n_goals, 4, device=env2.device)
        action[:, :3] = direction / norm
        action[:, 3] = -1.0

        obs_dict, _, _, _, _ = env2.step(action)
        obs = obs_dict["policy"]

        ee_pos = get_ee_pos(env2)
        box_carried = ee_pos + env2._grasp_ee_offset
        dists = (box_carried - env2._goal_pos_w).norm(dim=1)

        for i in range(n_goals):
            if dists[i] < min_dist2[i]:
                min_dist2[i] = dists[i]
            if place_step[i] is None and dists[i].item() < PLACE_DIST:
                place_step[i] = step

        if all(s is not None for s in place_step):
            break

    for i, g in enumerate(TRANSPORT_GOALS):
        reached = place_step[i] is not None
        step_str = str(place_step[i]) if reached else "—"
        tag = "(기존)" if i < 2 else "(신규)"
        print(f"  ({g[0]:.2f}, {g[1]:+.2f}, {g[2]:.2f}) {tag}  "
              f"{min_dist2[i].item():>9.4f}  "
              f"{'O' if reached else 'X':>4}  {step_str:>5}")
    env2.close()

    # ── 4. 최종 판정 ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("[최종 판정]")
    print(f"  EE 시작 위치: x={ee_reach[0]:.4f}")

    approach_ok = [reach_step[i] is not None for i in range(n_approach)]
    transport_ok = [place_step[i] is not None for i in range(n_goals)]

    print(f"\n  접근 가능 박스 위치:")
    for i, tgt in enumerate(APPROACH_TARGETS):
        print(f"    x={tgt[0]:.2f}: {'가능' if approach_ok[i] else '불가'}")

    print(f"\n  goal 도달 가능 여부:")
    for i, g in enumerate(TRANSPORT_GOALS):
        tag = "(기존)" if i < 2 else "(신규)"
        print(f"    ({g[0]:.2f}, {g[1]:+.2f}) {tag}: {'가능' if transport_ok[i] else '불가'}")

    print("="*65)


if __name__ == "__main__":
    main()
    simulation_app.close()
