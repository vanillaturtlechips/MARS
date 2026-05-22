"""환경 설계 전면 검증 (단일 실행).

실행:
  python training/single_robot/debug_verify.py --headless
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

MAX_STEPS  = 300
PLACE_DIST = 0.17
GRASP_DIST = 0.11

APPROACH_TARGETS = [
    (0.32, 0.00, 0.53),
    (0.35, 0.00, 0.53),
    (0.38, 0.00, 0.53),
    (0.40, 0.00, 0.53),
    (0.45, 0.00, 0.53),
    (0.50, 0.00, 0.53),
]

N_A = len(APPROACH_TARGETS)   # 6
N_G = len(PLACE_GOALS)        # 4
N_ENVS = max(N_A, N_G)        # 6 — 한 env로 두 테스트 공용

def get_ee(env):
    return env.robot.data.body_pos_w[:, env._ee_body_idx]

def main():
    cfg = WarehouseManipulationEnvCfg()
    cfg.scene.num_envs          = N_ENVS
    cfg.grasp_dist_threshold    = 999.0   # auto-grasp 차단 (approach) / 즉시 grasp (transport)
    cfg.place_dist_threshold    = PLACE_DIST
    env = WarehouseManipulationEnv(cfg)
    env.reset()

    origins = env.scene.env_origins   # (N_ENVS, 3)

    # ── 1. EE 실제 위치 ────────────────────────────────────────
    ee0    = get_ee(env)
    local0 = ee0[0] - origins[0]
    print("\n" + "="*62)
    print("[1] reach_pose EE 위치")
    print(f"    x={local0[0]:.4f}  y={local0[1]:.4f}  z={local0[2]:.4f}")

    # ── 2. 박스 위치별 접근 가능성 ────────────────────────────
    print("\n[2] 박스 위치별 EE 접근 가능성 (hand-crafted policy)")
    print(f"    {'박스':>22}  {'초기':>6}  {'최소':>6}  도달  스텝")
    print("    " + "-"*48)

    targets_w = torch.tensor(APPROACH_TARGETS, device=env.device) + origins[:N_A]
    env.reset()
    ee_now   = get_ee(env)
    init_d   = (ee_now[:N_A] - targets_w).norm(dim=1)
    min_d    = init_d.clone()
    reach_st = [None] * N_A

    for step in range(MAX_STEPS):
        ee_pos = get_ee(env)
        diff   = targets_w - ee_pos[:N_A]
        dist   = diff.norm(dim=1)
        norm   = dist.unsqueeze(1).clamp(min=1e-6)
        for i in range(N_A):
            if dist[i] < min_d[i]:
                min_d[i] = dist[i]
            if reach_st[i] is None and dist[i].item() < GRASP_DIST:
                reach_st[i] = step
        action = torch.zeros(N_ENVS, 4, device=env.device)
        action[:N_A, :3] = diff / norm
        action[:, 3] = -1.0
        env.step(action)

    for i, t in enumerate(APPROACH_TARGETS):
        rs = str(reach_st[i]) if reach_st[i] is not None else "—"
        ok = "O" if reach_st[i] is not None else "X"
        print(f"    ({t[0]:.2f},{t[1]:+.2f},{t[2]:.2f})  "
              f"{init_d[i].item():>6.3f}  {min_d[i].item():>6.3f}  {ok:>4}  {rs}")

    # ── 3. PLACE_GOALS transport 가능성 ───────────────────────
    print(f"\n[3] PLACE_GOALS transport 가능성")
    print(f"    현재 goals: {PLACE_GOALS}")
    print(f"    {'goal':>22}  {'최소':>6}  도달  스텝")
    print("    " + "-"*42)

    goals_w = torch.tensor(PLACE_GOALS, device=env.device) + origins[:N_G]
    env._goal_pos_w[:N_G] = goals_w
    obs_dict, _ = env.reset()
    env._goal_pos_w[:N_G] = goals_w
    obs = obs_dict["policy"]

    min_d2   = torch.full((N_G,), 9999.0, device=env.device)
    place_st = [None] * N_G

    for step in range(MAX_STEPS):
        direction = obs[:N_G, 9:12]
        norm      = direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        action    = torch.zeros(N_ENVS, 4, device=env.device)
        action[:N_G, :3] = direction / norm
        action[:, 3] = -1.0
        obs_dict, _, _, _, _ = env.step(action)
        obs = obs_dict["policy"]

        ee_pos   = get_ee(env)
        carried  = ee_pos[:N_G] + env._grasp_ee_offset[:N_G]
        dists    = (carried - env._goal_pos_w[:N_G]).norm(dim=1)
        for i in range(N_G):
            if dists[i] < min_d2[i]:
                min_d2[i] = dists[i]
            if place_st[i] is None and dists[i].item() < PLACE_DIST:
                place_st[i] = step
        if all(s is not None for s in place_st):
            break

    for i, g in enumerate(PLACE_GOALS):
        ps = str(place_st[i]) if place_st[i] is not None else "—"
        ok = "O" if place_st[i] is not None else "X"
        print(f"    ({g[0]:.2f},{g[1]:+.2f},{g[2]:.2f})  "
              f"{min_d2[i].item():>6.3f}  {ok:>4}  {ps}")

    env.close()

    # ── 판정 ───────────────────────────────────────────────────
    new_spawn_ok = all(reach_st[i] is not None for i in range(3))  # x=0.32~0.38
    goals_ok     = all(s is not None for s in place_st)
    print("\n" + "="*62)
    print("[판정]")
    print(f"  EE 위치: x={local0[0]:.4f}")
    print(f"  새 스폰 범위(x≤0.38) 접근: {'OK' if new_spawn_ok else 'FAIL'}")
    print(f"  새 goal 전체 도달:          {'OK' if goals_ok else 'FAIL'}")
    if new_spawn_ok and goals_ok:
        print("\n  → 훈련 시작 가능")
    else:
        print("\n  → 환경 재설계 필요 — 위 결과 공유해줘")
    print("="*62)

if __name__ == "__main__":
    main()
    simulation_app.close()
