"""workspace 한계 진단: 4개 goal 모두 hand-crafted policy로 도달 가능한지 확인.

실행:
  python training/single_robot/debug_transport.py --headless
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

def main():
    # 4개 env = 4개 goal 각각 할당
    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.grasp_dist_threshold = 999.0   # 즉시 grasp
    env_cfg.place_dist_threshold = 0.15
    env = WarehouseManipulationEnv(env_cfg)

    # goal을 4개 env에 각각 고정 할당
    obs_dict, _ = env.reset()
    goals = torch.tensor(PLACE_GOALS, device=env.device)
    env._goal_pos_w[:] = goals + env.scene.env_origins[:4]

    obs_dict, _ = env.reset()  # goal 재할당 후 obs 갱신
    env._goal_pos_w[:] = goals + env.scene.env_origins[:4]
    obs = obs_dict["policy"]

    print("\n[workspace 진단] 4개 goal 각각 도달 가능 여부")
    for i, g in enumerate(PLACE_GOALS):
        print(f"  env{i}: goal = {g}")
    print(f"\n{'step':>4}  " + "  ".join(f"env{i}_dist" for i in range(4)) + "  placed")
    print("-" * 65)

    placed = [False] * 4
    place_step = [None] * 4

    for step in range(500):
        # obs[9:12] = goal - box_pos_carried (올바른 transport 방향)
        direction = obs[:, 9:12]
        norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        action = torch.zeros(4, 4, device=env.device)
        action[:, :3] = direction / norm
        action[:, 3] = -1.0

        obs_dict, rew, terminated, timed_out, extras = env.step(action)
        obs = obs_dict["policy"]

        # 각 env dist 계산
        ee_pos, _ = env._get_ee_pose()
        box_carried = ee_pos + env._grasp_ee_offset
        dists = (box_carried - env._goal_pos_w).norm(dim=1)

        for i in range(4):
            if not placed[i] and dists[i].item() < 0.15:
                placed[i] = True
                place_step[i] = step

        if step % 50 == 0 or all(placed):
            dist_str = "  ".join(f"{dists[i].item():>9.4f}" for i in range(4))
            placed_str = "".join("O" if placed[i] else "." for i in range(4))
            print(f"{step:>4}  {dist_str}  {placed_str}")

        if all(placed):
            break

    print("\n[결과]")
    for i, g in enumerate(PLACE_GOALS):
        if placed[i]:
            print(f"  goal{i} {g}: 도달 가능 (step {place_step[i]})")
        else:
            final_dist = (box_carried[i] - env._goal_pos_w[i]).norm().item()
            print(f"  goal{i} {g}: 도달 불가 — 최종 dist={final_dist:.4f}m")

    print("\n[판정]")
    n_reachable = sum(placed)
    print(f"  도달 가능: {n_reachable}/4")
    if n_reachable == 4:
        print("  → workspace 문제 없음. entropy/reward 조정으로 해결 가능")
    else:
        print("  → workspace 한계. goal 위치 조정 필요")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
