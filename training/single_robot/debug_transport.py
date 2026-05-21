"""transport 진단: grasped 상태에서 goal 방향으로 이동 시 dist_box_goal 감소 여부 확인.

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
from envs.warehouse.warehouse_manipulation_env import WarehouseManipulationEnv, WarehouseManipulationEnvCfg

def main():
    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs = 4
    # 즉시 grasp → transport만 테스트
    env_cfg.grasp_dist_threshold = 999.0
    env = WarehouseManipulationEnv(env_cfg)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]  # (N, 30)

    # obs layout (teacher, 30-dim):
    # [0:3]  box_pos - ee_pos
    # [3:7]  box_quat
    # [7]    box_mass
    # [8]    gripper_w
    # [9:12] goal_pos_w - ee_pos  ← transport direction
    # [12:21] jpos
    # [21:30] jvel

    print("\n[진단] grasped 상태에서 obs[9:12] (goal direction) 방향으로 이동")
    print(f"{'step':>4}  {'dist_box_goal':>14}  {'reward':>8}  {'grasp%':>7}")
    print("-" * 45)

    for step in range(100):
        # obs[9:12] = goal_pos_w - ee_pos → 이 방향으로 이동하면 box가 goal에 가까워져야 함
        direction = obs[:, 9:12]
        norm = direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
        action = torch.zeros(env.num_envs, 4, device=env.device)
        action[:, :3] = direction / norm   # unit vector toward goal
        action[:, 3] = -1.0               # gripper closed

        obs_dict, rew, terminated, timed_out, extras = env.step(action)
        obs = obs_dict["policy"]

        log = extras.get("log", {})
        dist = log.get("dist_box_goal", float("nan"))
        grasp = log.get("grasp_rate", 0.0)
        print(f"{step:>4}  {dist:>14.4f}  {rew.mean().item():>8.2f}  {grasp:>6.1f}%")

    print("\n[판정]")
    print("  dist 감소 → env/IK 정상, PPO 수렴 문제 → reward/hyperparameter 조정")
    print("  dist 고정 → grasp offset 또는 IK 버그 → env 코드 수정 필요")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
