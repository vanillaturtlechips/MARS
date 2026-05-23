"""클라우드(RunPod)에서 실행 — Isaac Sim 속성 존재 여부 확인.

실행:
  python tests/check_isaac_attrs.py --headless
"""

import sys
from pathlib import Path
from isaaclab.app import AppLauncher

import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
sys.path.insert(0, str(Path(__file__).parents[1]))

from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationStudentEnvCfg, WarehouseManipulationEnv
)

cfg = WarehouseManipulationStudentEnvCfg()
cfg.scene.num_envs = 4  # 최소 env 수
env = WarehouseManipulationEnv(cfg)

# 한 스텝 실행
actions = torch.zeros(4, 9, device=env.device)
env.step(actions)

robot = env.robot
data  = robot.data

print("\n=== Isaac Lab ArticulationData 속성 검사 ===")
checks = [
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_vel_w",
    "body_jacobians",
    "body_jacobian_w",
    "body_pos_w",
    "body_quat_w",
    "joint_pos",
    "joint_vel",
]
for attr in checks:
    val = getattr(data, attr, None)
    if val is not None:
        print(f"  ✅ {attr}: shape={tuple(val.shape)}")
    else:
        print(f"  ❌ {attr}: 없음")

# body_lin_vel_w 실제 값 확인
ee_idx = env._ee_body_idx
if hasattr(data, "body_lin_vel_w"):
    vel = data.body_lin_vel_w[:, ee_idx, :]
    print(f"\n[body_lin_vel_w] EE 속도 (env 0): {vel[0].tolist()}")
    print(f"[body_lin_vel_w] 속도 크기 (env 0): {vel[0].norm().item():.4f} m/s")
    print("→ 값이 0이 아니면 속도 데이터 정상")

env.close()
simulation_app.close()
print("\n=== 검사 완료 ===")
