"""GUI 뷰어 — Phase 1.5 장애물 창고 네비게이션 시각 확인."""

import sys
import os

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs.warehouse  # noqa: F401

from envs.warehouse.warehouse_obstacle_env import WarehouseObstacleNavEnvCfg

CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "logs",
                    "warehouse_obstacle_nav", "model_100.pt")
NUM_ENVS = 4    # 선반이 있으므로 적게 해야 잘 보임
DEVICE = "cuda"


class Actor(nn.Module):
    def __init__(self, obs_dim=7, act_dim=3, hidden=(256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


ckpt = torch.load(CKPT, map_location=DEVICE)
actor = Actor().to(DEVICE)
actor_state = {k.replace("actor.", ""): v
               for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
actor.net.load_state_dict(actor_state)
actor.eval()
sys.stderr.write(f"[view] Loaded model_100.pt iter={ckpt['iter']}\n"); sys.stderr.flush()

env_cfg = WarehouseObstacleNavEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.scene.env_spacing = 14.0   # 선반 포함 env는 간격 넓혀야 안 겹침

env = gym.make("Isaac-WarehouseObstacleNav-Direct-v0", cfg=env_cfg)
obs, _ = env.reset()

sys.stderr.write("[view] 창 열림 — 닫거나 Ctrl+C로 종료\n")
sys.stderr.write("[view] 파란 박스=로봇, 갈색 박스=선반\n")
sys.stderr.flush()

step = 0
ep_count = 0
while simulation_app.is_running():
    with torch.no_grad():
        action = actor(obs["policy"])
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated | truncated
    ep_count += done.sum().item()
    step += 1

    if step % 150 == 0:
        sys.stderr.write(
            f"  step {step:5d} | reward {reward.mean().item():.2f} "
            f"| episodes {ep_count} "
            f"| done {done.sum().item()}/{NUM_ENVS}\n"
        )
        sys.stderr.flush()

env.close()
simulation_app.close()
