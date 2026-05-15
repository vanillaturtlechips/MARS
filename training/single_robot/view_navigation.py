"""GUI 뷰어 — 훈련된 정책으로 창고 로봇 시각 확인."""

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

from envs.warehouse.warehouse_env import WarehouseNavEnvCfg

CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "warehouse_nav", "model_999.pt")
NUM_ENVS = 16   # GUI는 적게 해야 잘 보임
DEVICE = "cuda"


class Actor(nn.Module):
    def __init__(self, obs_dim=6, act_dim=3, hidden=(256, 128, 64)):
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


# -- 모델 로드 --
ckpt = torch.load(CKPT, map_location=DEVICE)
actor = Actor().to(DEVICE)
actor_state = {k.replace("actor.", ""): v
               for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
actor.net.load_state_dict(actor_state)
actor.eval()
sys.stderr.write(f"[view] Loaded checkpoint iter={ckpt['iter']}\n"); sys.stderr.flush()

# -- 환경 (env_spacing 좁혀서 보기 좋게) --
env_cfg = WarehouseNavEnvCfg()
env_cfg.scene.num_envs = NUM_ENVS
env_cfg.scene.env_spacing = 8.0

env = gym.make("Isaac-WarehouseNav-Direct-v0", cfg=env_cfg)
obs, _ = env.reset()

sys.stderr.write("[view] 창 열림 — 닫거나 Ctrl+C로 종료\n"); sys.stderr.flush()

step = 0
while simulation_app.is_running():
    with torch.no_grad():
        action = actor(obs["policy"])
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 300 == 0:
        sys.stderr.write(
            f"  step {step:5d} | reward {reward.mean().item():.2f} "
            f"| done {(terminated | truncated).sum().item()}/{NUM_ENVS}\n"
        )
        sys.stderr.flush()

env.close()
simulation_app.close()
