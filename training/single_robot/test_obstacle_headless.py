"""Phase 1.5 헤드리스 검증 — checkpoint 로드 후 성공률/평균 보상 측정."""

import sys
import os
import argparse

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import envs.warehouse  # noqa: F401

from envs.warehouse.warehouse_obstacle_env import WarehouseObstacleNavEnvCfg

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "..", "logs",
                         "warehouse_obstacle_nav", "model_100.pt"))
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_episodes", type=int, default=20)
args, _ = parser.parse_known_args()

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


ckpt = torch.load(args.ckpt, map_location=DEVICE)
actor = Actor().to(DEVICE)
actor_state = {k.replace("actor.", ""): v
               for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
actor.net.load_state_dict(actor_state)
actor.eval()
sys.stderr.write(f"[test] Loaded {os.path.basename(args.ckpt)} iter={ckpt['iter']}\n")
sys.stderr.flush()

env_cfg = WarehouseObstacleNavEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = gym.make("Isaac-WarehouseObstacleNav-Direct-v0", cfg=env_cfg)

obs, _ = env.reset()
sys.stderr.write(f"[test] obs shape: {obs['policy'].shape}\n"); sys.stderr.flush()

ep_rewards = []
ep_lengths = []
goal_reached = []

current_reward = torch.zeros(args.num_envs, device=DEVICE)
current_length = torch.zeros(args.num_envs, device=DEVICE)
episodes_done = torch.zeros(args.num_envs, device=DEVICE)

while episodes_done.min() < args.num_episodes:
    with torch.no_grad():
        action = actor(obs["policy"])
    obs, reward, terminated, truncated, info = env.step(action)

    current_reward += reward
    current_length += 1
    done = terminated | truncated

    if done.any():
        idx = done.nonzero(as_tuple=True)[0]
        for i in idx:
            ep_rewards.append(current_reward[i].item())
            ep_lengths.append(current_length[i].item())
            max_len = env.unwrapped.max_episode_length
            goal_reached.append(current_length[i].item() < max_len - 1)
            current_reward[i] = 0.0
            current_length[i] = 0.0
            episodes_done[i] += 1

import numpy as np
ep_rewards = np.array(ep_rewards)
ep_lengths = np.array(ep_lengths)
success_rate = np.mean(goal_reached) * 100

sys.stderr.write("\n" + "=" * 60 + "\n")
sys.stderr.write(f"  체크포인트:         {os.path.basename(args.ckpt)} (iter={ckpt['iter']})\n")
sys.stderr.write("=" * 60 + "\n")
sys.stderr.write(f"  총 에피소드:        {len(ep_rewards)}\n")
sys.stderr.write(f"  골 도달률:          {success_rate:.1f}%\n")
sys.stderr.write(f"  평균 에피소드 길이: {ep_lengths.mean():.1f} 스텝\n")
sys.stderr.write(f"  평균 보상:          {ep_rewards.mean():.2f}\n")
sys.stderr.write(f"  보상 std:           {ep_rewards.std():.2f}\n")
sys.stderr.write("=" * 60 + "\n")
sys.stderr.flush()

env.close()
simulation_app.close()
