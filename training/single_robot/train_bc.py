"""BC 훈련: (student_obs_29D, teacher_action_9D) → Student actor 초기화

Isaac Lab 불필요. CPU/GPU 모두 가능.

Usage:
  python training/single_robot/train_bc.py \
    --data bc_data.pt --output bc_actor.pt \
    --epochs 100 --batch_size 4096 --lr 3e-4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).parents[2]))

parser = argparse.ArgumentParser()
parser.add_argument("--data",       type=str,   required=True)
parser.add_argument("--output",     type=str,   default="bc_actor.pt")
parser.add_argument("--epochs",     type=int,   default=100)
parser.add_argument("--batch_size", type=int,   default=4096)
parser.add_argument("--lr",         type=float, default=3e-4)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

data = torch.load(args.data, map_location=device)
obs  = data["obs"].float()
acts = data["actions"].float()
print(f"[BC] 데이터: obs={obs.shape}, actions={acts.shape}")

dataset = TensorDataset(obs, acts)
val_size  = max(1, int(0.05 * len(dataset)))
trn_size  = len(dataset) - val_size
trn_ds, val_ds = random_split(dataset, [trn_size, val_size])
trn_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

obs_dim = obs.shape[1]   # 29
act_dim = acts.shape[1]  # 9

actor = nn.Sequential(
    nn.Linear(obs_dim, 512), nn.ELU(),
    nn.Linear(512, 256),     nn.ELU(),
    nn.Linear(256, 128),     nn.ELU(),
    nn.Linear(128, act_dim),
).to(device)

optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
loss_fn   = nn.MSELoss()

best_val  = float("inf")
for epoch in range(1, args.epochs + 1):
    actor.train()
    trn_loss = 0.0
    for batch_obs, batch_act in trn_loader:
        pred = actor(batch_obs)
        loss = loss_fn(pred, batch_act)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    trn_loss /= len(trn_loader)

    actor.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_obs, batch_act in val_loader:
            val_loss += loss_fn(actor(batch_obs), batch_act).item()
    val_loss /= max(1, len(val_loader))
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{args.epochs}  trn={trn_loss:.5f}  val={val_loss:.5f}")

    if val_loss < best_val:
        best_val = val_loss
        # rsl_rl actor key 형식으로 저장 (actor.0.weight, ...)
        save_state = {}
        layer_idx = 0
        for i, module in enumerate(actor):
            if isinstance(module, nn.Linear):
                save_state[f"actor.{i}.weight"] = module.weight.data
                save_state[f"actor.{i}.bias"]   = module.bias.data
        torch.save(save_state, args.output)

print(f"[BC] 완료. best_val={best_val:.5f}  저장: {args.output}")
