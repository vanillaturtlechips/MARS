"""BC 데이터 수집: Teacher 실행 → (student_obs_29D, teacher_action_9D) 저장

Usage (RunPod):
  python training/single_robot/collect_bc_data.py \
    --headless --num_envs 512 \
    --teacher_ckpt logs/warehouse_manipulation/model_999.pt \
    --output bc_data.pt --num_steps 500000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs",      type=int,   default=512)
parser.add_argument("--teacher_ckpt",  type=str,   required=True)
parser.add_argument("--output",        type=str,   default="bc_data.pt")
parser.add_argument("--num_steps",     type=int,   default=500_000)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv,
    WarehouseManipulationStudentTransportEnvCfg,
    TEACHER_OBS_DIM,
    STUDENT_OBS_DIM,
)


def build_teacher_actor(obs_dim: int, act_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(obs_dim, 512), nn.ELU(),
        nn.Linear(512, 256),     nn.ELU(),
        nn.Linear(256, 128),     nn.ELU(),
        nn.Linear(128, act_dim),
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    state = ckpt["model_state_dict"]

    act_dim = state["actor.6.weight"].shape[0]
    obs_dim  = state["actor.0.weight"].shape[1]
    print(f"[Teacher] obs={obs_dim}D  action={act_dim}D")
    assert obs_dim == TEACHER_OBS_DIM, \
        f"Teacher obs {obs_dim} != TEACHER_OBS_DIM {TEACHER_OBS_DIM}."

    teacher = build_teacher_actor(obs_dim, act_dim).to(device)
    actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    teacher.load_state_dict(actor_state)
    teacher.eval()

    # Student env (student_mode=True, force_grasp=True) → obs["policy"]=29D, obs["critic"]=30D
    env_cfg = WarehouseManipulationStudentTransportEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = WarehouseManipulationEnv(env_cfg)
    env = RslRlVecEnvWrapper(env)

    all_obs  = []
    all_acts = []

    obs_dict, _ = env.reset()
    collected = 0
    print(f"[BC Collect] 목표 {args.num_steps:,} steps 수집 중...")

    while collected < args.num_steps:
        student_obs   = obs_dict["policy"]    # (N, 29)
        teacher_input = obs_dict["critic"]    # (N, 30)

        with torch.no_grad():
            teacher_action = teacher(teacher_input)  # (N, 9)

        all_obs.append(student_obs.cpu())
        all_acts.append(teacher_action.cpu())
        collected += args.num_envs

        obs_dict, _, _, _ = env.step(teacher_action)

        if collected % 50_000 == 0:
            print(f"  {collected:,} / {args.num_steps:,}")

    env.close()

    obs_tensor  = torch.cat(all_obs,  dim=0)
    acts_tensor = torch.cat(all_acts, dim=0)
    torch.save({"obs": obs_tensor, "actions": acts_tensor}, args.output)
    print(f"[BC Collect] 저장 완료: {args.output}  ({obs_tensor.shape[0]:,} pairs)")


if __name__ == "__main__":
    main()
    simulation_app.close()
