"""Phase 2 데모 — Teacher 정책 시각화 (창고 배경 + GUI)

실행 (로컬 GUI):
  python training/single_robot/demo_manipulation.py \
    --ckpt logs/warehouse_manipulation_teacher/model_2999.pt \
    --num_envs 4

실행 (RunPod Livestream):
  python training/single_robot/demo_manipulation.py \
    --ckpt logs/warehouse_manipulation_teacher/model_2999.pt \
    --num_envs 4 --livestream 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher
import numpy as np

def _lookat_quat_wxyz(eye, target):
    """카메라 시점 → wxyz quaternion 변환"""
    e, t = np.array(eye, float), np.array(target, float)
    u = np.array([0., 0., 1.])
    z = e - t
    z /= np.linalg.norm(z)
    x = np.cross(u, z)
    if np.linalg.norm(x) < 1e-6:
        u = np.array([0., 1., 0.])
        x = np.cross(u, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1)
        return (0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s)
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
        return ((R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s)
    elif R[1,1] > R[2,2]:
        s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
        return ((R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s)
    else:
        s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
        return ((R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s)

parser = argparse.ArgumentParser(description="Phase 2 Teacher 데모")
parser.add_argument("--ckpt",         type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=4)
parser.add_argument("--num_episodes", type=int, default=0,
                    help="이 수만큼 에피소드 후 자동 종료 (0=무한)")
parser.add_argument("--record", action="store_true",
                    help="헤드리스 카메라 녹화(UI 없는 깨끗한 mp4). nav 데모와 동일 방식.")
parser.add_argument("--video_out", type=str, default="/workspace/phase2_demo.mp4",
                    help="--record 시 저장 경로")
parser.add_argument("--cam_eye", type=str, default="2.0,1.8,1.6",
                    help="카메라 위치 x,y,z (작업공간 z≈0.8~1.2 기준)")
parser.add_argument("--cam_target", type=str, default="0.5,0.0,0.95",
                    help="카메라 타겟 x,y,z (팔 베이스 z=0.8, 박스 z=1.15 사이)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if args.record:
    args.enable_cameras = True   # view_camera 오프스크린 렌더 필요(헤드리스에서도 카메라는 렌더됨)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv,
    WarehouseManipulationEnvCfg,
)


def load_actor(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw  = ckpt.get("model_state_dict", ckpt)

    norm_mean = raw.get("actor_normalizer.running_mean", None)
    norm_var  = raw.get("actor_normalizer.running_var",  None)

    sd = {}
    for k, v in raw.items():
        if k.startswith("actor.net."):
            sd[k[len("actor."):]] = v
        elif k.startswith("actor.") and not k.startswith("actor_"):
            sd["net." + k[len("actor."):]] = v

    w_keys = sorted([k for k in sd if k.endswith(".weight")],
                    key=lambda k: int(k.split(".")[1]))
    in_dim  = sd[w_keys[0]].shape[1]
    out_dim = sd[w_keys[-1]].shape[0]
    hidden  = [sd[k].shape[0] for k in w_keys[:-1]]
    print(f"[Actor] obs={in_dim}D  action={out_dim}D  hidden={hidden}")

    class ActorMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = []
            d = in_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ELU()]
                d = h
            layers.append(nn.Linear(d, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).tanh()

    actor = ActorMLP().to(device)
    actor.load_state_dict(sd, strict=True)
    actor.eval()

    if norm_mean is not None:
        _mean = norm_mean.to(device)
        _std  = (norm_var.to(device) + 1e-8).sqrt()

        class NormalizedActor(nn.Module):
            def __init__(self, base: nn.Module):
                super().__init__()
                self.base = base

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.base((x - _mean) / _std)

        actor = NormalizedActor(actor).to(device)

    return actor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs  = args.num_envs
    env_cfg.enable_background = True   # 창고 배경 + 조명
    eye    = tuple(float(v) for v in args.cam_eye.split(","))
    target = tuple(float(v) for v in args.cam_target.split(","))
    # --record: nav 데모와 동일한 env.render(rgb_array)+RecordVideo 경로(이 파드에서 검증됨).
    #   이 파드는 standalone Camera 센서 초기화가 깨짐("Camera could not be initialized") →
    #   Camera 대신 env 자체 렌더를 녹화. UI 없음. (demo_play.py와 동일 메커니즘)
    env = WarehouseManipulationEnv(env_cfg, render_mode="rgb_array" if args.record else None)
    env._demo_place_override = True  # goal 근처에서 자동 박스 해제 (raw env에 설정 — wrap 전)

    actor = load_actor(args.ckpt, device)

    view_camera = None
    if args.record:
        import gymnasium as gym
        import os
        os.makedirs("/tmp/arm_rec", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder="/tmp/arm_rec", step_trigger=lambda s: s == 0,
            video_length=100000, disable_logger=True, name_prefix="arm",
        )
        try:
            env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        except Exception as _e:
            print(f"[Demo] 카메라뷰 설정 실패(무시): {_e}")
    else:
        quat = _lookat_quat_wxyz(eye, target)
        cam_cfg = CameraCfg(
            prim_path="/World/ViewCamera", update_period=0.0, height=720, width=1280,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
            offset=CameraCfg.OffsetCfg(pos=eye, rot=quat, convention="world"),
        )
        view_camera = Camera(cam_cfg)

    print(f"\n[Demo] {args.num_envs}개 env, 창고 배경 ON")
    print("[Demo] 카메라: /World/ViewCamera (stage에서 수정 가능)")
    print("[Demo] 종료: Ctrl+C\n")

    if getattr(args, "livestream", 0):
        print("[Livestream] 연결 대기 중 (40초)...")
        time.sleep(40)

    placed_count = 0
    episode_count = 0

    obs_dict, _ = env.reset()   # --record면 RecordVideo가 이 step부터 자동 녹화

    with torch.inference_mode():
        while simulation_app.is_running():
            obs = obs_dict["policy"]
            actions = actor(obs).clone()

            obs_dict, _, terminated, truncated, extras = env.step(actions)

            done = terminated | truncated
            if done.any():
                episode_count += done.sum().item()
                log = extras.get("log", {})
                if "place_rate" in log:
                    rate = log["place_rate"]
                    print(f"  누적 place_rate: {rate:.1f}%  (에피소드 {episode_count}개)")

            if args.num_episodes > 0 and episode_count >= args.num_episodes:
                print(f"[Demo] {episode_count}에피소드 완료 — 종료")
                break

    env.close()   # RecordVideo가 여기서 mp4 마무리

    if args.record:
        import glob, shutil
        mp4s = sorted(glob.glob("/tmp/arm_rec/*.mp4"))
        if mp4s:
            shutil.copy(mp4s[-1], args.video_out)
            print(f"[Demo] 녹화 저장: {args.video_out}  (RecordVideo, UI 없음·헤드리스)")
        else:
            print("[Demo] ⚠ RecordVideo가 mp4를 안 만듦 — env.render 확인")


if __name__ == "__main__":
    main()
    simulation_app.close()
