"""Phase 2 데모 녹화 — Isaac Lab Camera 센서로 headless frame capture → MP4

실행:
  apt install -y ffmpeg  # 한 번만

  python training/single_robot/demo_record.py \
    --ckpt logs/warehouse_manipulation/model_2999.pt \
    --num_episodes 3 \
    --output /workspace/phase2_demo.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 demo recorder")
parser.add_argument("--ckpt",         type=str, required=True)
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=3)
parser.add_argument("--capture_every",type=int, default=2)   # N스텝마다 1프레임 (30Hz→15fps)
parser.add_argument("--fps",          type=int, default=15)
parser.add_argument("--output",       type=str, default="/workspace/phase2_demo.mp4")
parser.add_argument("--width",        type=int, default=640)
parser.add_argument("--height",       type=int, default=360)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless       = True
args.enable_cameras = True

app_launcher    = AppLauncher(args)
simulation_app  = app_launcher.app

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv,
    WarehouseManipulationEnvCfg,
)


# ──────────────────────────────────────────────────────────────────
# Actor 로드
# ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────
# Camera 셋업 — Isaac Lab Camera 센서 (replicator orchestrator 불필요)
# ──────────────────────────────────────────────────────────────────
def _lookat_quat_wxyz(eye, target):
    """Look-at → wxyz quaternion (numpy only)."""
    e, t = np.array(eye, float), np.array(target, float)
    u = np.array([0., 0., 1.])
    z = e - t;  z /= np.linalg.norm(z)
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


def setup_camera(env: WarehouseManipulationEnv, width: int, height: int):
    from isaaclab.sensors import Camera, CameraCfg
    import isaaclab.sim as sim_utils

    origin = env.scene.env_origins[0].cpu().numpy()
    eye    = (origin + np.array([1.4, -1.2, 1.3])).tolist()
    target = (origin + np.array([0.5,  0.1,  0.5])).tolist()
    quat   = _lookat_quat_wxyz(eye, target)

    cam_cfg = CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(eye),
            rot=quat,
            convention="world",
        ),
    )
    camera = Camera(cam_cfg)
    return camera


# ──────────────────────────────────────────────────────────────────
# 영상 생성
# ──────────────────────────────────────────────────────────────────
def frames_to_video(frame_dir: Path, output: str, fps: int):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output,
    ]
    print(f"\n[ffmpeg] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ffmpeg 오류]\n{result.stderr}")
    else:
        size_mb = Path(output).stat().st_size / 1e6
        print(f"[완료] {output}  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs  = args.num_envs
    env_cfg.enable_background = True
    env = WarehouseManipulationEnv(env_cfg)

    actor = load_actor(args.ckpt, device)

    obs_dict, _ = env.reset()

    # Camera 초기화 (env.reset() 이후)
    camera = setup_camera(env, args.width, args.height)
    camera.reset()

    frame_dir = Path(tempfile.mkdtemp(prefix="mars_demo_"))
    print(f"\n[Record] 프레임 저장: {frame_dir}")
    print(f"[Record] {args.num_episodes}에피소드  capture_every={args.capture_every}  → {args.output}\n")

    episode_count = 0
    frame_idx     = 0
    step_count    = 0

    with torch.inference_mode():
        while episode_count < args.num_episodes and simulation_app.is_running():
            obs = obs_dict["policy"]
            actions = actor(obs)
            obs_dict, _, terminated, truncated, extras = env.step(actions)

            # Camera 업데이트 — env.step()이 이미 렌더링, 여기선 데이터만 읽음
            camera.update(dt=env.sim.get_rendering_dt())

            step_count += 1
            if step_count % args.capture_every == 0:
                rgb = camera.data.output.get("rgb")
                if rgb is not None:
                    # (1, H, W, 4) RGBA → (H, W, 3) RGB numpy
                    img = rgb.squeeze(0).cpu().numpy()[:, :, :3]
                    Image.fromarray(img.astype(np.uint8)).save(
                        frame_dir / f"frame_{frame_idx:06d}.png"
                    )
                    frame_idx += 1

            done = terminated | truncated
            if done.any():
                episode_count += done.sum().item()
                log  = extras.get("log", {})
                rate = log.get("place_rate", 0.0)
                print(f"  에피소드 {episode_count}/{args.num_episodes}"
                      f"  place_rate={rate:.1f}%  프레임={frame_idx}")

    env.close()

    if frame_idx == 0:
        print("[오류] 캡처된 프레임 없음")
        return

    frames_to_video(frame_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
    simulation_app.close()
