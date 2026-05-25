"""Phase 2 데모 녹화 — headless frame capture → MP4

실행:
  # ffmpeg 먼저 설치 (한 번만)
  apt install -y ffmpeg

  python training/single_robot/demo_record.py \
    --ckpt logs/warehouse_manipulation/model_2999.pt \
    --num_episodes 5 \
    --output /workspace/phase2_demo.mp4
"""

from __future__ import annotations

import argparse
import sys
import subprocess
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 demo recorder")
parser.add_argument("--ckpt",        type=str, required=True)
parser.add_argument("--num_envs",    type=int, default=1)
parser.add_argument("--num_episodes",type=int, default=5)
parser.add_argument("--fps",         type=int, default=5)
parser.add_argument("--output",      type=str, default="/workspace/phase2_demo.mp4")
parser.add_argument("--width",       type=int, default=640)
parser.add_argument("--height",      type=int, default=360)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True        # 항상 headless
args.enable_cameras = True  # omni.replicator 확장 활성화

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np

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


def _set_camera_lookat(eye, target):
    """pxr USD API로 카메라 위치/방향 설정 (omni.isaac.core 불필요)."""
    from pxr import UsdGeom, Gf
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cam = stage.GetPrimAtPath("/OmniverseKit_Persp")
    if not cam.IsValid():
        print("[Camera] /OmniverseKit_Persp 없음 — 기본 위치 사용")
        return

    e = np.array(eye, dtype=float)
    t = np.array(target, dtype=float)
    u = np.array([0.0, 0.0, 1.0])

    fwd = t - e
    fwd /= np.linalg.norm(fwd)
    rgt = np.cross(fwd, u)
    if np.linalg.norm(rgt) < 1e-6:
        u = np.array([0.0, 1.0, 0.0])
        rgt = np.cross(fwd, u)
    rgt /= np.linalg.norm(rgt)
    up2 = np.cross(rgt, fwd)

    # USD 카메라는 -Z 방향을 바라봄 (OpenGL 규약)
    m = Gf.Matrix4d(
        rgt[0], up2[0], -fwd[0], 0,
        rgt[1], up2[1], -fwd[1], 0,
        rgt[2], up2[2], -fwd[2], 0,
        e[0],   e[1],    e[2],   1,
    )
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


def setup_camera(env: WarehouseManipulationEnv, width: int, height: int):
    """env_0 기준으로 카메라 위치 설정 후 렌더 프로덕트 반환."""
    import omni.replicator.core as rep

    origin = env.scene.env_origins[0].cpu().numpy()
    eye    = (origin + np.array([1.4, -1.2, 1.3])).tolist()
    target = (origin + np.array([0.5,  0.1, 0.5])).tolist()
    _set_camera_lookat(eye, target)

    rp = rep.create.render_product("/OmniverseKit_Persp", (width, height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    return rp, annot


def capture_frame(annot) -> np.ndarray | None:
    import omni.replicator.core as rep
    rep.orchestrator.step(rt_subframes=0, delta_time=0.0)
    data = annot.get_data()
    if data is None or data.size == 0:
        return None
    return data[:, :, :3]   # RGB (H, W, 3) uint8


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
        print(f"[완료] {output}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs  = args.num_envs
    env_cfg.enable_background = True
    env = WarehouseManipulationEnv(env_cfg)

    actor = load_actor(args.ckpt, device)

    obs_dict, _ = env.reset()

    rp, annot = setup_camera(env, args.width, args.height)

    frame_dir = Path(tempfile.mkdtemp(prefix="mars_demo_"))
    print(f"\n[Record] 프레임 저장 위치: {frame_dir}")
    print(f"[Record] 목표: {args.num_episodes}에피소드, 출력: {args.output}\n")

    from PIL import Image

    episode_count = 0
    frame_idx = 0
    step_count = 0
    CAPTURE_EVERY = 6   # 30Hz 시뮬 기준 5fps 영상 — orchestrator.step() 호출 최소화

    with torch.inference_mode():
        while episode_count < args.num_episodes and simulation_app.is_running():
            obs = obs_dict["policy"]
            actions = actor(obs)
            obs_dict, _, terminated, truncated, extras = env.step(actions)
            step_count += 1

            rgb = None
            if step_count % CAPTURE_EVERY == 0:
                rgb = capture_frame(annot)
            if rgb is not None:
                Image.fromarray(rgb).save(frame_dir / f"frame_{frame_idx:06d}.png")
                frame_idx += 1

            done = terminated | truncated
            if done.any():
                episode_count += done.sum().item()
                log = extras.get("log", {})
                rate = log.get("place_rate", 0.0)
                print(f"  에피소드 {episode_count}/{args.num_episodes}  place_rate={rate:.1f}%  프레임={frame_idx}")

    env.close()

    if frame_idx == 0:
        print("[오류] 캡처된 프레임 없음 — replicator annotator 확인 필요")
        return

    frames_to_video(frame_dir, args.output, args.fps)


if __name__ == "__main__":
    main()
    simulation_app.close()
