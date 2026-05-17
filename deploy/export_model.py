"""모델 TorchScript export — Phase 1.5 / Phase 2 / Phase 3.

실행 예시:
  # Phase 1.5 (nav obstacle)
  python deploy/export_model.py --phase 1.5 \
    --ckpt logs/warehouse_obstacle_nav/model_100.pt \
    --out  deploy/jetson/actor_phase15.pt

  # Phase 2 Teacher
  python deploy/export_model.py --phase 2t \
    --ckpt logs/warehouse_manipulation_teacher/model_3000.pt \
    --out  deploy/jetson/actor_phase2_teacher.pt

  # Phase 2 Student
  python deploy/export_model.py --phase 2s \
    --ckpt logs/warehouse_manipulation_student/model_1500.pt \
    --out  deploy/jetson/actor_phase2_student.pt

  # Phase 3 MARL (per-robot actor)
  python deploy/export_model.py --phase 3 \
    --ckpt logs/warehouse_mappo/model_5399.pt \
    --out  deploy/jetson/actor_phase3_marl.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# 페이즈별 아키텍처 매개변수
# ──────────────────────────────────────────────────────────────────────────────

PHASE_CONFIGS: dict[str, dict] = {
    "1.5": {
        "obs_dim": 7,
        "act_dim": 3,
        "hidden_dims": [256, 128, 64],
        "desc": "Phase 1.5 장애물 내비게이션 (7→3)",
    },
    "2t": {
        "obs_dim": 33,
        "act_dim": 9,
        "hidden_dims": [512, 256, 128],
        "desc": "Phase 2 Teacher (33→9, 특권 정보)",
    },
    "2s": {
        "obs_dim": 25,
        "act_dim": 9,
        "hidden_dims": [512, 256, 128],
        "desc": "Phase 2 Student (25→9, 센서 관측)",
    },
    "3": {
        "obs_dim": 9,    # OBS_PER_ROBOT = 7 + (N_ROBOTS-1) = 9
        "act_dim": 3,
        "hidden_dims": [256, 128, 64],
        "desc": "Phase 3 MARL per-robot actor (9→3)",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Actor MLP (TorchScript 호환)
# ──────────────────────────────────────────────────────────────────────────────

class ActorMLP(nn.Module):
    """rsl_rl ActorCritic 의 actor 부분만 재현."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).tanh()


# ──────────────────────────────────────────────────────────────────────────────

def _load_actor_weights(ckpt_path: Path, model: ActorMLP) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw: dict = ckpt.get("model_state_dict", ckpt)

    actor_sd: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("actor.net."):
            actor_sd[k[len("actor."):]] = v
        elif k.startswith("actor."):
            actor_sd["net." + k[len("actor."):]] = v

    if not actor_sd:
        # rsl_rl 3.x 일부 저장 형식 재시도
        for k, v in raw.items():
            if "actor" not in k and k.startswith("net."):
                actor_sd[k] = v
        if not actor_sd:
            raise KeyError(
                f"actor 가중치 없음. 저장된 키 (처음 10개): {list(raw.keys())[:10]}"
            )

    missing, unexpected = model.load_state_dict(actor_sd, strict=False)
    if missing:
        print(f"[경고] 누락 가중치: {missing}")
    if unexpected:
        print(f"[경고] 예상 외 키: {unexpected}")


def export(phase: str, checkpoint: str, output: str) -> None:
    cfg = PHASE_CONFIGS[phase]
    ckpt_path = Path(checkpoint)
    out_path  = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg['desc']}]")
    print(f"  체크포인트: {ckpt_path}")
    print(f"  출력: {out_path}")

    model = ActorMLP(cfg["obs_dim"], cfg["act_dim"], cfg["hidden_dims"])
    _load_actor_weights(ckpt_path, model)
    model.eval()

    dummy = torch.zeros(1, cfg["obs_dim"])
    scripted = torch.jit.trace(model, dummy)
    scripted = torch.jit.freeze(scripted)
    torch.jit.save(scripted, str(out_path))

    # 검증
    loaded = torch.jit.load(str(out_path))
    out = loaded(dummy)
    assert out.shape == (1, cfg["act_dim"]), f"출력 shape 오류: {out.shape}"
    assert out.abs().max() <= 1.0 + 1e-5, "tanh 범위 초과"

    print(f"  검증 통과 — 출력 shape: {list(out.shape)}, 최대값: {out.abs().max().item():.4f}")
    print(f"  TorchScript 저장 완료: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARS 모델 TorchScript export")
    parser.add_argument("--phase", choices=list(PHASE_CONFIGS.keys()), required=True,
                        help="1.5 | 2t | 2s | 3")
    parser.add_argument("--ckpt",  required=True, help="rsl_rl 체크포인트 경로")
    parser.add_argument("--out",   required=True, help="TorchScript 출력 경로")
    args = parser.parse_args()

    export(args.phase, args.ckpt, args.out)
