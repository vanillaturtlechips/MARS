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
    --ckpt logs/warehouse_manipulation/model_2999.pt \
    --out  deploy/jetson/actor_phase2_final.pt

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
        "obs_dim": 30,
        "act_dim": 4,
        "hidden_dims": [512, 256, 128],
        "desc": "Phase 2 Teacher/Student (30→4, Cartesian delta)",
    },
    "2s": {
        "obs_dim": 30,
        "act_dim": 4,
        "hidden_dims": [512, 256, 128],
        "desc": "Phase 2 Student (30→4, Teacher=Student 전략)",
    },
    "3": {
        "obs_dim": 17,   # OBS_PER_ROBOT = 7 + 5*(N_ROBOTS-1) = 17 (N_ROBOTS=3)
        "act_dim": 3,
        "hidden_dims": [256, 128, 64],
        "desc": "Phase 3 MARL per-robot actor (17→3)",
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

def _load_actor_weights(ckpt_path: Path, model: ActorMLP) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """actor 가중치 로드. normalizer (mean, var) 도 반환 (없으면 None)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw: dict = ckpt.get("model_state_dict", ckpt)

    actor_sd: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("actor.net."):
            actor_sd[k[len("actor."):]] = v
        elif k.startswith("actor.") and not k.startswith("actor_"):
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

    norm_mean = raw.get("actor_normalizer.running_mean", None)
    norm_var  = raw.get("actor_normalizer.running_var",  None)
    return norm_mean, norm_var


def export(phase: str, checkpoint: str, output: str) -> None:
    cfg = PHASE_CONFIGS[phase]
    ckpt_path = Path(checkpoint)
    out_path  = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg['desc']}]")
    print(f"  체크포인트: {ckpt_path}")
    print(f"  출력: {out_path}")

    base_model = ActorMLP(cfg["obs_dim"], cfg["act_dim"], cfg["hidden_dims"])
    norm_mean, norm_var = _load_actor_weights(ckpt_path, base_model)
    base_model.eval()

    # empirical_normalization=True 로 훈련된 경우 normalizer를 모델에 bake-in
    if norm_mean is not None and norm_var is not None:
        _mean = norm_mean
        _std  = (norm_var + 1e-8).sqrt()

        class NormalizedActorMLP(nn.Module):
            def __init__(self, base: ActorMLP):
                super().__init__()
                self.base = base
                self.register_buffer("mean", _mean)
                self.register_buffer("std",  _std)

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                return self.base((obs - self.mean) / self.std)

        export_model: nn.Module = NormalizedActorMLP(base_model)
        print(f"  [normalizer] bake-in 완료 (mean shape: {_mean.shape})")
    else:
        export_model = base_model

    dummy = torch.zeros(1, cfg["obs_dim"])
    scripted = torch.jit.trace(export_model, dummy)
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
