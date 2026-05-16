"""Phase 1.5 모델을 TorchScript로 export — 훈련 PC에서 실행."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


# Phase 1.5 관측 공간 정의 (warehouse_obstacle_env.py 와 동일)
OBS_DIM = 7    # [goal_x_body, goal_y_body, goal_dist, vx_body, vy_body, omega_z, min_obs_dist]
ACT_DIM = 3    # [cmd_vx, cmd_vy, cmd_omega]  — 모두 [-1, 1] 정규화

# rsl_rl MLP 구조 (agents/rsl_rl_ppo_cfg.py 에서 가져온 값)
HIDDEN_DIMS = [256, 128, 64]


class ActorMLP(torch.nn.Module):
    """rsl_rl ActorCritic 의 actor 부분만 재현 — TorchScript 호환."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, act_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).tanh()   # actor 출력은 tanh → [-1, 1]


def _load_actor_weights(checkpoint_path: Path, model: ActorMLP) -> None:
    """rsl_rl 체크포인트에서 actor 가중치만 추출해 로드."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # rsl_rl 저장 형식: 'model_state_dict' 키 안에 'actor.net.*' 또는 'actor.*'
    raw: dict = ckpt.get("model_state_dict", ckpt)

    actor_sd: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        # 'actor.net.0.weight' → 'net.0.weight'
        if k.startswith("actor."):
            actor_sd["net." + k[len("actor."):]] = v

    if not actor_sd:
        raise KeyError(
            "actor 가중치를 체크포인트에서 찾지 못함. "
            f"저장된 키: {list(raw.keys())[:10]}"
        )

    missing, unexpected = model.load_state_dict(actor_sd, strict=False)
    if missing:
        print(f"[경고] 누락된 가중치: {missing}")
    if unexpected:
        print(f"[경고] 예상치 못한 키: {unexpected}")


def export(checkpoint: str, output: str) -> None:
    ckpt_path = Path(checkpoint)
    out_path  = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"체크포인트 로드: {ckpt_path}")
    model = ActorMLP(OBS_DIM, ACT_DIM, HIDDEN_DIMS)
    _load_actor_weights(ckpt_path, model)
    model.eval()

    # TorchScript trace — 입력 shape 고정 (배치 1)
    dummy = torch.zeros(1, OBS_DIM)
    scripted = torch.jit.trace(model, dummy)
    scripted = torch.jit.freeze(scripted)   # 추론 전용, 가중치 상수화

    torch.jit.save(scripted, str(out_path))
    print(f"TorchScript 저장 완료: {out_path}")

    # 간단 검증
    loaded = torch.jit.load(str(out_path))
    out = loaded(dummy)
    assert out.shape == (1, ACT_DIM), f"출력 shape 오류: {out.shape}"
    assert out.abs().max() <= 1.0 + 1e-5, "tanh 범위 초과"
    print(f"검증 통과 — 출력: {out.detach().numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1.5 모델 TorchScript export")
    parser.add_argument(
        "--checkpoint",
        default="logs/warehouse_obstacle_nav/model_100.pt",
        help="rsl_rl 체크포인트 경로",
    )
    parser.add_argument(
        "--output",
        default="deploy/jetson/actor_phase15.pt",
        help="TorchScript 출력 경로",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output)
