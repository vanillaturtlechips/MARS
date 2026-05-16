"""Jetson 추론 엔진 — TorchScript actor 로드 후 관측 → 행동 변환.

단독 실행 시 더미 관측으로 10회 추론해 latency 측정.
ros2_bridge.py 에서 import 해 사용.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

# Phase 1.5 행동 스케일 (warehouse_obstacle_env.py 와 동일)
MAX_VX    = 1.5   # m/s
MAX_VY    = 1.0   # m/s
MAX_OMEGA = 2.0   # rad/s

OBS_DIM = 7
ACT_DIM = 3

# 기본 모델 경로 (ros2_bridge 또는 단독 실행 시 오버라이드 가능)
DEFAULT_MODEL = Path(__file__).parent / "actor_phase15.pt"


class WarehousePolicy:
    """TorchScript actor 래퍼 — 스레드 안전, CPU 전용."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"모델 파일 없음: {path}\n"
                "훈련 PC에서 deploy/export_model.py 를 먼저 실행하세요."
            )
        self._model = torch.jit.load(str(path), map_location="cpu")
        self._model.eval()
        print(f"[WarehousePolicy] 모델 로드 완료: {path}")

    @torch.inference_mode()
    def act(
        self,
        goal_x_body: float,
        goal_y_body: float,
        goal_dist:   float,
        vx_body:     float,
        vy_body:     float,
        omega_z:     float,
        min_obs_dist: float,
    ) -> tuple[float, float, float]:
        """관측 7개 → (cmd_vx, cmd_vy, cmd_omega) [m/s, m/s, rad/s]."""
        obs = torch.tensor(
            [[goal_x_body, goal_y_body, goal_dist, vx_body, vy_body, omega_z, min_obs_dist]],
            dtype=torch.float32,
        )
        action = self._model(obs)[0]   # (3,) tanh 출력 [-1, 1]
        cmd_vx    = float(action[0]) * MAX_VX
        cmd_vy    = float(action[1]) * MAX_VY
        cmd_omega = float(action[2]) * MAX_OMEGA
        return cmd_vx, cmd_vy, cmd_omega


# ---------------------------------------------------------------------------
# 단독 실행: latency 벤치마크
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jetson 추론 latency 측정")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--n", type=int, default=100, help="반복 횟수")
    args = parser.parse_args()

    policy = WarehousePolicy(args.model)

    # 워밍업
    for _ in range(10):
        policy.act(1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)

    t0 = time.perf_counter()
    for _ in range(args.n):
        cmd_vx, cmd_vy, cmd_omega = policy.act(1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n--- 추론 결과 ---")
    print(f"cmd_vx={cmd_vx:.3f} m/s  cmd_vy={cmd_vy:.3f} m/s  cmd_omega={cmd_omega:.3f} rad/s")
    print(f"\n--- Latency ({args.n}회 평균) ---")
    print(f"{elapsed / args.n:.2f} ms/iter  ({1000 * args.n / elapsed:.0f} Hz)")
    print(f"목표: < 10 ms (100 Hz)  — Jetson Orin Nano Super 기준 충분히 달성 가능")
