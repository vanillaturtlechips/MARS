"""Phase 2 Teacher 정책 평가 — place_success_rate 측정.

실행:
  python training/single_robot/eval_manipulation.py \
    --ckpt logs/warehouse_manipulation_teacher/model_2400.pt

  # 여러 체크포인트 비교
  python training/single_robot/eval_manipulation.py \
    --ckpt logs/warehouse_manipulation_teacher/model_2100.pt \
           logs/warehouse_manipulation_teacher/model_2400.pt \
           logs/warehouse_manipulation_teacher/model_2700.pt \
           logs/warehouse_manipulation_teacher/model_2999.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 2 Teacher/Student eval")
parser.add_argument("--ckpt",        type=str, nargs="+", required=True)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--num_envs",    type=int, default=64)
parser.add_argument("--background",  action="store_true", default=False, help="창고 배경 로드 (GUI 시각화용)")
parser.add_argument("--student",     action="store_true", default=False)  # 하위 호환
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if args.livestream == 0 and not getattr(args, "headless", False):
    args.headless = False   # GUI 기본값 유지 — --headless 명시 시만 헤드리스
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os as _os
_rsl_rl_src = "/workspace/rsl_rl"
if _os.path.isdir(_rsl_rl_src) and _rsl_rl_src not in sys.path:
    sys.path.insert(0, _rsl_rl_src)

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parents[2]))
from envs.warehouse.warehouse_manipulation_env import (
    WarehouseManipulationEnv,
    WarehouseManipulationEnvCfg,
    WarehouseManipulationStudentEnvCfg,
    TEACHER_OBS_DIM,
    STUDENT_OBS_DIM,
)


# ------------------------------------------------------------------
# 종료 원인 추적 env 래퍼
# ------------------------------------------------------------------
class EvalManipulationEnv(WarehouseManipulationEnv):
    """_reset_idx 직전에 종료 원인을 기록."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 0=미기록 1=placed 2=dropped 3=timeout
        self._outcome = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stat_placed  = 0
        self.stat_dropped = 0
        self.stat_timeout = 0
        self.stat_ep_lens: list[float] = []

    def _get_dones(self):
        terminated, timed_out = super()._get_dones()

        ee_pos, _ = self._get_ee_pose()
        box_pos_carried = ee_pos + self._grasp_ee_offset
        dist_box_goal   = (box_pos_carried - self._goal_pos_w).norm(dim=1)

        placed  = self._grasped & (dist_box_goal < self.cfg.place_dist_threshold)
        dropped = self._grasped & (self.box.data.root_pos_w[:, 2] < 0.30)

        self._outcome[timed_out]               = 3
        self._outcome[terminated & dropped]    = 2
        self._outcome[terminated & placed]     = 1

        return terminated, timed_out

    def _reset_idx(self, env_ids):
        if env_ids is not None:
            ids = (
                env_ids.long()
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
            )
            for idx in ids:
                o = self._outcome[idx].item()
                ep_len = self.episode_length_buf[idx].item()
                self.stat_ep_lens.append(ep_len)
                if o == 1:
                    self.stat_placed  += 1
                elif o == 2:
                    self.stat_dropped += 1
                else:
                    self.stat_timeout += 1
                self._outcome[idx] = 0
        super()._reset_idx(env_ids)

    def reset_stats(self):
        self.stat_placed  = 0
        self.stat_dropped = 0
        self.stat_timeout = 0
        self.stat_ep_lens = []

    @property
    def total_episodes(self):
        return self.stat_placed + self.stat_dropped + self.stat_timeout


# ------------------------------------------------------------------
# Actor 로드
# ------------------------------------------------------------------
def load_actor(ckpt_path: str, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw  = ckpt.get("model_state_dict", ckpt)

    # 체크포인트 키 정규화 (actor.net.* → net.*)
    sd = {}
    for k, v in raw.items():
        if k.startswith("actor.net."):
            sd[k[len("actor."):]] = v
        elif k.startswith("actor."):
            sd["net." + k[len("actor."):]] = v

    # 체크포인트에서 실제 네트워크 차원 자동 감지
    w_keys = sorted([k for k in sd if k.endswith(".weight")],
                    key=lambda k: int(k.split(".")[1]))
    in_dim     = sd[w_keys[0]].shape[1]   # 첫 레이어 입력 차원
    out_dim    = sd[w_keys[-1]].shape[0]  # 마지막 레이어 출력 차원
    hidden     = [sd[k].shape[0] for k in w_keys[:-1]]  # 히든 레이어 크기

    print(f"  [아키텍처] obs={in_dim}, hidden={hidden}, act={out_dim}")

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
    missing, _ = actor.load_state_dict(sd, strict=True)
    actor.eval()
    return actor, in_dim, out_dim


# ------------------------------------------------------------------
# 단일 체크포인트 평가
# ------------------------------------------------------------------
@torch.inference_mode()
def eval_ckpt(ckpt_path: str, env: EvalManipulationEnv, num_episodes: int, device: str):
    actor, ckpt_obs_dim, ckpt_act_dim = load_actor(ckpt_path, device)
    env_obs_dim = TEACHER_OBS_DIM  # Teacher == Student == 30-dim
    env_act_dim = 4   # Cartesian [dx,dy,dz,gripper]

    obs_mismatch = (ckpt_obs_dim != env_obs_dim or ckpt_act_dim != env_act_dim)
    if obs_mismatch:
        print(f"  [경고] 체크포인트({ckpt_obs_dim}→{ckpt_act_dim})와 "
              f"현재 env({env_obs_dim}→{env_act_dim}) 불일치 → 랜덤 policy 사용")

    obs_dict, _ = env.reset()
    env.reset_stats()          # 초기 reset 이후에 카운터 초기화
    obs = obs_dict["policy"]

    if getattr(args, "livestream", 0):
        print("[Livestream] 브라우저에서 접속하세요. 40초 대기...")
        time.sleep(40)

    while env.total_episodes < num_episodes:
        if obs_mismatch:
            actions = torch.zeros(obs.shape[0], env_act_dim, device=device)
        else:
            actions = actor(obs)
        # DirectRLEnv.step: tensor 입력, (obs, rew, terminated, truncated, extras) 반환
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

    n  = env.total_episodes
    pl = env.stat_placed
    dr = env.stat_dropped
    to = env.stat_timeout
    avg_len = sum(env.stat_ep_lens) / len(env.stat_ep_lens) if env.stat_ep_lens else 0

    return {
        "ckpt":          Path(ckpt_path).name,
        "episodes":      n,
        "place_rate":    pl / n * 100,
        "drop_rate":     dr / n * 100,
        "timeout_rate":  to / n * 100,
        "avg_ep_len":    avg_len,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_cfg = WarehouseManipulationEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.enable_background = args.background
    env = EvalManipulationEnv(env_cfg)

    print(f"\n[Eval] 에피소드: {args.num_episodes}, 병렬 env: {args.num_envs}\n")

    results = []
    for ckpt_path in args.ckpt:
        print(f"  평가 중: {Path(ckpt_path).name} ...")
        r = eval_ckpt(ckpt_path, env, args.num_episodes, device)
        results.append(r)
        print(f"    place {r['place_rate']:.1f}%  drop {r['drop_rate']:.1f}%  "
              f"timeout {r['timeout_rate']:.1f}%  avg_len {r['avg_ep_len']:.1f}")

    print(f"\n{'='*60}")
    print(f"  {'체크포인트':<25} {'place%':>7} {'drop%':>7} {'timeout%':>9} {'avg_len':>8}")
    print(f"  {'-'*57}")
    best = max(results, key=lambda x: x["place_rate"])
    for r in results:
        marker = " ★" if r["ckpt"] == best["ckpt"] else ""
        print(f"  {r['ckpt']:<25} {r['place_rate']:>6.1f}% {r['drop_rate']:>6.1f}% "
              f"{r['timeout_rate']:>8.1f}% {r['avg_ep_len']:>7.1f}{marker}")
    print(f"{'='*60}")
    print(f"  최고 체크포인트: {best['ckpt']} (place {best['place_rate']:.1f}%)\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
