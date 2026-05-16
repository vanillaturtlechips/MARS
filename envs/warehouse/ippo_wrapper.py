"""IPPO 배치 재구성 래퍼.

RslRlVecEnvWrapper 위에 씌워서, joint obs/act 를 per-robot 배치로 확장.

  joint  (E, N × obs_per_robot) → per-robot (E×N, obs_per_robot)
  joint  (E, N × act_per_robot) ← per-robot (E×N, act_per_robot)

협력 보상: 팀 보상을 각 로봇에 동일하게 부여 (cooperative MARL 표준 관행).

배치 확장 방식:
  env 0 robot 0, env 0 robot 1, env 0 robot 2,
  env 1 robot 0, env 1 robot 1, env 1 robot 2, ...
  → repeat_interleave(N) 와 일치
"""

from __future__ import annotations

import torch


class IPPOReshapeWrapper:
    """rsl_rl OnPolicyRunner ↔ RslRlVecEnvWrapper 사이 삽입 래퍼.

    Parameters
    ----------
    vec_env       : RslRlVecEnvWrapper (num_envs = E)
    n_robots      : 로봇 수 N
    obs_per_robot : 로봇 1대 관측 차원
    """

    def __init__(self, vec_env, n_robots: int, obs_per_robot: int):
        self._env = vec_env
        self.n = n_robots
        self.obs_per_robot = obs_per_robot
        self._E = vec_env.num_envs

    # ── rsl_rl 필수 속성 ──────────────────────────────────────────
    @property
    def num_envs(self) -> int:
        return self._E * self.n

    @property
    def device(self):
        return self._env.device

    # ── rsl_rl 3.x 인터페이스 ────────────────────────────────────
    def step(self, actions: torch.Tensor):
        """(E×N, act_per_robot) → joint step → (E×N, obs_per_robot)."""
        act_per_robot = actions.shape[-1]
        joint_act = actions.view(self._E, self.n * act_per_robot)

        obs, rew, dones, extras = self._env.step(joint_act)

        obs_split  = self._split_obs(obs)
        rew_exp    = rew.repeat_interleave(self.n)
        dones_exp  = dones.repeat_interleave(self.n)
        extras_exp = self._expand_extras(extras)

        return obs_split, rew_exp, dones_exp, extras_exp

    def get_observations(self):
        result = self._env.get_observations()
        if isinstance(result, tuple):
            obs, extras = result
            return self._split_obs(obs), extras
        return self._split_obs(result)

    def reset(self):
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, extras = result
            return self._split_obs(obs), extras
        return self._split_obs(result)

    # ── 내부 유틸 ─────────────────────────────────────────────────
    def _split_obs(self, obs):
        """(E, N×obs_per_robot) → (E×N, obs_per_robot)."""
        if isinstance(obs, dict):
            return {k: v.reshape(-1, self.obs_per_robot) for k, v in obs.items()}
        if isinstance(obs, torch.Tensor):
            return obs.reshape(-1, self.obs_per_robot)
        return obs

    def _expand_extras(self, extras: dict) -> dict:
        """shape (E, ...) 인 텐서를 (E×N, ...) 로 확장."""
        if not isinstance(extras, dict):
            return extras
        out = {}
        for k, v in extras.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == self._E:
                out[k] = v.repeat_interleave(self.n)
            elif isinstance(v, dict):
                out[k] = {
                    kk: vv.repeat_interleave(self.n)
                    if isinstance(vv, torch.Tensor) and vv.shape[0] == self._E
                    else vv
                    for kk, vv in v.items()
                }
            else:
                out[k] = v
        return out

    def __getattr__(self, name):
        return getattr(self._env, name)
