"""True CTDE 래퍼 (Centralized Training, Decentralized Execution).

Actor : per-robot obs (9-dim) — 각 로봇 독립 실행
Critic: global obs (27-dim)  — 전체 상태 중앙 평가

보상 분리:
  env._get_rewards() → extras["per_robot_rewards"] (E, N_ROBOTS)
  wrapper가 reshape(-1) → (E*N,) 로 각 actor에 개별 분배
"""

from __future__ import annotations

import collections.abc

import gymnasium as gym
import numpy as np
import torch


class IPPOReshapeWrapper:
    """rsl_rl OnPolicyRunner ↔ RslRlVecEnvWrapper 사이 삽입 래퍼.

    Parameters
    ----------
    vec_env        : RslRlVecEnvWrapper (num_envs = E)
    n_robots       : 로봇 수 N
    obs_per_robot  : 로봇 1대 관측 차원 (actor obs)
    act_per_robot  : 로봇 1대 행동 차원 (기본 3)
    """

    def __init__(self, vec_env, n_robots: int, obs_per_robot: int, act_per_robot: int = 3):
        self._env = vec_env
        self.n = n_robots
        self.obs_per_robot = obs_per_robot
        self.act_per_robot = act_per_robot
        self._E = vec_env.num_envs

    # ── rsl_rl 필수 속성 ──────────────────────────────────────────
    @property
    def num_envs(self) -> int:
        return self._E * self.n

    @property
    def num_obs(self) -> int:
        return self.obs_per_robot  # actor: 9-dim per-robot

    @property
    def num_actions(self) -> int:
        return self.act_per_robot

    @property
    def num_privileged_obs(self) -> int:
        return self.obs_per_robot * self.n  # critic: 27-dim global state

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_per_robot,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.act_per_robot,),
            dtype=np.float32,
        )

    @property
    def device(self):
        return self._env.device

    # ── rsl_rl 3.x 인터페이스 ────────────────────────────────────
    def step(self, actions: torch.Tensor):
        """(E×N, act_per_robot) → joint step → (E×N, obs_per_robot)."""
        joint_act = actions.reshape(self._E, self.n * self.act_per_robot)

        obs, rew, dones, extras = self._env.step(joint_act)

        # obs: (E, N*obs_per_robot) = (E, 27)
        actor_obs = self._split_actor_obs(obs)          # (E*N, 9)
        critic_obs = self._expand_critic_obs(obs)       # (E*N, 27)

        # per-robot 보상 분리 (credit assignment)
        per_robot_rew = extras.pop("per_robot_rewards", None)
        if per_robot_rew is not None:
            rew_exp = per_robot_rew.reshape(-1)         # (E*N,)
        else:
            rew_exp = rew.repeat_interleave(self.n, dim=0)

        dones_exp = dones.repeat_interleave(self.n, dim=0)
        extras_exp = self._expand_extras(extras)

        # critic obs를 extras에 삽입 → rsl_rl이 privileged obs로 사용
        if "observations" not in extras_exp:
            extras_exp["observations"] = {}
        extras_exp["observations"]["critic"] = critic_obs

        return actor_obs, rew_exp, dones_exp, extras_exp

    def get_observations(self):
        result = self._env.get_observations()
        if isinstance(result, tuple):
            obs, extras = result
            actor_obs = self._split_actor_obs(obs)
            critic_obs = self._expand_critic_obs(obs)
            if not isinstance(extras, dict):
                extras = {}
            if "observations" not in extras:
                extras["observations"] = {}
            extras["observations"]["critic"] = critic_obs
            return actor_obs, extras
        obs = result
        actor_obs = self._split_actor_obs(obs)
        critic_obs = self._expand_critic_obs(obs)
        return actor_obs, {"observations": {"critic": critic_obs}}

    def reset(self):
        result = self._env.reset()
        if isinstance(result, tuple):
            obs, extras = result
            actor_obs = self._split_actor_obs(obs)
            critic_obs = self._expand_critic_obs(obs)
            if not isinstance(extras, dict):
                extras = {}
            if "observations" not in extras:
                extras["observations"] = {}
            extras["observations"]["critic"] = critic_obs
            return actor_obs, extras
        obs = result
        return self._split_actor_obs(obs), {}

    # ── 내부 유틸 ─────────────────────────────────────────────────
    def _split_actor_obs(self, obs) -> torch.Tensor:
        """(E, N*obs_per_robot) → (E*N, obs_per_robot): per-robot actor 입력."""
        # TensorDict / dict / tensor 모두 처리
        if hasattr(obs, "batch_size"):          # TensorDict
            obs = obs["policy"]
        elif isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        return obs.reshape(self._E * self.n, self.obs_per_robot)

    def _expand_critic_obs(self, obs) -> torch.Tensor:
        """(E, N*obs_per_robot) → (E*N, N*obs_per_robot): global state 복제."""
        if hasattr(obs, "batch_size"):          # TensorDict
            obs = obs.get("critic", obs["policy"])
        elif isinstance(obs, dict):
            obs = obs.get("critic", obs.get("policy", next(iter(obs.values()))))
        return obs.repeat_interleave(self.n, dim=0)

    def _expand_extras(self, extras) -> dict:
        """shape (E, ...) 인 텐서를 (E×N, ...) 로 확장."""
        if not isinstance(extras, collections.abc.Mapping):
            return extras
        out = {}
        for k, v in extras.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == self._E:
                out[k] = v.repeat_interleave(self.n, dim=0)
            elif isinstance(v, collections.abc.Mapping):
                out[k] = {
                    kk: vv.repeat_interleave(self.n, dim=0)
                    if isinstance(vv, torch.Tensor) and vv.shape[0] == self._E
                    else vv
                    for kk, vv in v.items()
                }
            else:
                out[k] = v
        return out

    def __getattr__(self, name):
        return getattr(self._env, name)
