"""
SyncVectorEnv : run N copies of an env in lockstep to decorrelation.

On-policy spend most of their time collecting data. Stepping N independent
environments copies per iteration gives N transitions per policy forward pass

"""

from collections.abc import Callable 
import numpy as np 
from src.basic.env.base import Env 

class SyncVectorEnv:
    def __init__(self, env_fns: list[Callable[[], Env]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

        # mirror the single-env interface attributes 
        proto = self.envs[0]
        self.obs_dim = proto.obs_dim 

        self.action_type = proto.action_type 
        self.n_actions = proto.n_actions if proto.action_type == "discrete" else None
        self.action_dim = proto.action_dim if proto.action_type == "continuous" else None
        self.action_low = proto.action_low if proto.action_type == "continuous" else None
        self.action_high = proto.action_high if proto.action_type == "continuous" else None
        self.max_episode_steps = proto.max_episode_steps

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs = [
            env.reset(seed = None if seed is None else seed + i)
            for i, env in enumerate(self.envs)
        ]
        return np.stack(obs).astype(np.float32)
    def step(self, actions: np.ndarray):
        obs = np.zeros((self.num_envs, self.obs_dim), dtype = np.float32)
        final_obs = np.zeros((self.num_envs, self.obs_dim), dtype = np.float32)
        rewards = np.zeros((self.num_envs,), dtype = np.float32)
        terminateds = np.zeros((self.num_envs,), dtype = np.bool_)
        truncateds = np.zeros((self.num_envs,), dtype = np.bool_)

        for i, env in enumerate(self.envs):
            obs[i], rewards[i], terminateds[i], truncateds[i] = env.step(actions[i])
            if terminateds[i] or truncateds[i]:
                final_obs[i] = obs[i].copy()
                obs[i] = env.reset(seed = None if seed is None else seed + i)
        return obs, rewards, terminateds, truncateds, final_obs

    