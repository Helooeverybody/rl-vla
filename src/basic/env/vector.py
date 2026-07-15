"""
SyncVectorEnv : run N copies of an env in lockstep to decorrelation.

In Deep RL, data collection is the bottleneck.

Neural Network Inefficiency: Neural networks are designed to process data 
in batches (e.g., 32 or 64 images at once) using parallel GPU computation. 
If an agent plays only one game at a time, it sends inputs to the neural network 
one by one (batch size of 1), which is incredibly slow.

Correlated Data: If you take 32 consecutive frames from a single game of 
Mario, those frames look almost identical. Training a neural network on 
highly correlated data makes learning unstable.

The Solution: We run N completely independent copies of the game at the exact 
same time. The neural network predicts N actions in a single forward pass,
 and the environments process those N actions simultaneously. This speeds up data collection and guarantees that the batch of data is diverse (decorrelated).
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
            o, rewards[i], terminateds[i], truncateds[i] = env.step(actions[i])
            if terminateds[i] or truncateds[i]:
                final_obs[i] = o
                o = env.reset()
            obs[i] = o

        return obs, rewards, terminateds, truncateds, final_obs

    