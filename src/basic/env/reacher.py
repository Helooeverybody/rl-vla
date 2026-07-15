"""
Reacher env: move a point to a randomly placed target. 

Description:
The easiest task in the suite, and that is exactly its job: a dense, smooth,
short-horizon problem. When a new implementation is broken, Reacher tells you 
immediately; when an algorithm struggle on Pendulum but sails through Reacher, 
we know the problem is exploration, not learning rule.

Observation (6): agent position (2), target position (2), and the vector
                 from agent to target (2). 
Action (2):      a displacement (dx, dy) in [-1, 1]^2, scaled by max_step.
Reward:          -distance to target at every step, plus a + 1 bonus on arrival.
Terminated:      the agent is within `reach_radius` of the target.
Truncated:       after 100 steps.
"""

import numpy as np 
from src.basic.env.base import Env 

class ReacherEnv(Env):
    action_type = "continuous"
    action_dim = 2 
    obs_dim = 6 
    max_episode_steps = 100 

    max_step = 0.08 
    reach_radius = 0.05 

    def __init__(self):
        super().__init__()
        self.action_low = np.array([-1.0, -1.0], dtype=np.float32)
        self.action_high = np.array([1.0, 1.0], dtype=np.float32)
        self.agent = np.zeros(2)
        self.target = np.zeros(2)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.agent, self.target, self.target - self.agent]).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed(seed)
        self.agent = self.rng.uniform(0.1, 0.9, size=2)
        self.target = self.rng.uniform(0.1, 0.9, size=2)
        self._elapsed_steps = 0
        return self._obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        self._elapsed_steps +=1 

        delta = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0) * self.max_step
        self.agent = np.clip(self.agent + delta, 0.0, 1.0)
        distance = np.linalg.norm(self.agent - self.target)  

        if distance < self.reach_radius:
            return self._obs(), 1.0, True, False
            
        truncated = self._elapsed_steps >= self.max_episode_steps
        return self._obs(), -distance, False, truncated
    