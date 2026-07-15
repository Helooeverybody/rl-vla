"""
Navigation: GridWorld with continuous actions version
The agent moves by continuous displacements(dx, dy)

Observation(4): (x, y, dx, dy) where (x, y) is the agent's position 
                and (dx, dy) is the displacement vector.
Action(2):      (dx, dy) - the continuous displacement vector.
Reward:         +1 if reaching goal, -1 if falling into a hole, -0.01 step 
                penalty otherwise.
Truncated:      after 100 steps

As in GridWorld, episodes start from a random free
19 position (exploring starts); with a fixed start, the agent learns 
to avoid 20 the hole region and then never finds the goal at all
"""

import numyp as np 
from src.basic.env.base import Env

class Navigation(Env):
    action_type = "continuous"
    action_dim = 2 
    obs_dim = 4 
    max_episode_steps = 100 

    max_step = 0.08 
    goal_radius = 0.08 
    hole_radius = 0.10

    def __init__(self, random_start: bool = True):
        super().__init__()
        self.random_start = random_start
        self.action_low = np.array([-1.0, -1.0], dtype = np.float32)
        self.action_high = np.array([1.0, 1.0], dtype = np.float32)
        self.start = np.array([0.1, 0.1], dtype = np.float32)
        self.goal = np.array([0.9, 0.9], dtype = np.float32)
        self.holes = np.array([[0.5, 0.5], [0.3, 0.7], [0.65, 0.35]], dtype = np.float32)
        self.agent = self.start.copy()

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.agent, self.goal - self.agent], axis = 0).astype(np.float32)
    
    def _far_from_hazards(self, point: np.ndarray) -> bool:
        for hole in self.holes:
            if np.linalg.norm(point - hole) < self.hole_radius:
                return False
        if np.linalg.norm(point - self.goal) < self.goal_radius:
            return False
        return True
    
    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed(seed)
        self._elapsed_steps = 0
        if self.random_start:
            while True:
                candidate = self.rng.uniform(0.0, 1.0, size = 2).astype(np.float32)
                if self._far_from_hazards(candidate):
                    self.agent = candidate
                    break
        else:
            self.agent = self.start.copy()
        return self._obs()
    def step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        self._elapsed_steps += 1
        delta = np.clip(np.asarray(action, dtype = np.float64).ravel(), -1.0, 1.0) * self.max_step
        self.agent = np.clip(self.agent + delta, 0.0, 1.0).astype(np.float32)
        if np.linalg.norm(self.agent - self.goal) < self.goal_radius:
            return self._obs(), 1.0, True, False
            
        for hole in self.holes:
            if np.linalg.norm(self.agent - hole) < self.hole_radius:
                return self._obs(), -1.0, True, False

        truncated = self._elapsed_steps >= self.max_episode_steps
        return self._obs(), -0.01, False, truncated



