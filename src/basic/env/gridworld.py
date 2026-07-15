"""
GridWorld env: navigate a grid to reach a target, tiny discrete env

Description: 
The agent lives on a 5x5 grid:

        S . . . .        S = start (0, 0)
        . . . H .        G = goal  (+1 reward, episode terminates)
        . H . . .        H = hole  (-1 reward, episode terminates)
        . . . H .        . = free cell (-0.01 step penalty)
        . . . . G


Actions: 0 = up, 1 = down, 2 = left, 3 = right.
With probability `slip` the agent moves in a random direction instead of the
chosen one, which makes the env stochastic
Observation: a one-hot vector of length 25 marking the agent’s cell.

"""

import numpy as np 
from src.basic.env.base import Env

class GridWorldEnv(Env):
    action_type = "discrete"
    n_actions = 4
    max_episode_steps = 100 
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  

    def __init__(self, size: int = 5, slip: float = 0.0,
                 random_start: bool = True) -> None:
        super().__init__()
        self.size = size
        self.slip = slip
        self.random_start = random_start
        self.obs_dim = size * size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.holes = {(1, 3), (2, 1), (3, 3)}
        self.free_cells = [
            (r, c) for r in range(size) for c in range(size)
            if (r, c) not in self.holes and (r, c) != self.goal
        ]
        self.pos = self.start
    
    def _obs(self) -> np.ndarray:
        one_hot = np.zeros(self.obs_dim, dtype = np.float32)
        one_hot[self.pos[0] * self.size + self.pos[1]] = 1.0 
        return one_hot
    
    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed(seed)
        self._elapsed_steps = 0
        if self.random_start:
            self.pos = self.free_cells[self.rng.integers(len(self.free_cells))]
        else:
            self.pos = self.start
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool]:
        self._elapsed_steps += 1 

        # random action with probability slip
        if self.slip > 0 and self.rng.random() < self.slip:
            action = int(self.rng.integers(0, self.n_actions))

        dr, dc = self.MOVES[int(action)]
        row = min(max(self.pos[0] + dr, 0), self.size - 1)
        col = min(max(self.pos[1] + dc, 0), self.size - 1)
        self.pos = (row, col)

        if self.pos == self.goal:
            return self._obs(), 1.0, True, False
        if self.pos in self.holes:
            return self._obs(), -1.0, True, False
        truncated = self._elapsed_steps >= self.max_episode_steps
        return self._obs(), -0.01, False, truncated


        



