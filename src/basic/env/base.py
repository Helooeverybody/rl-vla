from abc import ABC, abstractmethod
import numpy as np 

class Env(ABC):
    """
    Base class for all env 

    obs_dim: the dimension of the observation space
    action_type: "discrete" or "continuous"
    n_actions: the number of actions (only for discrete action space)
    action_dim: the dimension of the action space (only for continuous action space)
    action_low: lower bound of the action space (only for continuous action space)
    action_high: upper bound of the action space (only for continuous action space)
    max_epsisode_steps: the maximum number of steps per episode
    """

    max_episode_steps: int
    obs_dim: int 
    action_type: str

    # Discrete env
    n_actions: int

    # Continuous env
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray

    def __init__(self):
        self.rng = np.random.default_rng()
        self._elapsed_steps = 0 

    def seed(self, seed: int | None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def reset(self, seed: int | None = None):
        """ Start a new episode and return the initial observation """

    @abstractmethod
    def step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        """ Apply one action and return ( obs, reward, terminated, truncated)"""

    
       


