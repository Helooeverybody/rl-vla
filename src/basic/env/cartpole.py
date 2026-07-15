"""
CartPole with continuous action space: classic balancing task 
-------------------
Description:
A pole is hinged on a cart that slides along a rail. The agent pushed the cart
left or right and must keep the pole upright 

The action: continuous force [-1,1] scaled to += 10 N 
Observation: [cart position, cart velocity, pole angle, pole angular velocity]
Reward: +1 for every step the pole remains upright
Terminated: The pole falls over (|angle| > 12 degrees) or the cart goes out of bounds (position > 2.4)
truncated: The episode length exceeds 500 steps
"""


import numpy as np 
from src.basic.env.base import Env 

class CartPoleEnv(Env):
    action_type = "continuous"
    action_dim = 1 
    obs_dim = 4 
    max_episode_steps = 500 

    G = 9.8 
    M_CART = 1.0 
    M_POLE = 0.1 
    L = 0.5 
    DT = 0.02
    FORCE_MAG = 10.0 # mapping from [-1,1] to [-FORCE_MAG, FORCE_MAG]

    ANGLE_LIMIT = 12 * np.pi / 180 # 12 degree
    POSITION_LIMIT = 2.4 

    def __init__(self):
        super().__init__()
        self.action_low = np.array([-1.0], dtype=np.float32)
        self.action_high = np.array([1.0], dtype=np.float32)
        self.state = np.zeros(4, dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed(seed)
        self.state = self.rng.uniform(low = -0.05, high = 0.05, size = (4,), dtype=np.float32)
        self._elapsed_steps = 0
        return self.state

    def step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        self._elapsed_steps +=1
        force = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0) * self.FORCE_MAG)

        x, x_dot, theta, theta_dot = self.state 
        cos_th, sin_th = np.cos(theta), np.sin(theta)
        total_mass = self.M_CART + self.M_POLE
        pole_mass_length = self.M_POLE * self.L


        temp = (force + pole_mass_length * theta_dot ** 2 * sin_th) / total_mass
        theta_acc = (self.GRAVITY * sin_th - cos_th * temp) / (
            self.LENGTH * (4.0 / 3.0 - self.MASS_POLE * cos_th ** 2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_th / total_mass

        # Semi-implicit Euler integration.
        x_dot += self.DT * x_acc
        x += self.DT * x_dot
        theta_dot += self.DT * theta_acc
        theta += self.DT * theta_dot
        self.state = np.array([x, x_dot, theta, theta_dot])

        terminated = bool(abs(x) > self.POSITION_LIMIT
                          or abs(theta) > self.ANGLE_LIMIT)
        truncated = self._elapsed_steps >= self.max_episode_steps
        return self.state.astype(np.float32), 1.0, terminated, truncated


