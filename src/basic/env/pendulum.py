"""

Classic continuous control environment: Pendulum-Swingup
-------------
Description: 
The pendulum environment consists of a pendulum that starts in 
a random position and the goal is to swing it up so it stays upright. 
The state consists of the cosine and sine of the angle, as well 
as the angular velocity. The action is a single scalar representing the 
torque applied to the pendulum. The reward is based on how close the pendulum 
is to being upright and how much torque is used.

State(internal): angle theta (0 = upright) and angular velocity theta_dot
State(observation): [cos(theta), sin(theta), theta_dot]
                    (cos/sin insteads to avoid discontinuity at the angle 
                    wraparound)
Action: torque applied to the pendulum (scalar)
Reward: - (theta^2 + 0.1*theta_dot^2 + 0.001*action^2)
        (always < 0, the closer to 0 the better)
"""
import numpy as np 
from src.basic.env.base import Env

def angle_normalize(theta: float) -> float:
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

class PendulumEnv(Env):
    action_type = "continuous"
    action_dim = 1 
    obs_dim = 3 
    max_episode_steps = 200

    MAX_SPEED = 8.0 
    MAX_TORQUE = 2.0
    DT = 0.05    # integration time step
    G = 10.0     # gravity constant
    M = 1.0      # mass of the pendulum
    L = 1.0      # length of the pendulum

    def __init__(self):
        super().__init__()
        self.action_low = np.array([-self.MAX_TORQUE], dtype=np.float32)
        self.action_high = np.array([self.MAX_TORQUE], dtype=np.float32)
        self.theta = 0.0 
        self.theta_dot = 0.0 

    def _obs(self) -> np.ndarray:
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot], dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seed(seed)
        self.theta = self.rng.uniform(low = -np.pi, high = np.pi)
        self.theta_dot = self.rng.uniform(low = -1.0, high = 1.0)
        self._elapsed_steps = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool]:
        self._elapsed_steps += 1
        torque = float(np.clip(np.asarray(action).ravel()[0], - self.MAX_TORQUE, self.MAX_TORQUE))
        
        theta = angle_normalize(self.theta)
        reward = -(theta ** 2 + 0.1 * self.theta_dot ** 2 + 0.001 * (torque ** 2))

        acc = 3 * self.G / (2 * self.L) * np.sin(theta) + 3.0 / (self.M * self.L ** 2) * torque
        self.theta_dot = np.clip(self.theta_dot + acc * self.DT, -self.MAX_SPEED, self.MAX_SPEED)
        self.theta = self.theta + self.theta_dot * self.DT
        
        truncated = self._elapsed_steps >= self.max_episode_steps
        return self._obs(), reward, False, truncated

