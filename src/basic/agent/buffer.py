"""
RolloutBuffer(on-policy): Stores the latest trajectory segment only, collected from N parallel envs
(shape: n_steps x num_envs). After one update the data is stale and thrown away
It compute advantage with GAE.

ReplayBuffer(off-policy): A large ring buffer of (s,a,r,s',done) transitions collected by any past policy
. Off-policy algo can learn from old data. Old entries are overwritten when full.
"""

import numpy as np 
import torch 

class RolloutBuffer:
    """Fixed-size on-policy buffer, time-major (n_steps, num_envs), with GAE-lambda 
    advantage estimation"""

    def __init__(self, n_steps: int, num_envs: int, obs_dim: int,
                action_shape: tuple, gamma: float = 0.99, lam: float = 0.95):
        self.obs = np.zeros((n_steps, num_envs, obs_dim), dtype= np.float32)
        self.actions = np.zeros((n_steps, num_envs, *action_shape), dtype = np.float32)
        self.rewards = np.zeros((n_steps, num_envs), dtype = np.float32)
        self.values = np.zeros((n_steps, num_envs), dtype = np.float32)
        self.log_probs = np.zeros((n_steps, num_envs), dtype = np.float32)
        self.dones = np.zeros((n_steps, num_envs), dtype = np.float32)
        self.advantages = np.zeros((n_steps, num_envs), dtype = np.float32)
        self.returns = np.zeros((n_steps, num_envs), dtype = np.float32)
        self.n_steps = n_steps 
        self.num_envs = num_envs 
        self.gamma, self.lam = gamma, lam 
        self.t = 0 
        self._computed = False 

    @property 
    def size(self) -> int: 
        return self.n_steps * self.num_envs 

    @property 
    def full(self) -> bool:
        return self.t == self.n_steps 

    def add(self, obs, actions, rewards, values, log_probs, dones) -> None:
        assert self.t < self.n_steps, "RolloutBuffer overflow"
        self.obs[self.t]  = obs 
        self.actions[self.t] = actions.reshape(self.actions[self.t].shape)
        self.rewards[self.t] = rewards 
        self.values[self.t] = values 
        self.log_probs[self.t] = log_probs 
        self.dones[self.t] = dones 
        self.t += 1

    def compute_gae(self, last_values: np.ndarray) -> None:
        """
        TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        GAE:           A_t = delta_t + gamma * lambda * (1-done_t) * A_{t+1}
        R_t = A_t + V(s_t)

        (1 - done_t) cuts both recursions at episode boundaries: because the
        vector env auto-resets, values[t+1] on a boundary belongs to the NEXT
        episode and must not leak into this one. ‘last_values‘ is V of the
        observation after the final stored step, used to bootstrap rollouts
        that stop mid-episode.

        """

        assert self.full, "compute_gae() expects a complete rollout"
        gae = np.zeros(self.num_envs, dtype = np.float32)
        for t in reversed(range(self.n_steps)):
            next_values = self.values[t+1] if t < self.n_steps - 1 else last_values 
            not_done = 1.0 - self.done[t]
            delta = (self.rewards[t] + self.gamma * next_values * not_done - self.values[t])
            gae = delta + self.gamma * self.lam * not_done * gae 
            self.advantages[t] = gae 

        self.returns = self.advantages + self.values 
        self._computed = True 

    def get(self) -> dict[str, torch.Tensor]:
        assert self.full and self._computed 
        n = self.size 
        data = {
            "obs": torch.as_tensor(self.obs.reshape(n, -1)),
            "actions": torch.as_tensor(
                self.actions.reshape(n, *self.actions.shape[2:])
            ), 
            "log_probs": torch.as_tensor(self.log_probs.reshape(n)),
            "values": torch.as_tensor(self.values.reshape(n)),
            "advantages": torch.as_tensor(self.advantages.reshape(n)), 
            "returns": torch.as_tensor(self.returns.reshape(n)), 

        }
        self.t = 0 
        self._computed = False 
        return data 


class ReplayBuffer:
    """
    Uniform-sampling ring buffer for off-policy algorithms 
    "done" stores *termination only*. On truncation(time limit) the caller stores done = False so 
    the target r + gamma * Q(s', a') still bootstraps-- the state was not really ternimal, jus stop looking
    """

    def __init__(self, size: int, obs_dim: int, action_dim: int):
        self.obs = np.zeros((size, obs_dim), dtype = np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype = np.float32)
        self.actions = np.zeros((size, action_dim), dtype = np.float32)
        self.rewards = np.zeros(size, dtype = np.float32)
        self.dones = np.zeros(size, dtype = np.float32)
        self.max_size = size 
        self.ptr = 0 
        self.count = 0 

    def add(self, obs, action, reward, next_obs, terminated: bool) -> None:
        self.obs[self.ptr] = obs 
        self.actions[self.ptr] = action 
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs 
        self.dones[self.ptr] = float(terminated)
        self.ptr = (self.ptr + 1) % self.max_size 
        self.count = min(self.count + 1, self.max_size)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        assert self.count > 0 , (
            "ReplayBuffer is empty -- call add() before sample(). The training "
            "loop guards this with ‘update_after‘, so hitting it means an "
            "agent’s _update() was called too early."
        )
        idx = rng.integers(self.count, size = batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "actions": torch.as_tensor(self.actions[idx]),
            "rewards": torch.as_tensor(self.rewards[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "dones": torch.as_tensor(self.dones[idx]),
        }
    def __len__(self) -> int: 
        return self.count 
    


