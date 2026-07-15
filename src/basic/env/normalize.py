"""
Running observation normalisation.

NormalizeEnv : normalize the observations and/or actions of an environment.

obs_norm = clip( (obs - mean) / sqrt(var + eps), -clip, clip )


"""

import numpy as np 

class RunningMeanStd:
    """ Numerically stable running mean/variance over batches of vectors. """
    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype = np.float64)
        self.var = np.ones(dim, dtype = np.float64)
        self.count = 1e-4 
    def update(self, batch: np.ndarray) -> None:
        """
        This method uses Chan’s parallel Welford algorithm. It allows you to
        take the mean/variance of a new batch of data and accurately merge 
        it with the historical mean/variance, weighting them properly based 
        on the number of samples
        """
        batch = np.atleast_2d(batch)
        b_mean = batch.mean(axis = 0)
        b_var = batch.var(axis = 0)
        b_count = batch.shape[0]

        delta = b_mean - self.mean 
        total = self.count + b_count 

        self.mean = self.mean + delta * b_count / total 
        m2 = (self.var * self.count + b_var * b_count
              + delta ** 2 * self.count * b_count / total)
        self.var = m2 / total
        self.count = total

class NormalizeObservation:
    def __init__(self, env, clip: float = 10.0):
        self.env = env 
        self.clip = clip 
        self.training = True 
        self.rms = RunningMeanStd(env.obs_dim)
        self.is_vector = hasattr(env, "num_envs")

        # Mirror the env interface 
        # Discrete envs lack action_dim/action_low/high; continuous envs lack
        # n_actions. Default missing attributes to None instead of crashing.
        for attr in ("obs_dim", "action_type", "n_actions", "action_dim",
                     "action_low", "action_high", "max_episode_steps"):
            setattr(self, attr, getattr(env, attr, None))
        if self.is_vector:
            self.num_envs = env.num_envs
    
    def _normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.training and update:
            self.rms.update(obs)
        normed = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        return self._normalize(self.env.reset(seed = seed))

    def step(self, action):
        if self.is_vector:
            obs, rewards, terminated, truncated, final_obs = self.env.step(action)
            obs = self._normalize(obs, update = True)
            final_obs = self._normalize(final_obs, update = False)
            return obs, rewards, terminated, truncated, final_obs
        obs, rewards, terminated, truncated = self.env.step(action)
        obs = self._normalize(obs, update = True)
        return obs, rewards, terminated, truncated

    