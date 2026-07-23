"""
Shared machinery for the on-policy agents(A2C, PPO)

Both algorithms follow the same outer loop -- collect a rollout from N 
parallel envs, estimates advantages with GAE, update, reapeat and differ only 
in how they update
"""

from collections import deque 
import numpy as np 
import torch 
from src.baisc.env import NormalizeObservation, SyncVectorEnv
from src.basic.agent.buffer import RolloutBuffer 
from src.basic.agent.policy import CategoricalPolicy, GaussianPolicy, ValueNet 
from src.basic.agent.utils import evaluate, set_seed, to_tensor 
from src.common.logging import Logger 

class OnPolicyAgent:
    def __init__(self, 
                 env_fn, 
                 num_envs: int = 4, 
                 n_steps: int = 128, 
                 gamma: float = 0.99, 
                 gae_lambda: float = 0.96,
                 lr: float = 3e-4, 
                 vf_coef: float = 0.5, 
                 ent_coef: float = 0.5, 
                 max_grad_norm: float = 0.5,
                 hidden = (64,64), 
                 normalize_obs: bool = False, 
                 seed: int = 0, 
                 run_name: str = "on_policy",
                 log_dir: str |None = None, 
                 wandb_config: dict |None = None):
        
        self._seed(seed)
        self.gamma = gamma 
        self.lr = lr 
        self.vf_coef = vf_coef 
        self.ent_coef = ent_coef 
        self.max_grad_norm = max_grad_norm 
        self.rng = np.random.default_rng(seed)
        self.logger = Logger(run_name, log_dir, wandb_config)

        self.venv = SyncVectorEnv([env_fn for _ in range(num_envs)])

        self.eval_env = env_fn()
        self.normalizer = None 
        if normalize_obs: 
            self.venv = NormalizeObservation(self.venv)
            self.eval_env = NormalizeObservation(self.venv)
            self.eval_env.rms  = self.venv.rms  # eval sees the SAME statistics
            self.eval_env.training = False    # never update them 
            self.normalizer =  self.venv
        self.discrete = self.venv.action_type == "discrete"

        if self.discrete: 
            self.policy = CategoricalPolicy(self.venv.obs_dim, self.venv.n_actions, hidden)
            action_shape: tuple = ()
        else:
            self.policy = GaussianPolicy(self.venv.obs_dim, self.venv.action_dim, hidden)
            action_shape = (self.venv.action_dim,)
        self.value_fn = ValueNet(self.venv.obs_dim, hidden)
        self.parameters = (list(self.policy.parameters()) + list(self.value_fn.parameters()))
        self.optimizer = torch.optim.Adam(self.parameters, lr = lr)
        self.buffer = RolloutBuffer(n_steps, num_envs, self.venv.obs_dim, action_shape, gamma, gae_lambda)

        self._obs = self.venv.reset(seed = seed)
        self._ep_returns = np.zeros(num_envs)
        self._recent_returns : deque = deque(maxlen = 50)
        self.global_step = 0 
    def act(self, obs: np.adarray):
        """Deterministic action for evaluation(mode of the policy)"""

        action = self.policy.act_deterministic(to_tensor(obs))
        if self.discrete: 
            return int(action.item())
        return np.clip(action.numpy()[0], self.venv.action_low, self.venv.action_high)
    def _env_actions(self, actions: torch.Tensor) -> np.ndarray:
        """Batch of policy samples -> batch of env actions"""

        if self.discrete:
            return actions.numpy()
        return np.clip(actions.numpy(), self.venv.action_low, self.venv.action_high)
    def collect_rollout(self) -> None:
        """Run the current policy for n_steps in all N envs at once"""
        for _ in range(self.buffer.n_steps):
            obs_t = to_tensor(self._obs)
            actions,log_probs = self.policy.sample(obs_t)
            with torch.no_grad():
                values = self.value_fn(obs_t).numpy()
            next_obs, rewards, terminated, truncated, final_obs = self.venv.step(self._env_actions(actions))
            self._ep_returns += rewards

            train_rewards = rewards.copy()
            if truncated.any():
                with torch.no_grad():
                    final_values = self.value_fn(
                        to_tensor(final_obs[truncated])
                    ).numpy()
                train_rewards[truncated] += self.gamma * final_values 
            dones = terminated | truncated 
            self.buffer.add(self._obs, actions.numpy(), train_rewards, values, log_probs.numpy(), dones.astype(np.float32))
            for i in np.flatnonzero(dones):
                self._recent_returns.append(self._ep_returns[i])
                self._ep_returns[i] = 0.0
            self._obs = next_obs 
            self.global_step += self.venv.num_envs

        with torch.no_grad():
            last_values = self.value_fn(to_tensor(self._obs)).numpy()
        self.buffer.compute_gae(last_values)

    def _update(self) -> dict[str, float]:
        raise NotImplementedError

    def _on_iteration(self, iteration: int, total_iterations: int) -> None:
        """Hook called before each iteration(PPO uses it to anneal the lr)"""

    def train(self, total_steps: int, log_interval: int = 10) -> None:
        n_iterations = max(total_steps // self.buffer.size, 1)
        for it in range(1, n_iterations + 1):
            self._on_iteration(it, n_iterations)
            self.collect_rollout()
            stats = self._update()
            if it % log_interval == 0 or it == n_iterations:
                mean_ret = (float(np.mean(self._recent_returns)) if self._recent_returns else float("nan"))
                self.logger.log(self.global_step, mean_ep_return = mean_ret, **stats)
    def evaluate(self, episodes:int = 10) -> float:
        return evaluate(self.eval_env, self.act, episodes)

    def save(self, path: str):
        state = {
            "policy": self.policy.state_dict(),
            "value_fn": self.value_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.normalizer is not None: 
            state["obs_rms"] =self.normalizer.state_dict()
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, weights_only = False)
        self.policy.load_state_dict(state["policy"])
        self.value_fn.load_state_dict(state["value_fn"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.normalizer is not None and "obs_rms" in state:
            self.normalizer.load_state_dict(state["obs_rms"])


