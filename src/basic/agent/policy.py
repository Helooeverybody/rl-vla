"""
Neural nets shared by all agents 

Categorical Policy:         pi(a|s) for discrete action spaces (A2C, PPO)
Gaussian Policy:            pi(a|s) for continuous action spaces (A2C, PPO)
Squashed Gaussian Policy:   pi(a|s) bounded stochastic policy for continuous action spaces (SAC)
Deterministic Policy:       a(s) for continuous action spaces (DDPG)
ValueNet:                   V(s) for value-based methods(A2C, PPO)
QNet:                       Q(s, a) for value-based methods(DDPG, SAC)

Stochastic policies:
    sample(obs) -> action, log_prob [acting]
    evaluate_actions(obs, action) -> log_prob, entropy [training]
"""

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical, Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0 

def mlp(sizes, activation = nn.Tanh, output_activation = nn.Identity) -> nn.Sequential:
    """Build an MLP, e.g mlp()"""

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) -2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    return nn.Sequential(*layers)

def orthogonal_init(net: nn.Module, output_gain: float) -> None:
    """Orthogonal weight initialization, the standard for on-policy RL"""

    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linears):
        gain = output_gain if i == len(linears) -1 else np.sqrt(2)
        nn.init.orthogonal_(layer.weight, gain = gain)
        nn.init.constant_(layer.bias, 0.0)




# ---------------
# On-policy RL
# ---------------


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden = (64,64)):
        super().__init__()
        self.logits_net = mlp([obs_dim, *hidden, n_actions])
        orthogonal_init(self.logits_net, output_gain = 0.01)
    
    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits = self.logits_net(obs))
    
    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action)
        
    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.distribution(obs)
        return dist.log_prob(action), dist.entropy()
    
    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.logits_net(obs).argmax(dim = -1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden = (64, 64)):
        super().__init__()
        self.mu_net = mlp([obs_dim, *hidden, action_dim])
        orthogonal_init(self.mu_net, output_gain = 0.01)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mu = self.mu_net(obs)
        std = torch.exp(torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mu, std)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(axis = -1) # sum over action dimensions for multi-dimensional actions
    
    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor):
        dist = self.distribution(obs)
        return dist.log_prob(action).sum(axis = -1), dist.entropy().sum(axis = -1)
    
    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu_net(obs)

#-----------
# Off-policy RL
#-----------

class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low: float, action_high: float, hidden = (256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, action_dim], activation = nn.ReLU, output_activation = nn.Tanh)
        low = torch.as_tensor(np.asarray(action_low), dtype = torch.float32)
        high = torch.as_tensor(np.asarray(action_high), dtype = torch.float32)
        self.register_buffer("scale", (high - low) /2.0)
        self.register_buffer("bias", (high + low) /2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.scale + self.bias
    
class SquashedGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high, hidden = (256, 256)):
        super().__init__()
        self.body = mlp([obs_dim, *hidden], activation = nn.ReLU, output_activation = nn.ReLU)
        self.mu_head = nn.Linear(hidden[-1], action_dim)
        self.log_std_head = nn.Linear(hidden[-1], action_dim)
        low = torch.as_tensor(np.asarray(action_low), dtype = torch.float32)
        high = torch.as_tensor(np.asarray(action_high), dtype = torch.float32)
        self.register_buffer("scale", (high - low) /2.0)
        self.register_buffer("bias", (high + low) /2.0)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.body(obs)
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.rsample()  # reparameterization trick
        squashed_action = torch.tanh(action) * self.scale + self.bias
        log_prob = dist.log_prob(action).sum(axis=-1) 
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)  # correction for tanh squashing
        return squashed_action, log_prob

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden = (64, 64)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, 1])
        orthogonal_init(self.net, output_gain = 1.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden = (256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + action_dim, *hidden, 1], activation = nn.ReLU)
        orthogonal_init(self.net, output_gain = 1.0)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim = -1)
        return self.net(x).squeeze(-1)