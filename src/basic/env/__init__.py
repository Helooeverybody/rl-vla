from src.basic.env.base import Env
from src.basic.env.cartpole import CartPoleEnv
from src.basic.env.gridworld import GridWorldEnv
from src.basic.env.navigation import NavigationEnv
from src.basic.env.normalize import NormalizeObservation, RunningMeanStd
from src.basic.env.pendulum import PendulumEnv
from src.basic.env.reacher import ReacherEnv
from src.basic.env.vector import SyncVectorEnv

# The four continuous-control tasks form the benchmark suite: every algorithm
# in this repository (A2C, PPO, DDPG, TD3, SAC) can run on all of them, so
# they support like-for-like comparisons. GridWorld is discrete, so only the
# policy-gradient methods (A2C, PPO) apply to it.


ENVS = {
    "pendulum": PendulumEnv,      # swing-up: hard exploration, truncation only
    "cartpole": CartPoleEnv,      # balance: dense reward, early termination
    "reacher": ReacherEnv,        # reaching: dense, easy — the sanity check
    "navigation": NavigationEnv,  # traps + near-sparse reward: exploration test
    "gridworld": GridWorldEnv,    # discrete (A2C / PPO only)
}

CONTINUOUS_ENVS = ["pendulum", "cartpole", "reacher", "navigation"]


def make_env(name: str) -> Env:
    try:
        return ENVS[name]()
    except KeyError:
        raise ValueError(f"Unknown env '{name}'. Available: {sorted(ENVS)}")

__all__ = ["Env", "CartPoleEnv", "GridWorldEnv", "NavigationEnv", "NormalizeObservation", "RunningMeanStd", "PendulumEnv", "ReacherEnv", "SyncVectorEnv", "ENVS", "CONTINUOUS_ENVS", "make_env"]
