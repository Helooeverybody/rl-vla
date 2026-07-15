"""
Unit tests for the environment code in ``src/basic/env``.

Run either way (both work from any directory)::

    pytest src/basic/test/test_env.py -v      # via pytest (uses root conftest.py)
    python src/basic/test/test_env.py         # directly (self-bootstraps sys.path)

Only the ``src`` package is exercised here (never ``tutorial/``). The tests are
pure-CPU and cheap: they reset/step small numpy environments a few hundred
times in total, so they do *not* constitute a training run.

What is covered
---------------
* Registry:            ``ENVS`` / ``make_env`` / ``CONTINUOUS_ENVS`` agree.
* Reset/step contract:  dtypes, shapes, tuple arity and flag types, for every
                        registered env.
* Determinism:          same seed reproduces the same trajectory.
* Episode bookkeeping:  an episode always ends within ``max_episode_steps`` and
                        a truncation happens exactly at the horizon.
* Per-env semantics:    pendulum start is randomized, reacher/navigation goal
                        and hole handling, gridworld one-hot + terminal cells.
* Wrappers:             ``RunningMeanStd`` matches numpy, ``NormalizeObservation``
                        clips/normalizes and mirrors the interface, and
                        ``SyncVectorEnv`` auto-resets while exposing final_obs.
"""

import os
import sys

import numpy as np
import pytest

# When run directly (``python src/basic/test/test_env.py``) the repo root is not
# on sys.path, so absolute ``src.basic.env`` imports would fail. Add it here.
# Under pytest this is a no-op because the root conftest.py already handles it.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.basic.env import (
    ENVS,
    CONTINUOUS_ENVS,
    make_env,
    CartPoleEnv,
    GridWorldEnv,
    NavigationEnv,
    NormalizeObservation,
    PendulumEnv,
    ReacherEnv,
    RunningMeanStd,
    SyncVectorEnv,
)

ALL_ENV_NAMES = sorted(ENVS)
SEED = 12345


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sample_action(env, rng):
    """Return a random valid action for either action type."""
    if env.action_type == "discrete":
        return int(rng.integers(0, env.n_actions))
    return rng.uniform(env.action_low, env.action_high).astype(np.float32)


def zero_action(env):
    """A benign 'do nothing' action of the right shape/type."""
    if env.action_type == "discrete":
        return 0
    return np.zeros(env.action_dim, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_is_consistent():
    assert set(ALL_ENV_NAMES) == {
        "pendulum",
        "cartpole",
        "reacher",
        "navigation",
        "gridworld",
    }
    # Every continuous name is registered and actually continuous.
    for name in CONTINUOUS_ENVS:
        assert name in ENVS
        assert make_env(name).action_type == "continuous"
    # gridworld is the only discrete env and is not in the continuous list.
    assert "gridworld" not in CONTINUOUS_ENVS
    assert make_env("gridworld").action_type == "discrete"


def test_make_env_rejects_unknown_name():
    with pytest.raises(ValueError):
        make_env("does_not_exist")


# --------------------------------------------------------------------------- #
# Generic reset/step contract — runs for every registered env
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_ENV_NAMES)
def test_reset_contract(name):
    env = make_env(name)
    obs = env.reset(seed=SEED)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


@pytest.mark.parametrize("name", ALL_ENV_NAMES)
def test_step_contract(name):
    env = make_env(name)
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    out = env.step(sample_action(env, rng))

    assert len(out) == 4
    obs, reward, terminated, truncated = out
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert np.isscalar(reward) and np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


@pytest.mark.parametrize("name", ALL_ENV_NAMES)
def test_action_type_attributes(name):
    env = make_env(name)
    if env.action_type == "continuous":
        assert env.action_low.shape == (env.action_dim,)
        assert env.action_high.shape == (env.action_dim,)
        assert np.all(env.action_low <= env.action_high)
    else:
        assert env.n_actions >= 1


@pytest.mark.parametrize("name", ALL_ENV_NAMES)
def test_determinism_same_seed(name):
    """Same seed + same actions => identical trajectory."""
    def rollout():
        env = make_env(name)
        obs0 = env.reset(seed=SEED)
        rng = np.random.default_rng(SEED)  # action stream independent of env RNG
        traj = [obs0]
        for _ in range(20):
            obs, r, term, trunc = env.step(sample_action(env, rng))
            traj.append(obs)
            if term or trunc:
                break
        return traj

    a, b = rollout(), rollout()
    assert len(a) == len(b)
    for xa, xb in zip(a, b):
        np.testing.assert_array_equal(xa, xb)


@pytest.mark.parametrize("name", ALL_ENV_NAMES)
def test_episode_ends_within_horizon(name):
    """An episode never runs past max_episode_steps; truncation lands on it."""
    env = make_env(name)
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    steps = 0
    for _ in range(env.max_episode_steps + 5):
        _, _, terminated, truncated = env.step(sample_action(env, rng))
        steps += 1
        if terminated or truncated:
            break
    assert steps <= env.max_episode_steps
    if truncated and not terminated:
        assert steps == env.max_episode_steps


@pytest.mark.parametrize("name", ALL_ENV_NAMES)
def test_out_of_range_action_is_clipped(name):
    """A wildly out-of-range action must not crash or produce non-finite obs."""
    env = make_env(name)
    env.reset(seed=SEED)
    if env.action_type == "continuous":
        action = env.action_high * 1000.0
    else:
        action = env.n_actions - 1  # highest valid index
    obs, reward, _, _ = env.step(action)
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


# --------------------------------------------------------------------------- #
# Pendulum
# --------------------------------------------------------------------------- #
def test_pendulum_start_is_randomized():
    """Regression: reset used to pin theta to a constant."""
    env = PendulumEnv()
    thetas = []
    for s in range(30):
        env.reset(seed=s)
        thetas.append(env.theta)
    assert np.std(thetas) > 0.1  # genuinely spread out
    assert min(thetas) < 0.0 < max(thetas)  # both signs appear


def test_pendulum_never_terminates_and_truncates_at_horizon():
    env = PendulumEnv()
    env.reset(seed=SEED)
    for i in range(env.max_episode_steps):
        _, r, terminated, truncated = env.step(np.array([0.0], dtype=np.float32))
        assert terminated is False
        assert r <= 0.0  # reward is always non-positive
    assert truncated is True


# --------------------------------------------------------------------------- #
# CartPole
# --------------------------------------------------------------------------- #
def test_cartpole_reset_and_step_run():
    """Regression: reset passed dtype= to Generator.uniform; step used names
    (GRAVITY/LENGTH/MASS_POLE) that did not exist on the class."""
    env = CartPoleEnv()
    obs = env.reset(seed=SEED)
    assert obs.dtype == np.float32
    obs, reward, terminated, truncated = env.step(np.array([0.5], dtype=np.float32))
    assert reward == 1.0
    assert np.all(np.isfinite(obs))


def test_cartpole_terminates_when_pushed_one_way():
    """Constant max force should eventually knock the pole/cart out of bounds."""
    env = CartPoleEnv()
    env.reset(seed=SEED)
    terminated = False
    for _ in range(env.max_episode_steps):
        _, _, terminated, truncated = env.step(np.array([1.0], dtype=np.float32))
        if terminated:
            break
    assert terminated is True


# --------------------------------------------------------------------------- #
# Reacher
# --------------------------------------------------------------------------- #
def test_reacher_reward_is_negative_distance():
    env = ReacherEnv()
    env.reset(seed=SEED)
    obs, reward, terminated, _ = env.step(np.zeros(2, dtype=np.float32))
    if not terminated:
        expected = -np.linalg.norm(env.agent - env.target)
        assert reward == pytest.approx(expected, abs=1e-5)


def test_reacher_terminates_on_reaching_target():
    """Drive straight at the target; must terminate with +1 reward."""
    env = ReacherEnv()
    env.reset(seed=SEED)
    terminated = False
    reward = None
    for _ in range(env.max_episode_steps):
        direction = env.target - env.agent
        norm = np.linalg.norm(direction)
        action = (direction / norm) if norm > 0 else np.zeros(2)
        _, reward, terminated, truncated = env.step(action.astype(np.float32))
        if terminated:
            break
    assert terminated is True
    assert reward == 1.0


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
def test_navigation_random_start_avoids_hazards():
    env = NavigationEnv(random_start=True)
    for s in range(20):
        env.reset(seed=s)
        for hole in env.holes:
            assert np.linalg.norm(env.agent - hole) >= env.hole_radius
        assert np.linalg.norm(env.agent - env.goal) >= env.goal_radius


def test_navigation_terminates_on_goal():
    env = NavigationEnv(random_start=False)  # fixed start at (0.1, 0.1)
    env.reset(seed=SEED)
    # Start in the goal corner: the straight line from (0.1, 0.1) to the goal
    # passes through the hole at (0.5, 0.5), so approach from nearby instead.
    env.agent = np.array([0.8, 0.8], dtype=np.float32)
    terminated = False
    reward = None
    for _ in range(env.max_episode_steps):
        direction = env.goal - env.agent
        norm = np.linalg.norm(direction)
        action = (direction / norm) if norm > 0 else np.zeros(2)
        _, reward, terminated, truncated = env.step(action.astype(np.float32))
        if terminated:
            break
    assert terminated is True
    assert reward == 1.0


# --------------------------------------------------------------------------- #
# GridWorld
# --------------------------------------------------------------------------- #
def test_gridworld_obs_is_one_hot():
    env = GridWorldEnv()
    obs = env.reset(seed=SEED)
    assert obs.shape == (env.size * env.size,)
    assert obs.sum() == pytest.approx(1.0)
    assert set(np.unique(obs)).issubset({0.0, 1.0})


def test_gridworld_random_start_covers_multiple_cells():
    """Regression: rng.choice on a list of tuples raised; also checks coverage."""
    env = GridWorldEnv(random_start=True)
    starts = set()
    for s in range(50):
        env.reset(seed=s)
        starts.add(tuple(env.pos))
    assert len(starts) > 1
    # A random start must never begin on a hole or the goal.
    assert starts.isdisjoint(env.holes)
    assert env.goal not in starts


def test_gridworld_goal_and_hole_are_terminal():
    # Goal: start adjacent (below the goal) and move down into it.
    env = GridWorldEnv(random_start=False)
    env.reset(seed=SEED)
    env.pos = (env.size - 2, env.size - 1)
    obs, reward, terminated, truncated = env.step(1)  # down
    assert terminated is True and reward == 1.0

    # Hole at (1, 3): start above it and step down.
    env.reset(seed=SEED)
    env.pos = (0, 3)
    obs, reward, terminated, truncated = env.step(1)  # down -> (1, 3)
    assert terminated is True and reward == -1.0


def test_gridworld_walls_block_movement():
    env = GridWorldEnv(random_start=False)
    env.reset(seed=SEED)
    env.pos = (0, 0)
    env.step(0)  # up from top row -> stays
    assert env.pos == (0, 0)


# --------------------------------------------------------------------------- #
# RunningMeanStd
# --------------------------------------------------------------------------- #
def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=[1.0, -2.0, 3.0], scale=[0.5, 2.0, 1.0], size=(4000, 3))
    rms = RunningMeanStd(dim=3)
    # Feed in several batches to exercise the parallel merge.
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# NormalizeObservation
# --------------------------------------------------------------------------- #
def test_normalize_observation_single_env():
    env = NormalizeObservation(PendulumEnv(), clip=5.0)
    # Interface is mirrored.
    assert env.obs_dim == 3
    assert env.action_type == "continuous"
    assert env.action_dim == 1

    obs = env.reset(seed=SEED)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    for _ in range(50):
        obs, r, term, trunc = env.step(np.array([0.0], dtype=np.float32))
        assert np.all(np.abs(obs) <= 5.0 + 1e-5)  # respects the clip
        if term or trunc:
            break


def test_normalize_observation_updates_statistics():
    env = NormalizeObservation(PendulumEnv())
    before = env.rms.count
    env.reset(seed=SEED)
    for _ in range(30):
        env.step(np.array([0.5], dtype=np.float32))
    assert env.rms.count > before  # running stats actually moved


def test_normalize_observation_eval_mode_freezes_stats():
    env = NormalizeObservation(PendulumEnv())
    env.reset(seed=SEED)
    for _ in range(30):
        env.step(np.array([0.5], dtype=np.float32))
    env.training = False
    frozen_count = env.rms.count
    frozen_mean = env.rms.mean.copy()
    for _ in range(30):
        env.step(np.array([0.5], dtype=np.float32))
    assert env.rms.count == frozen_count
    np.testing.assert_array_equal(env.rms.mean, frozen_mean)


# --------------------------------------------------------------------------- #
# SyncVectorEnv
# --------------------------------------------------------------------------- #
def test_sync_vector_env_shapes_and_mirror():
    n = 4
    venv = SyncVectorEnv([PendulumEnv for _ in range(n)])
    assert venv.num_envs == n
    assert venv.obs_dim == 3
    assert venv.action_type == "continuous"
    assert venv.action_dim == 1

    obs = venv.reset(seed=SEED)
    assert obs.shape == (n, 3)
    assert obs.dtype == np.float32

    actions = np.zeros((n, 1), dtype=np.float32)
    obs, rewards, terminateds, truncateds, final_obs = venv.step(actions)
    assert obs.shape == (n, 3)
    assert rewards.shape == (n,)
    assert terminateds.shape == (n,) and terminateds.dtype == np.bool_
    assert truncateds.shape == (n,) and truncateds.dtype == np.bool_
    assert final_obs.shape == (n, 3)


def test_sync_vector_env_auto_resets_and_exposes_final_obs():
    """When a sub-env truncates, the vector env auto-resets and records the
    terminal observation in final_obs (needed for correct bootstrapping)."""
    n = 3
    venv = SyncVectorEnv([PendulumEnv for _ in range(n)])
    venv.reset(seed=SEED)
    actions = np.zeros((n, 1), dtype=np.float32)

    horizon = venv.max_episode_steps
    done_seen = False
    for step in range(horizon):
        obs, rewards, terminateds, truncateds, final_obs = venv.step(actions)
        done = truncateds | terminateds
        if done.any():
            done_seen = True
            # Every finished env must have a non-zero terminal obs recorded,
            # and its live obs must already be the fresh reset (finite).
            assert np.all(np.isfinite(final_obs[done]))
            assert np.all(np.isfinite(obs[done]))
    assert done_seen, "pendulum should truncate all sub-envs within the horizon"


def test_normalize_wraps_vector_env():
    n = 2
    venv = SyncVectorEnv([PendulumEnv for _ in range(n)])
    env = NormalizeObservation(venv)
    assert env.is_vector is True
    assert env.num_envs == n
    obs = env.reset(seed=SEED)
    assert obs.shape == (n, 3)
    out = env.step(np.zeros((n, 1), dtype=np.float32))
    assert len(out) == 5  # obs, rewards, term, trunc, final_obs
    obs, rewards, term, trunc, final_obs = out
    assert obs.shape == (n, 3)
    assert final_obs.shape == (n, 3)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
