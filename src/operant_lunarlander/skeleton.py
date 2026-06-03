from collections import defaultdict
import numpy as np


def make_table(n_actions=4):
    return defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))


def run_episode(env, table, select, update, featurize, max_steps=1000, train=True, rng=None, seed=None):
    if rng is None:
        rng = np.random.default_rng()
    obs, _ = env.reset(seed=seed)
    s = featurize(obs)
    total = 0.0
    steps = 0
    visited = []
    while steps < max_steps:
        a = select(table, s, rng)
        obs2, r, term, trunc, _ = env.step(a)
        s2 = featurize(obs2)
        if train:
            update(table, s, a, float(r), s2, term)
        visited.append((s, a))
        total += float(r)
        s = s2
        steps += 1
        if term or trunc:
            break
    return total, steps, visited


def run_training(env, table, select, update, featurize, n_episodes=500, max_steps=1000, rng=None, seed=0):
    if rng is None:
        rng = np.random.default_rng(seed)
    returns = []
    for ep in range(n_episodes):
        total, _, _ = run_episode(env, table, select, update, featurize, max_steps=max_steps, train=True, rng=rng, seed=seed + ep)
        returns.append(total)
    return returns


def evaluate(env, table, select, featurize, n_episodes=50, max_steps=1000, rng=None, seed=10000):
    if rng is None:
        rng = np.random.default_rng(seed)
    noop = lambda *a, **k: None
    returns = []
    for ep in range(n_episodes):
        total, _, _ = run_episode(env, table, select, noop, featurize, max_steps=max_steps, train=False, rng=rng, seed=seed + ep)
        returns.append(total)
    return returns
