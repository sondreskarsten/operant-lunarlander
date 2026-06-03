import numpy as np
from .skeleton import make_table
from .agents import melioration_agent


class ConcurrentVI:
    def __init__(self, vi_means=(30.0, 90.0), n_actions=2, rng=None):
        self.vi_means = vi_means
        self.n_actions = n_actions
        self.rng = rng if rng is not None else np.random.default_rng()
        self.armed = [False] * n_actions
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.armed = [False] * self.n_actions
        return np.zeros(1), {}
    def step(self, action):
        for i in range(self.n_actions):
            if not self.armed[i]:
                if self.rng.random() < 1.0 / self.vi_means[i]:
                    self.armed[i] = True
        r = 0.0
        if self.armed[action]:
            r = 1.0
            self.armed[action] = False
        return np.zeros(1), r, False, False, {}


def run_matching(agent, vi_means=(30.0, 90.0), n_steps=20000, seed=0):
    rng = np.random.default_rng(seed)
    env = ConcurrentVI(vi_means=vi_means, n_actions=2, rng=rng)
    table = make_table(2)
    select, update = agent["select"], agent["update"]
    env.reset(seed=seed)
    s = 0
    B = np.zeros(2)
    R = np.zeros(2)
    for _ in range(n_steps):
        a = select(table, s, rng)
        _, r, _, _, _ = env.step(a)
        update(table, s, a, float(r), 0, False)
        B[a] += 1
        if r > 0:
            R[a] += 1
    return B, R


def fit_generalized_matching(conditions=None, alpha=0.1, beta=1.0, n_steps=20000, seed=0):
    if conditions is None:
        conditions = [(20, 60), (30, 90), (45, 45), (60, 30), (90, 30), (40, 120), (120, 40)]
    log_b = []
    log_r = []
    for k, (v1, v2) in enumerate(conditions):
        agent = melioration_agent(alpha=alpha, beta=beta, n_actions=2)
        B, R = run_matching(agent, vi_means=(v1, v2), n_steps=n_steps, seed=seed + k)
        log_b.append(np.log(B[0] / B[1]))
        log_r.append(np.log(R[0] / R[1]))
    log_b = np.array(log_b)
    log_r = np.array(log_r)
    design = np.vstack([log_r, np.ones_like(log_r)]).T
    coef, _, _, _ = np.linalg.lstsq(design, log_b, rcond=None)
    return {"slope": float(coef[0]), "bias": float(np.exp(coef[1])), "log_r": log_r.tolist(), "log_b": log_b.tolist(), "conditions": conditions}


if __name__ == "__main__":
    print(fit_generalized_matching())
