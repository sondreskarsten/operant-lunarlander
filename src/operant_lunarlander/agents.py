import numpy as np


def softmax(x, beta=1.0):
    z = beta * (np.asarray(x) - np.max(x))
    e = np.exp(z)
    return e / np.sum(e)


def td_agent(alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_steps=150000, n_actions=4):
    state = {"step": 0}
    def epsilon():
        frac = min(1.0, state["step"] / eps_decay_steps)
        return eps_start + (eps_end - eps_start) * frac
    def select(table, s, rng):
        state["step"] += 1
        if rng.random() < epsilon():
            return int(rng.integers(n_actions))
        return int(np.argmax(table[s]))
    def update(table, s, a, r, s2, done):
        target = r if done else r + gamma * np.max(table[s2])
        table[s][a] += alpha * (target - table[s][a])
    def greedy(table, s, rng):
        return int(np.argmax(table[s]))
    def action_dist(table, s):
        d = np.zeros(n_actions)
        d[int(np.argmax(table[s]))] = 1.0
        return d
    return {"select": select, "update": update, "greedy": greedy, "action_dist": action_dist, "state": state}


def melioration_agent(alpha=0.1, beta=1.0, n_actions=4, baseline=True):
    base = {}
    def select(table, s, rng):
        p = softmax(table[s], beta)
        return int(rng.choice(n_actions, p=p))
    def update(table, s, a, r, s2, done):
        p = softmax(table[s], beta)
        if baseline:
            c = base.get(s, [0.0, 0])
            c[1] += 1
            c[0] += (r - c[0]) / c[1]
            base[s] = c
            b = c[0]
        else:
            b = 0.0
        one = np.zeros(n_actions)
        one[a] = 1.0
        table[s] += alpha * (r - b) * (one - p)
    def greedy(table, s, rng):
        return int(np.argmax(table[s]))
    def action_dist(table, s):
        return softmax(table[s], beta)
    return {"select": select, "update": update, "greedy": greedy, "action_dist": action_dist, "baseline": base}


def expected_sarsa_agent(alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_steps=150000, n_actions=4):
    state = {"step": 0}
    def epsilon():
        frac = min(1.0, state["step"] / eps_decay_steps)
        return eps_start + (eps_end - eps_start) * frac
    def policy(table, s, eps):
        d = np.full(n_actions, eps / n_actions)
        d[int(np.argmax(table[s]))] += 1.0 - eps
        return d
    def select(table, s, rng):
        state["step"] += 1
        if rng.random() < epsilon():
            return int(rng.integers(n_actions))
        return int(np.argmax(table[s]))
    def update(table, s, a, r, s2, done):
        if done:
            target = r
        else:
            target = r + gamma * float(np.dot(policy(table, s2, epsilon()), table[s2]))
        table[s][a] += alpha * (target - table[s][a])
    def greedy(table, s, rng):
        return int(np.argmax(table[s]))
    def action_dist(table, s):
        d = np.zeros(n_actions)
        d[int(np.argmax(table[s]))] = 1.0
        return d
    return {"select": select, "update": update, "greedy": greedy, "action_dist": action_dist, "state": state}


def boltzmann_td_agent(alpha=0.1, gamma=0.99, beta=1.0, n_actions=4):
    def select(table, s, rng):
        p = softmax(table[s], beta)
        return int(rng.choice(n_actions, p=p))
    def update(table, s, a, r, s2, done):
        target = r if done else r + gamma * np.max(table[s2])
        table[s][a] += alpha * (target - table[s][a])
    def greedy(table, s, rng):
        return int(np.argmax(table[s]))
    def action_dist(table, s):
        return softmax(table[s], beta)
    return {"select": select, "update": update, "greedy": greedy, "action_dist": action_dist}


def bush_mosteller_step(p, action, reinforced, alpha=0.1, scheme="R-I"):
    p = np.array(p, dtype=np.float64)
    if reinforced:
        p[action] += alpha * (1.0 - p[action])
        for j in range(len(p)):
            if j != action:
                p[j] -= alpha * p[j]
    elif scheme == "R-P":
        p[action] -= alpha * p[action]
        for j in range(len(p)):
            if j != action:
                p[j] += alpha * (1.0 - p[j]) / (len(p) - 1)
    return p / p.sum()


if __name__ == "__main__":
    print(softmax([0.0, 1.0, -1.0, 0.5], beta=1.0))
    print(bush_mosteller_step([0.5, 0.5], 0, True))
