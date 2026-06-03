import numpy as np
from .skeleton import make_table
from .agents import td_agent, melioration_agent, expected_sarsa_agent, bush_mosteller_step
from .featurizer import constant_featurizer, interval_featurizer


class FR:
    def __init__(self, n=5):
        self.n = n
        self.count = 0
    def tick(self, rng):
        pass
    def respond(self, rng):
        self.count += 1
        if self.count >= self.n:
            self.count = 0
            return True
        return False


class VR:
    def __init__(self, n=5):
        self.n = n
    def tick(self, rng):
        pass
    def respond(self, rng):
        return rng.random() < 1.0 / self.n


class FI:
    def __init__(self, t=10):
        self.t = t
        self.timer = 0
    def tick(self, rng):
        self.timer += 1
    def respond(self, rng):
        if self.timer >= self.t:
            self.timer = 0
            return True
        return False


class VI:
    def __init__(self, t=10):
        self.t = t
        self.armed = False
    def tick(self, rng):
        if not self.armed and rng.random() < 1.0 / self.t:
            self.armed = True
    def respond(self, rng):
        if self.armed:
            self.armed = False
            return True
        return False


def make_schedule(kind="VR", param=5):
    return {"FR": FR, "VR": VR, "FI": FI, "VI": VI}[kind](param)


class OperantChamber:
    def __init__(self, schedule=None, magnitude=1.0, punish_prob=0.0, punish_mag=1.0, response_cost=0.0, extinction=False, rng=None):
        self.schedule = schedule if schedule is not None else make_schedule("VR", 5)
        self.magnitude = magnitude
        self.punish_prob = punish_prob
        self.punish_mag = punish_mag
        self.response_cost = response_cost
        self.extinction = extinction
        self.rng = rng if rng is not None else np.random.default_rng()
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return np.zeros(1), {}
    def step(self, action):
        self.schedule.tick(self.rng)
        r = 0.0
        if action == 0:
            if not self.extinction and self.schedule.respond(self.rng):
                r += self.magnitude
            if self.rng.random() < self.punish_prob:
                r -= self.punish_mag
            r -= self.response_cost
        return np.zeros(1), r, False, False, {}


class ConcurrentSchedule:
    def __init__(self, schedules=None, magnitudes=(1.0, 1.0), rng=None):
        self.schedules = schedules if schedules is not None else [make_schedule("VI", 30), make_schedule("VI", 90)]
        self.magnitudes = magnitudes
        self.rng = rng if rng is not None else np.random.default_rng()
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return np.zeros(1), {}
    def step(self, action):
        for s in self.schedules:
            s.tick(self.rng)
        r = self.magnitudes[action] if self.schedules[action].respond(self.rng) else 0.0
        return np.zeros(1), r, False, False, {}


class MeliorationTrap:
    def __init__(self, a=0.8, b=0.4, c=0.5, d=0.0, leak=0.02, rng=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.leak = leak
        self.x = 0.5
        self.rng = rng if rng is not None else np.random.default_rng()
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = 0.5
        return np.array([self.x]), {}
    def _rates(self, x):
        return self.a - self.c * x, self.b - self.d * (1.0 - x)
    def step(self, action):
        pa, pb = self._rates(self.x)
        p = pa if action == 0 else pb
        p = min(max(p, 0.0), 1.0)
        r = 1.0 if self.rng.random() < p else 0.0
        choice_a = 1.0 if action == 0 else 0.0
        self.x = (1.0 - self.leak) * self.x + self.leak * choice_a
        return np.array([self.x]), r, False, False, {}
    def optimum(self, grid=201):
        xs = np.linspace(0.0, 1.0, grid)
        g = []
        for x in xs:
            pa, pb = self._rates(x)
            g.append(x * min(max(pa, 0.0), 1.0) + (1.0 - x) * min(max(pb, 0.0), 1.0))
        g = np.array(g)
        return {"x_opt": float(xs[int(np.argmax(g))]), "rate_opt": float(np.max(g))}
    def matching_point(self):
        x = (self.a - self.b + self.d) / (self.c + self.d)
        x = min(max(x, 0.0), 1.0)
        pa, pb = self._rates(x)
        return {"x_match": float(x), "rate_match": float(x * min(max(pa, 0.0), 1.0) + (1.0 - x) * min(max(pb, 0.0), 1.0))}


def run_continuing(env, agent, featurize, n_steps=50000, seed=0, train=True):
    rng = np.random.default_rng(seed)
    table = make_table(2)
    select, update = agent["select"], agent["update"]
    obs, _ = env.reset(seed=seed)
    s = featurize(obs)
    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps)
    xs = np.zeros(n_steps)
    for i in range(n_steps):
        a = select(table, s, rng)
        obs2, r, term, trunc, _ = env.step(a)
        s2 = featurize(obs2)
        if train:
            update(table, s, a, float(r), s2, term)
        rewards[i] = r
        actions[i] = a
        xs[i] = obs2[0]
        s = s2
    return rewards, actions, xs, table


def melioration_trap_experiment(n_steps=60000, n_bins=20, gamma=0.99, alpha_q=0.2, alpha_mel=0.1, seed=0, tail=0.2):
    env_opt = MeliorationTrap()
    opt = env_opt.optimum()
    match = env_opt.matching_point()
    feat = interval_featurizer(n_bins=n_bins)
    k = int(n_steps * tail)
    rules = {
        "q_learning": td_agent(alpha=alpha_q, gamma=gamma, eps_decay_steps=n_steps // 2, n_actions=2),
        "expected_sarsa": expected_sarsa_agent(alpha=alpha_q, gamma=gamma, eps_decay_steps=n_steps // 2, n_actions=2),
        "melioration": melioration_agent(alpha=alpha_mel, beta=1.0, n_actions=2),
    }
    out = {"optimum": opt, "matching_point": match, "rules": {}}
    for name, agent in rules.items():
        r, a, x, _ = run_continuing(MeliorationTrap(), agent, feat, n_steps=n_steps, seed=seed)
        out["rules"][name] = {
            "reward_rate_tail": float(np.mean(r[-k:])),
            "x_tail": float(np.mean(x[-k:])),
            "frac_A_tail": float(np.mean(a[-k:] == 0)),
        }
    return out


def fit_matching_general(make_pair, params, alpha_mel=0.1, beta=1.0, n_steps=20000, seed=0):
    log_b = []
    log_r = []
    excl = []
    for k, prm in enumerate(params):
        agent = melioration_agent(alpha=alpha_mel, beta=beta, n_actions=2)
        env = ConcurrentSchedule(schedules=make_pair(prm), rng=np.random.default_rng(seed + k))
        r, a, _, _ = run_continuing(env, agent, constant_featurizer(), n_steps=n_steps, seed=seed + k)
        b1 = np.sum(a == 0)
        b2 = np.sum(a == 1)
        rr1 = np.sum((a == 0) & (r > 0))
        rr2 = np.sum((a == 1) & (r > 0))
        excl.append(max(b1, b2) / (b1 + b2))
        lb = np.log(b1 / b2) if b1 > 0 and b2 > 0 else np.nan
        lr = np.log(rr1 / rr2) if rr1 > 0 and rr2 > 0 else np.nan
        log_b.append(lb)
        log_r.append(lr)
    log_b = np.array(log_b)
    log_r = np.array(log_r)
    mask = np.isfinite(log_b) & np.isfinite(log_r)
    if mask.sum() >= 2:
        design = np.vstack([log_r[mask], np.ones(int(mask.sum()))]).T
        coef, _, _, _ = np.linalg.lstsq(design, log_b[mask], rcond=None)
        slope = float(coef[0])
        bias = float(np.exp(coef[1]))
    else:
        slope = float("nan")
        bias = float("nan")
    return {"slope": slope, "bias": bias, "mean_exclusivity": float(np.mean(excl)), "n_graded": int(mask.sum()), "log_r": log_r.tolist(), "log_b": log_b.tolist()}


def schedule_matching_table(n_steps=20000, seed=0):
    vi_vi = fit_matching_general(lambda p: [make_schedule("VI", p[0]), make_schedule("VI", p[1])], [(20, 60), (30, 90), (45, 45), (60, 30), (90, 30), (40, 120), (120, 40)], n_steps=n_steps, seed=seed)
    vr_vr = fit_matching_general(lambda p: [make_schedule("VR", p[0]), make_schedule("VR", p[1])], [(8, 24), (12, 36), (18, 18), (24, 12), (36, 12), (16, 48), (48, 16)], n_steps=n_steps, seed=seed)
    vi_vr = fit_matching_general(lambda p: [make_schedule("VI", p[0]), make_schedule("VR", p[1])], [(30, 30), (45, 20), (60, 15), (90, 12), (30, 12), (60, 24), (90, 30)], n_steps=n_steps, seed=seed)
    return {"conc_VI_VI": vi_vi, "conc_VR_VR": vr_vr, "conc_VI_VR": vi_vr}


def extinction_experiment(acquire_steps=8000, extinction_steps=8000, alpha=0.02, response_cost=0.02, q0=1.0, seed=0, window=300, threshold=0.2):
    schedules = {"CRF": ("FR", 1), "VR5": ("VR", 5), "VI10": ("VI", 10)}
    out = {}
    for name, (kind, param) in schedules.items():
        agent = td_agent(alpha=alpha, gamma=0.0, eps_start=1.0, eps_end=0.05, eps_decay_steps=acquire_steps // 2, n_actions=2)
        rng = np.random.default_rng(seed)
        table = make_table(2)
        table[0] = np.full(2, q0, dtype=np.float64)
        env = OperantChamber(schedule=make_schedule(kind, param), response_cost=response_cost, rng=rng)
        s = 0
        acq = np.zeros(acquire_steps)
        for t in range(acquire_steps):
            a = agent["select"](table, s, rng)
            _, r, _, _, _ = env.step(a)
            agent["update"](table, s, a, float(r), 0, False)
            acq[t] = a
        env.extinction = True
        responses = 0
        steps_to_ext = extinction_steps
        recent = []
        for t in range(extinction_steps):
            a = agent["select"](table, s, rng)
            _, r, _, _, _ = env.step(a)
            agent["update"](table, s, a, float(r), 0, False)
            responses += int(a == 0)
            recent.append(int(a == 0))
            if len(recent) > window:
                recent.pop(0)
            if len(recent) == window and np.mean(recent) < threshold:
                steps_to_ext = t
                break
        out[name] = {"acq_response_rate": float(np.mean(acq[-window:] == 0)), "responses_in_extinction": responses, "steps_to_extinction": steps_to_ext}
    return out


def operant_battery(seed=0):
    return {
        "melioration_trap": melioration_trap_experiment(seed=seed),
        "schedule_matching": schedule_matching_table(seed=seed),
        "extinction": extinction_experiment(seed=seed),
    }


def main(out_path="results/operant_battery.json"):
    import json
    res = operant_battery()
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
