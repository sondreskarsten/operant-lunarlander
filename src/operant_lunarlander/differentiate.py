import json
import numpy as np
import gymnasium as gym
from .featurizer import lunar_featurizer
from .skeleton import make_table, run_training, evaluate, run_episode
from .agents import td_agent, melioration_agent
from .schedules import fit_generalized_matching


def make_lunar():
    return gym.make("LunarLander-v3", continuous=False)


def collect_states(env, featurize, n_episodes=20, max_steps=1000, seed=99999):
    rng = np.random.default_rng(seed)
    rnd_select = lambda table, s, r: int(r.integers(env.action_space.n))
    noop = lambda *a, **k: None
    seen = set()
    for ep in range(n_episodes):
        _, _, visited = run_episode(env, make_table(env.action_space.n), rnd_select, noop, featurize, max_steps=max_steps, train=False, rng=rng, seed=seed + ep)
        for s, _ in visited:
            seen.add(s)
    return list(seen)


def policy_divergence(table_a, dist_a, table_b, dist_b, states):
    tv = []
    agree = 0
    for s in states:
        da = dist_a(table_a, s)
        db = dist_b(table_b, s)
        tv.append(0.5 * np.sum(np.abs(da - db)))
        if int(np.argmax(da)) == int(np.argmax(db)):
            agree += 1
    return {"mean_tv": float(np.mean(tv)), "argmax_agreement": float(agree / len(states))}


def differentiate(n_train=600, n_eval=100, n_bins=7, alpha_td=0.1, gamma=0.99, alpha_mel=0.05, beta_mel=1.0, matching_beta=1.0, seed=0):
    env = make_lunar()
    feat = lunar_featurizer(n_bins=n_bins)
    a = td_agent(alpha=alpha_td, gamma=gamma, n_actions=4)
    b = melioration_agent(alpha=alpha_mel, beta=beta_mel, n_actions=4)
    table_a = make_table(4)
    table_b = make_table(4)
    train_a = run_training(env, table_a, a["select"], a["update"], feat, n_episodes=n_train, seed=seed)
    train_b = run_training(env, table_b, b["select"], b["update"], feat, n_episodes=n_train, seed=seed)
    eval_a = evaluate(env, table_a, a["greedy"], feat, n_episodes=n_eval, seed=seed + 50000)
    eval_b = evaluate(env, table_b, b["select"], feat, n_episodes=n_eval, seed=seed + 50000)
    eval_b_argmax = evaluate(env, table_b, b["greedy"], feat, n_episodes=n_eval, seed=seed + 50000)
    states = collect_states(make_lunar(), feat)
    div = policy_divergence(table_a, a["action_dist"], table_b, b["action_dist"], states)
    matching = fit_generalized_matching(beta=matching_beta)
    out = {
        "config": {"n_train": n_train, "n_eval": n_eval, "n_bins": n_bins, "alpha_td": alpha_td, "gamma": gamma, "alpha_mel": alpha_mel, "beta_mel": beta_mel, "matching_beta": matching_beta, "seed": seed},
        "td_eval_return": {"mean": float(np.mean(eval_a)), "std": float(np.std(eval_a))},
        "melioration_eval_return": {"mean": float(np.mean(eval_b)), "std": float(np.std(eval_b))},
        "melioration_argmax_eval_return": {"mean": float(np.mean(eval_b_argmax)), "std": float(np.std(eval_b_argmax))},
        "td_train_last50": float(np.mean(train_a[-50:])),
        "melioration_train_last50": float(np.mean(train_b[-50:])),
        "policy_divergence": div,
        "matching_law": {"slope": matching["slope"], "bias": matching["bias"]},
        "n_states_compared": len(states),
        "td_states_visited": len(table_a),
        "melioration_states_visited": len(table_b),
    }
    return out, train_a, train_b


def granularity_sweep(bins_list=(4, 7, 10), n_train=400, n_eval=50, seed=0):
    rows = []
    for nb in bins_list:
        out, _, _ = differentiate(n_train=n_train, n_eval=n_eval, n_bins=nb, seed=seed)
        rows.append({
            "n_bins": nb,
            "td": out["td_eval_return"]["mean"],
            "melioration": out["melioration_eval_return"]["mean"],
            "gap": out["td_eval_return"]["mean"] - out["melioration_eval_return"]["mean"],
            "mean_tv": out["policy_divergence"]["mean_tv"],
        })
    return rows


def main(n_train=600, n_eval=100, out_path="results/comparison.json"):
    out, train_a, train_b = differentiate(n_train=n_train, n_eval=n_eval)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
