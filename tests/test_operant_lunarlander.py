import numpy as np
import gymnasium as gym
from operant_lunarlander.featurizer import lunar_featurizer, constant_featurizer, make_bin_edges, LUNAR_LOW, LUNAR_HIGH
from operant_lunarlander.skeleton import make_table, run_episode, run_training, evaluate
from operant_lunarlander.agents import td_agent, melioration_agent, softmax
from operant_lunarlander.schedules import ConcurrentVI, run_matching, fit_generalized_matching
from operant_lunarlander.differentiate import differentiate, policy_divergence, make_lunar


def test_featurizer_bounds():
    f = lunar_featurizer(n_bins=7)
    assert f(LUNAR_LOW) == (0, 0, 0, 0, 0, 0, 0, 0)
    assert f(LUNAR_HIGH) == (6, 6, 6, 6, 6, 6, 1, 1)
    assert len(f(np.zeros(8))) == 8


def test_constant_featurizer():
    f = constant_featurizer()
    assert f(np.random.rand(8)) == 0


def test_softmax_simplex():
    p = softmax([1.0, 2.0, 3.0, 0.0], beta=1.0)
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p > 0).all()


def test_td_update_moves_toward_target():
    a = td_agent(alpha=0.5, gamma=0.0)
    t = make_table(4)
    a["update"](t, (0,), 1, 10.0, (0,), True)
    assert abs(t[(0,)][1] - 5.0) < 1e-9


def test_melioration_update_changes_preference():
    b = melioration_agent(alpha=0.5, beta=1.0, baseline=False)
    t = make_table(4)
    before = t[(0,)].copy()
    b["update"](t, (0,), 2, 1.0, (0,), False)
    assert t[(0,)][2] > before[2]


def test_selectors_return_valid_actions():
    rng = np.random.default_rng(0)
    t = make_table(4)
    a = td_agent()
    b = melioration_agent()
    assert a["select"](t, (0,), rng) in range(4)
    assert b["select"](t, (0,), rng) in range(4)


def test_make_table_default_zeros():
    t = make_table(4)
    assert np.allclose(t[("x",)], np.zeros(4))


def test_run_episode_finite():
    env = make_lunar()
    feat = lunar_featurizer()
    a = td_agent()
    t = make_table(4)
    total, steps, visited = run_episode(env, t, a["select"], a["update"], feat, max_steps=300, seed=0)
    assert np.isfinite(total)
    assert steps > 0
    assert len(visited) == steps


def test_concurrent_vi_delivers_reinforcement():
    rng = np.random.default_rng(0)
    env = ConcurrentVI((5.0, 5.0), 2, rng)
    env.reset(seed=0)
    got = sum(env.step(i % 2)[1] for i in range(2000))
    assert got > 0


def test_operant_chamber_extinction_zeroes_reward():
    from operant_lunarlander.operant import OperantChamber, make_schedule
    rng = np.random.default_rng(0)
    env = OperantChamber(schedule=make_schedule("VR", 3), rng=rng)
    env.reset(seed=0)
    acq = sum(env.step(0)[1] for _ in range(2000))
    env.extinction = True
    ext = sum(env.step(0)[1] for _ in range(2000))
    assert acq > 0
    assert ext <= 0


def test_schedule_factory_types():
    from operant_lunarlander.operant import make_schedule, FR, VR, FI, VI
    assert isinstance(make_schedule("FR", 2), FR)
    assert isinstance(make_schedule("VR", 2), VR)
    assert isinstance(make_schedule("FI", 2), FI)
    assert isinstance(make_schedule("VI", 2), VI)


def test_melioration_trap_points():
    from operant_lunarlander.operant import MeliorationTrap
    e = MeliorationTrap()
    assert e.matching_point()["x_match"] > e.optimum()["x_opt"]
    assert e.optimum()["rate_opt"] > e.matching_point()["rate_match"]


def test_trap_maximizer_beats_melioration():
    from operant_lunarlander.operant import melioration_trap_experiment
    out = melioration_trap_experiment(n_steps=30000, seed=0)
    mel = out["rules"]["melioration"]["reward_rate_tail"]
    esarsa = out["rules"]["expected_sarsa"]["reward_rate_tail"]
    assert esarsa > mel
    assert out["rules"]["melioration"]["x_tail"] > 0.6


def test_schedule_matching_interval_graded_ratio_exclusive():
    from operant_lunarlander.operant import schedule_matching_table
    t = schedule_matching_table(n_steps=12000, seed=0)
    assert 0.6 <= t["conc_VI_VI"]["slope"] <= 1.4
    assert t["conc_VR_VR"]["mean_exclusivity"] > 0.9


def test_extinction_acquires_and_extinguishes():
    from operant_lunarlander.operant import extinction_experiment
    out = extinction_experiment(acquire_steps=6000, extinction_steps=6000, seed=0)
    assert out["CRF"]["acq_response_rate"] > 0.8
    assert out["CRF"]["steps_to_extinction"] < 6000


def test_bush_mosteller_simplex():
    from operant_lunarlander.agents import bush_mosteller_step
    p = bush_mosteller_step([0.5, 0.5], 0, True, alpha=0.2)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > 0.5


def test_matching_law_holds():
    r = fit_generalized_matching(n_steps=12000, seed=0)
    assert 0.6 <= r["slope"] <= 1.4
    assert 0.5 <= r["bias"] <= 2.0


def test_differentiate_keys_and_finiteness():
    out, ra, rb = differentiate(n_train=40, n_eval=10, seed=0)
    for k in ["td_eval_return", "melioration_eval_return", "policy_divergence", "matching_law"]:
        assert k in out
    assert np.isfinite(out["td_eval_return"]["mean"])
    assert np.isfinite(out["melioration_eval_return"]["mean"])
    assert 0.0 <= out["policy_divergence"]["mean_tv"] <= 1.0
    assert 0.0 <= out["policy_divergence"]["argmax_agreement"] <= 1.0
