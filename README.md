# operant-lunarlander

An instrument for **mechanistically differentiating** two reinforcement-driven learning rules on the same task and the same state representation:

- **(a) Law of effect with foresight** — reward-maximizing temporal-difference control (Q-learning). Value bootstraps over successor states; selection is greedy.
- **(b) Melioration** — a myopic linear-operator reinforcement-preference rule (gradient-bandit form). No temporal bootstrapping; selection allocates behaviour by relative reinforcement (probability matching).

The purpose is not to win the LunarLander leaderboard. It is to make the *difference* between maximizing and meliorating legible: where the two policies diverge, and whether (b) carries the empirical signature of operant choice (the matching law).

This continues the comparison line closed in the `losore-schema-engine` work — same intent (generalize across learning rules by contrasting their mechanics), clean benchmark instead of a messy real-world domain.

## The conceptual fork

Reinforcement learning *is* the computational formalization of operant conditioning: reward = reinforcer, policy = response tendencies, value = expected future reinforcement; Thorndike's Law of Effect is the direct ancestor of reward-based RL. So "apply operant conditioning to RL" is only non-trivial if (a) and (b) are made to *diverge*. They diverge on exactly two operational axes:

| Axis | (a) maximizing TD | (b) melioration |
|---|---|---|
| Temporal credit | `target = r + γ·max_a' Q(s',a')` (bootstraps future) | `H(s,a) += α·(r − r̄ₛ)·(1[a] − π(·|s))` (immediate reinforcement only) |
| Selection | greedy `argmax Q` | probability matching `π = softmax(H)` |

Both are model-free and tabular over the identical state featurizer. The *only* differences are bootstrapping-vs-myopia and greedy-vs-matching — which are the operational content of "law of effect with foresight" versus "melioration."

## Deviation from literal Bush–Mosteller

The agreed design named a Bush–Mosteller melioration agent. Literal two-action probability-space Bush–Mosteller does **not** extend cleanly to four actions and signed/continuous reward. (b) therefore uses the linear-operator preference generalization (gradient-bandit, Sutton & Barto §2.8): reinforcement raises the emitted response's preference, punishment lowers it, with a per-state running-mean reinforcement baseline. This preserves the operant content (myopic, reinforcement-driven, no foresight) and reproduces the matching law (below). An earlier rendering that tracked per-response reward *probability* and applied softmax was discarded: it equalizes at the matching equilibrium and collapses to indifferent ~50/50 choice (matching slope ≈ 0), so it is not a faithful operant learner.

## Architecture

```
src/operant_lunarlander/
  featurizer.py     binning over the 8-dim LunarLander obs -> discrete state key (configurable n_bins); constant featurizer for stateless schedules
  skeleton.py       agent-agnostic table + episode/training/eval rollouts; the table holds n_actions floats per state, interpreted as Q (a) or preference H (b)
  agents.py         five rules: td_agent (Q-learning), expected_sarsa_agent, boltzmann_td_agent, melioration_agent (gradient-bandit), bush_mosteller_step (probability-space); each exposes select / update / greedy / action_dist
  schedules.py      ConcurrentVI diagnostic env + generalized-matching-law fit (the matching positive control)
  operant.py        operant lever battery: schedule family (FR/VR/FI/VI), OperantChamber (magnitude / punishment / response-cost / extinction), ConcurrentSchedule (VI-VI/VR-VR/VI-VR), MeliorationTrap (the matching-vs-maximizing discriminator), and the experiment runners
  differentiate.py  LunarLander harness: train rules, compare return, policy divergence, matching control, granularity sweep
  ceiling.py        optional SB3 PPO optimality ceiling (separate from the core comparison)
```

The shared skeleton is the point: the same loop, table, and featurizer run both agents, so any measured difference is attributable to the update + selection rules, not to engineering. (This is why the optimality ceiling uses a library agent but the (a)-vs-(b) contrast does not — a tuned SB3 agent for (a) would confound algorithmic difference with replay/target-net engineering.)

## Positive control: does (b) actually meliorate?

Before trusting (b)'s behaviour on a sequential MDP, validate it in the domain where the matching law was discovered: concurrent variable-interval schedules. `fit_generalized_matching` runs (b) across seven VI ratio conditions and fits Baum's generalized matching law, `log(B₁/B₂) = a·log(R₁/R₂) + log b`.

Result: **slope a = 0.973, bias b = 1.000** — strict, unbiased matching. (b) is a faithful operant learner; its LunarLander behaviour is melioration, not a bug.

Note: concVI-VI validates lawful reinforcement-proportional allocation but is a *weak* discriminator between matching and maximizing, because on VI schedules the matching allocation is itself near reward-maximizing. Clean separation requires the melioration trap (below).

## Operant levers

The instrument exposes the canonical operant manipulations on low-dimensional, tractable envs (where, unlike LunarLander, tabular learning is not coverage-starved, so the learning-rule differences are not masked by representation).

### Schedule family — interval vs ratio (`schedule_matching_table`)

Melioration run across seven ratio conditions on each of three concurrent schedules, fitting the generalized matching law:

| Schedule | slope a | bias b | mean exclusivity |
|---|---|---|---|
| conc VI-VI | 0.973 | 1.00 | 0.70 |
| conc VI-VR | 0.966 | 1.02 | 0.69 |
| conc VR-VR | undefined | — | 0.97 |

Interval schedules produce graded matching (slope ≈ 1); the ratio-ratio schedule produces near-exclusive choice of the richer ratio, so the log-ratio slope is mathematically undefined (one alternative receives ~0 reinforcers). This is the textbook interval-vs-ratio distinction — and the reason VR-VR cannot separate matching from maximizing (both predict exclusivity there).

### Melioration trap — the discriminator (`melioration_trap_experiment`)

Two alternatives; A's local reinforcement probability falls as A is chosen more (`p_A = a − c·x`, x = leaky fraction of recent A-choices), B's is constant (`p_B = b`). Parameters chosen so the **matching point (x_m = 0.8, rate 0.40) ≠ the optimum (x* = 0.4, rate 0.48)**. The observation is x, binned; the identical featurizer/skeleton run all rules.

| Rule | tail x | tail reward rate |
|---|---|---|
| optimum (analytic) | 0.40 | 0.48 |
| matching point (analytic) | 0.80 | 0.40 |
| expected-SARSA (maximizing) | 0.51 | 0.47 |
| Q-learning (maximizing) | 0.76 | 0.41 |
| melioration | 1.00 | 0.30 |

Melioration is **caught** — local reinforcement drives it past even the matching point into near-exclusive A, at rate 0.30. The maximizing rules **escape** via foresight over the allocation-state: expected-SARSA reaches near-optimal 0.47, Q-learning partially escapes at 0.41. This is the clean maximization-vs-melioration separation concVI-VI could not provide, and it also resolves *within* the maximizing family (expected-SARSA > Q-learning under these continuing, self-induced-state dynamics).

### Extinction (`extinction_experiment`)

Acquire responding (respond/withhold chamber, small response cost) under three schedules, then withhold reinforcement:

| Acquisition schedule | acq response rate | steps to extinction |
|---|---|---|
| CRF (FR1) | 0.98 | 444 |
| VR5 | 0.97 | 380 |
| VI10 | 0.96 | 322 |

All schedules acquire, then extinguish, with a clean monotonic ordering: CRF (highest acquired value) is most resistant. Note this is the acquired-value-magnitude effect and is **anti-PREE** — the real partial-reinforcement-extinction effect (partial *more* resistant) requires discrimination/sequential mechanisms that a simple value learner does not have. The lever is faithful; the molar PREE phenomenon is explicitly out of reach for this rule class.

Other levers present on `OperantChamber`: reinforcement magnitude, punishment probability/magnitude, response cost.

## Run

```bash
pip install -e .
python -m operant_lunarlander.differentiate         # LunarLander: writes results/comparison.json
python -m operant_lunarlander.operant               # operant battery: writes results/operant_battery.json
python -m operant_lunarlander.schedules             # matching-law fit
pytest -q                                            # 18 tests
```

Optional optimality ceiling:

```bash
pip install -e ".[ceiling]"
python -m operant_lunarlander.ceiling
```

## Current results (n_train=600, n_bins=7, seed=0, single CPU, ~24s)

| Metric | Value |
|---|---|
| Matching law slope / bias | 0.973 / 1.000 |
| Policy divergence (mean TV) | 0.63 |
| Policy divergence (argmax agreement) | 0.43 |
| TD eval return (greedy) | −354 ± 65 |
| Melioration eval return (matching) | −198 ± 97 |
| Melioration eval return (argmax) | −219 ± 58 |
| States visited (TD / melioration) | 182 / 271 |

**The performance comparison is not yet conclusive, and the repo says so.** At this budget neither agent solves LunarLander (both deeply negative), and the *ranking* between them is a confound, not a signal:

- Granularity sweep (`results/granularity_sweep.json`): the TD−melioration gap sign-flips with binning — +14 (4 bins), +177 (7 bins), −15 (10 bins). A robust maximization-vs-melioration effect would not flip sign under reparametrization of the same observations.
- Tabular binning over an 8-dim space is coverage-starved: ~100s of states visited out of `n_bins⁶·4`, so Q is essentially uninformed in most states and greedy-on-uninformed-Q dives. The state representation, not the learning rule, is the binding constraint — the direct analogue of the proxy-reward binding constraint in `losore-schema-engine`.

What *is* stable across binning: policy divergence (mean TV ≈ 0.63–0.75 for all of 4/7/10 bins) and the matching control (binning-independent). Those are the defensible findings of v0.

## Binding constraint and next steps

The clean maximization-vs-melioration separation is **delivered** on the melioration trap (above), where the state is low-dimensional and tabular learning is not coverage-starved. The remaining open item is the *LunarLander* performance comparison specifically:

1. Replace binning with tile/coarse coding (linear function approximation) so (a)'s foresight can generalize across states — without generalization a tabular agent cannot express a landing policy on this observation space.
2. Match the eval protocol (same temperature for both, or report both deterministic and stochastic reads for each) to remove the greedy-vs-stochastic asymmetry.
3. ~~Add a melioration-trap diagnostic where matching is provably suboptimal~~ — **done** (`MeliorationTrap`); this is now the headline discriminator.

These are empirical knobs (`n_train`, `n_bins`, `granularity_sweep`, the schedule set, the trap parameters), not design questions. Whether the *LunarLander* gap survives function approximation is to be measured, not assumed.

## Counterpoint tracking

```
old: (b) loses to (a) on LunarLander reward => melioration is the cause
new: on LunarLander the gap is a discretization/undertraining artifact (sign-flips across n_bins; both agents fail)
because: tabular binning over 8 continuous dims is coverage-starved, so neither rule can express a good policy
  - resolved elsewhere by: the melioration trap (low-dim state) cleanly separates the rules; LunarLander transfer still needs function approximation
  - caution: "operant agent loses to TD" is also the expected outcome for a descriptive rule, so a LunarLander loss alone is not evidence of anything

old: concVI-VI matching (slope≈1) proves (b) is melioration-specific
new: it proves (b) produces lawful reinforcement-proportional allocation, necessary but not sufficient
because: on VI schedules matching ≈ maximizing, so a working maximizer would also show slope≈1
  - resolved by: the melioration trap, where matching point (0.8) and optimum (0.4) provably differ and only the myopic rule is caught

old: the melioration trap proves (b) is uniquely suboptimal
new: it proves myopia is caught by a trap that foresight escapes; the trap separates myopic-vs-bootstrapping, which is one of the two axes
because: a myopic maximizer (gradient bandit) would also be caught — the escape is due to bootstrapping over the allocation-state, not to "maximizing" per se
  - so the trap isolates the temporal-credit axis; the selection axis (greedy vs matching) is isolated by the VI-VI matching control

old: extinction ordering (CRF most resistant) is the partial-reinforcement-extinction effect
new: it is the opposite — the acquired-value-magnitude effect (anti-PREE)
because: a value learner extinguishes fastest where acquired value is lowest (sparse schedules); real PREE needs discrimination/sequential memory
  - PREE is explicitly out of reach for this rule class and is not claimed
```

## References

Thorndike (1911) *Animal Intelligence*; Skinner (1938) *The Behavior of Organisms*; Herrnstein (1961) matching law; Herrnstein & Vaughan (1980) melioration; Baum (1974) generalized matching law; Bush & Mosteller (1955) *Stochastic Models for Learning*; Sutton & Barto (2018) *Reinforcement Learning* (Ch. 1 history; §2.8 gradient bandit); Loewenstein & Seung (2006) matching as a fixed point of covariance-based plasticity.
