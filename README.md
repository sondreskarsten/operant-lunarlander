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
  agents.py         td_agent (a), melioration_agent (b); each exposes select / update / greedy / action_dist
  schedules.py      ConcurrentVI diagnostic env + generalized-matching-law fit
  differentiate.py  headline harness: train both, compare return, policy divergence, matching control, granularity sweep
  ceiling.py        optional SB3 PPO optimality ceiling (separate from the core comparison)
```

The shared skeleton is the point: the same loop, table, and featurizer run both agents, so any measured difference is attributable to the update + selection rules, not to engineering. (This is why the optimality ceiling uses a library agent but the (a)-vs-(b) contrast does not — a tuned SB3 agent for (a) would confound algorithmic difference with replay/target-net engineering.)

## Positive control: does (b) actually meliorate?

Before trusting (b)'s behaviour on a sequential MDP, validate it in the domain where the matching law was discovered: concurrent variable-interval schedules. `fit_generalized_matching` runs (b) across seven VI ratio conditions and fits Baum's generalized matching law, `log(B₁/B₂) = a·log(R₁/R₂) + log b`.

Result: **slope a = 0.973, bias b = 1.000** — strict, unbiased matching. (b) is a faithful operant learner; its LunarLander behaviour is melioration, not a bug.

Note: concVI-VI validates lawful reinforcement-proportional allocation but is a *weak* discriminator between matching and maximizing, because on VI schedules the matching allocation is itself near reward-maximizing. Clean separation of matching from maximizing requires ratio (concVR) or melioration-trap schedules — a planned addition.

## Run

```bash
pip install -e .
python -m operant_lunarlander.differentiate         # writes results/comparison.json
python -m operant_lunarlander.schedules             # matching-law fit
pytest -q                                            # 11 tests
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

To turn the performance comparison into a clean test of maximization-vs-melioration:

1. Replace binning with tile/coarse coding (linear function approximation) so (a)'s foresight can generalize across states — without generalization a tabular agent cannot express a landing policy on this observation space.
2. Match the eval protocol (same temperature for both, or report both deterministic and stochastic reads for each) to remove the greedy-vs-stochastic asymmetry.
3. Add a concVR / melioration-trap diagnostic where matching is *provably* suboptimal, so divergence from optimality is attributable to melioration rather than to representation.

These are empirical knobs (`n_train`, `n_bins`, `granularity_sweep`, the schedule set), not design questions. Whether the gap survives is to be measured, not assumed.

## Counterpoint tracking

```
old: (b) loses to (a) on LunarLander reward => melioration is the cause
new: at v0 the gap is a discretization/undertraining artifact (sign-flips across n_bins; both agents fail)
because: tabular binning over 8 continuous dims is coverage-starved, so neither rule can express a good policy
  - may be true because: with function approximation and matched eval the gap may stabilize and reveal a real maximize-vs-meliorate difference
  - may be wrong because: the difference could remain dominated by representation/eval choices; "operant agent loses to TD" is also the *expected* outcome for a descriptive (non-normative) rule, so a loss is not itself evidence of anything

old: concVI-VI matching (slope≈1) proves (b) is melioration-specific
new: it proves (b) produces lawful reinforcement-proportional allocation, necessary but not sufficient for melioration-vs-maximization
because: on VI schedules matching ≈ maximizing, so a working maximizer would also show slope≈1
  - resolved by: adding a concVR / melioration-trap schedule where the two equilibria provably differ
```

## References

Thorndike (1911) *Animal Intelligence*; Skinner (1938) *The Behavior of Organisms*; Herrnstein (1961) matching law; Herrnstein & Vaughan (1980) melioration; Baum (1974) generalized matching law; Bush & Mosteller (1955) *Stochastic Models for Learning*; Sutton & Barto (2018) *Reinforcement Learning* (Ch. 1 history; §2.8 gradient bandit); Loewenstein & Seung (2006) matching as a fixed point of covariance-based plasticity.
