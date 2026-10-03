# Dig tunnel 8: a stronger component representation on top of stack_resid_v2 — no GO, closed (3 October 2026)

2025 development months only; nothing declared, exported or uploaded; no official score or leaderboard read. Scratch `runs/dig/stack3/`.

**Base.** `base_v2.py` reproduces stack_resid_v2's out-of-month development predictions exactly (January −0.0445, July −0.1249,
November −0.1801, December −0.3287% against `ctl_v55m.new`). Every reading below is the change of complete-month RMSE against that v2
base, with paired-day 95% intervals in percent. The equal-footing mean covers January, July and December. Two forms are screened:

- **stage 2:** a second three-seed LightGBM bag on label − v2, applied at w 0.5. LIRF_pop, the guard rows and v2's held rows stay.
- **replace:** v2's learner refitted with the extra columns and the same guard.

Bar: equal-footing mean ≤ −0.05%, no month worse, two intervals below zero.

**Feasibility.** All candidate inputs exist for the ranking refits.

| Input | Development source | Ranking source |
|---|---|---|
| e32 network seeds 401-416 | LOMO (Jan/Jul), forward (Nov), refit (Dec) | runner `networks/ens_ctx_e32` |
| Combination member seeds 601-616, masked member seeds 701-716 | member directories | member directories (ranking file) |
| Residual-stage parts adj2 / cat / rnet | `runs/lomo_dev/residual`, `runs/fold_residual` | runner `residual/residual.parquet` |

Ranking seed means match the runner's e32 table exactly. They match the combination and masked tables within 23 s and 4 s (the runner
re-predicted those). The spread distributions are close: e32 seed-std median 32.3 s in development against 32.7 s on the ranking.

| Candidate | Jan | Jul | Nov | Dec | Equal-footing mean |
|---|---|---|---|---|---|
| stage 2, v2's 46 columns (control) | +0.016 | +0.032 | +0.021 | −0.045 | +0.001% |
| stage 2, 46 + seed spreads (std / range / skew of e32, combination, masked; member-vs-member differences) | +0.003 | +0.003 | +0.008 | −0.057 | −0.017% |
| stage 2, seed spreads + chain level / discrepancy only | +0.003 | −0.027 | −0.008 | −0.028 | −0.018% |
| replace, 46 + seed spreads | −0.010 | −0.004 | −0.000 | +0.012 | −0.001% |
| stage 2, 46 + residual parts (adj2 − cat, rnet − trees, spread) | +0.013 | +0.029 | +0.013 | −0.035 | +0.002% |
| replace, 46 + residual parts | +0.001 | +0.001 | +0.007 | +0.008 | +0.003% |
| replace, 46 + airport and airport × match | −0.003 | −0.014 | −0.005 | +0.003 | −0.004% |
| stage 2, 46 + airport | +0.011 | +0.017 | +0.016 | −0.039 | −0.004% |
| stage 2, everything (46 + airport + parts + seed) | −0.002 | +0.001 | −0.001 | −0.060 | −0.020% |

**Reading.** v2 already holds what the stack's components say. Seed-level uncertainty, finer residual-stage splits and airport
interactions add at most −0.02%. The gain sits in December alone, its interval crosses zero, and January and July are flat or
worse. A second stage on v2's own columns finds nothing (+0.001%). Individual committee learners were not assembled: their ranking
tables exist (29 files in the runner's `leaves/leaf`), but their development tables would have to be rebuilt for all four months,
and every cheaper split was null. **Closed; nothing declared.**
