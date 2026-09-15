# v10 accumulation plan (written 15 September 2026, before execution)

Official best: **v9 = 267.4454** (12th of 141 at 05:03 UTC; third 246.626).
V9 = frozen v7 models + LIRF turnaround mixture (`docs/lirf_mixture/`) + −12 s
floor. The user approved pursuing v10 after compacting the session. Nothing
below has been declared or run yet.

## Goal and honest expectation

Accumulate the remaining small, two-fold-validated items into one declared
candidate. It cannot reach top 5 (260.4) or top 3 (246.6); say so in reports.

## Items, with the evidence already on disk

1. **`cat_long` at 25% on the clock route** (ladder campaign, `runs/azure_arcw0_ladder/*_cat_long`).
   Fixed blend `0.75·v7 + 0.25·cat_long` on clock-available rows scores
   322.152 / 220.714 vs v7 322.819 / 221.138 (−0.207% / −0.192%); the SSE
   gain sits in ordinary rows (seasonal −1.53e8, forward −3.47e7) with zero
   effect on one-day rows (`docs/lirf_mixture/README.md`, session scratch
   `stack4.py`). Needs a **Jan–Nov refit on arcw0**: variant `cat_long` in
   `configs/ladder_v1.json` (CatBoost depth 8, lr 0.055, l2 10, rsm 0.9,
   subsample 0.8, Plain/Bernoulli, `nm_residual` target, features
   `core_weather_nmclock`, bundle `72e86725e23367ede5b558027cfc8eeb7ef6cfbaed23980537766ef6a34a7f34`),
   **9,929 trees** (median of best iterations 9,892 seasonal / 9,966 forward),
   fitting the 1,898,549 clock-present Jan–Nov rows like the v6 refit. Dev
   fits took 1,287 / 1,636 s wall and 10–12 GB RSS at 16 threads; expect
   ~35 min and ~14 GB. Check arcw0 free disk first (about 1.2 GB at last
   check; the .cbm will be large). Ladder source is frozen; a refit helper
   belongs in a new package, following `tools/launch_nmclock_refit.py` /
   `src/prc2026_surface_refit/` as the pattern.
2. **Negative-proxy routing** to the v5 no-clock pipeline (`docs/postprocess/`,
   rule `route_negative_proxy`): −0.13% / −0.05% on the folds, 169 ranking
   rows; v5 ranking predictions are the frozen
   `runs/azure_arcw0_weather/20260914T024435836344Z_refit_cat_weather_v4_100/ranking_predictions.parquet`.
   Its 2026 effect is unknown (v8's decomposition left 0.41e9 SSE
   unattributed); keep it as a separately declared candidate so the folds
   decide, never as a bundled assumption.
3. **LIRF clock-present fill mixture** (optional, ~0.2%): among clock-present
   LIRF rows with schedule interval > 3600 s, 6.2% are schedule fills;
   separators are NM delay (AOBT−SCHED in 1800–3600 s: 9% fill), no stand
   conflict, different-airline previous arrival (session scratch `ext1.py`).
   Only worth declaring if the two-fold gain clears 0.5% together with the
   other items; otherwise drop.
4. **EGLL/LTFM/LFPG fills** (optional, ~0.2–0.3%): 101 / 51 / 21 clock-present
   fill rows among y>1h rows in Jan–Oct; not yet prototyped.

Do **not** touch: the 11 long-schedule LIRF rows (> 43200 s; any change there
would be leaderboard-derived after v8), the LIRF population classifier (at its
ceiling: extra stand features and a mean component gave 314.39 / 219.75 vs
314.23 / 219.88), rotation features in the main models (residual correlation
|r| < 0.07 everywhere), December labels, per-airport weights.

## Procedure (same discipline as v9)

1. Declare `configs/v10_v1.json` with parents hash-bound (v9 export, its
   classifier, the surface refit ranking predictions, the v5 refit
   predictions, the ladder cat_long runs), fixed candidates (each item alone
   and their union), the 0.5%-on-both-folds rule, and disclosure of the
   scratch evidence above.
2. Development: reconstruct v9 exactly (v7 via `tools/run_postprocess.py:reconstruct_fold`,
   then the mixture via `tools/run_lirf_mixture.py` functions), add each item
   from saved fold predictions, score both complete folds, paired days.
4. Never re-upload v1–v9.

## Constraints that still apply

Azure only (arcw0 running; arcw1–arcw7 deallocated); no AWS; December
reserved; no leaderboard probes or leaderboard-derived rules; report official
results plainly, including losses.
