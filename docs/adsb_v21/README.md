# ADS-B v21: learners fitted on LIRF's ordinary rows (LightGBM screen) and a residual seed bag (declared 18 September 2026)

Declaration `configs/adsb_v21_v1.json` (16:42 UTC, commit fd457c4; amended 16:55 UTC, commit
d214359, before any candidate was scored). Baseline: **v18's development predictions** (330.302 /
271.528 / 192.187 on Jan / Jul / Nov 2025, `runs/adsb_v20_v2/selection.json`). Rule, pre-registered:
not worse than v18 on any month, mean ratio at most 0.9998 (0.02%, the standing rule since
adsb_v20_v2), every month's paired-day interval below zero; lowest mean ratio wins; otherwise nothing.

## Why

The deep audit of 18 September (`docs/audit_20260918/README.md`) found that LIRF's non-population
departures (ordinary LIRF rows with an NM clock, 37,406 development rows, RMSE 357 s) carry 19.8% of
the incumbent's squared error on rows with labels up to six hours and are predicted by v9 alone —
no full-data learner was ever fitted on LIRF, because the fill population's day-long labels break
early stopping. This round fits learners on every airport with only the fill population (LIRF, no
AOBT_3/LOBT clock, no aircraft type: the v9 mixture's rows, 449 development / 383 ranking rows)
excluded, through the runner's new `--exclude-population` option (commit 056b625), and blends them
on LIRF rows only; every non-LIRF row keeps the v18 recipe. The same audit found the v12 residual
stage worth 0.95% and single-seeded; the declaration fits it as a five-seed bag
(`residual_model.seeds`, `fit_boosters` in `tools/run_adsb_v11.py`), so the reproduction candidate
`ens_v20_bag_q_all_70` measures the bag alone and can qualify.

## Learners (`runs/adsb_v21/`, arcw4, 16 threads, one chain)

| variant | recipe | seed |
| --- | --- | --- |
| lgb_lirf_s31 | timeline_v1 LightGBM, all airports, fill population excluded | 20260931 |
| lgb_lirf_cap_s32 | as s31 plus `--max-label 21600` | 20260932 |
| lgb_lirf_nm_s33 (amendment) | as s31 with `--target nm_residual` (label minus the NM clock) | 20260933 |

Ensembles: `ens_v21_lgb_lirf` (s31 + s32), `ens_v21_lgb_lirf3` (all three). Candidates: the
incumbent recipe with an extra blend `(1 - w) * v9 + w * table` on the LIRF scope, w = 0.3 / 0.5 /
0.7 for the ensembles, 0.5 for each learner alone.

## Amendment (16:55 UTC)

A preview of lgb_lirf_s31's seasonal fold on the LIRF non-population rows, made while the chain was
still running and before the selection tool had scored anything, showed two things:

1. A label-target learner cannot follow a day-long NM clock: the one-day LIRF row of July 2025
   (proxy 87,181 s, label 87,002 s) is predicted at 2,186 s where v9's clock-residual models give
   86,891 s — 72% of the class's July squared error in one row (the class's RMSE 441 → 818 s).
2. Without ADS-B the learner loses to v9 (January 2025, LIRF entirely unmatched: 293.6 vs 284.2 s);
   on matched rows it wins (July 2025, 94% matched: 429.7 vs 440.6 s; blend 0.5: 427.9 s, −2.9%).

The LIRF blend scope was therefore restricted to **`lirf_nonpop_matched`**: matched surface
segment, |NM proxy| and |schedule interval| both under six hours (all label-free and available on
the ranking rows). One unrestricted candidate (`lirf_ens_50_all`) is kept as the record of the
failure, and the NM-residual variant lgb_lirf_nm_s33 was added. Learners, residual bag, baseline and
rule are unchanged. Ranking scope: LIRF is 80% matched in January 2026 and 0.2% in July 2026, so a
selected candidate acts on January's LIRF rows.

## Results

(to follow)
