# LIRF disagreement gate (`configs/lirf_gate_v1.json`, declared 02:05 UTC 29 September 2026): selected `u600_m600`, v45 built

## Where it came from

July 2026's 15,515 LIRF non-population rows all sit on the *unmatched* LIRF blend (v9 0, network 0.583, tree 0.417) because adsb.lol
has no LIRF ground coverage that month. The natural development months never put LIRF there in summer (July 2025 is 94% matched), so
that regime is scored only by the coverage campaign's July 2025 set thinned to July 2026's coverage (`runs/coverage_regime/a13_jul`,
LIRF matched 0.6%), with the seasonal-fold LIRF trees (`runs/coverage_regime/measure/2025-07/predictions.parquet`) and the seasonal-fold
LIRF network (`runs/adsb_ctx/regime_check_lirf/2025-07.parquet`) re-predicted on the thinned inputs.

A first re-read of the unmatched blend weights there showed a −60% least-squares gain that was a single one-day row (label 87,002 s:
v9 86,638, network 4,919, tree 42,425). With labels below 20,000 s the shipped blend reads 425.5 s against 438.4 s for v9 alone, so
the weights are right and that idea is closed. What the same rows showed is the network's tail compression: label 8,208 s, network
1,984, tree 5,972, v9 6,103. Where the tree sits far above the network, the tree is usually right. The gate leans to the tree on
those rows only.

## Declared round

Gate: on a LIRF non-population scope, rows with `tree − network > T` take the leaning blend (unmatched: v9 0 / net 0.2 / tree 0.8;
matched: 0.15 / 0.2 / 0.65); every other row keeps v34's blend. Eight candidates (T 300/600/1200, scopes alone and combined, one
v9-leaning variant). Primary metric: squared error on rows with labels below 20,000 s (one-day and clock-day labels dominate any
unfiltered sum and are outside what the gate is for), paired-day 95% interval, 10,000 draws. Rule: thinned-July unmatched change at
most −1% with the interval below zero and no natural month's unmatched interval entirely above zero; matched part: natural-July
matched change at most −0.5% with the interval below zero and November/December not entirely above zero. Population rows excluded.

**Results (`results_v1.json`; module `src/prc2026_final/lirf_gate.py`, runner `tools/run_lirf_gate.py`, tests `tests/test_lirf_gate.py`):**

| candidate | thinned July, unmatched | natural July, matched | eligible |
|---|---|---|---|
| u300 | −2.05% [−4.37, +0.09], 444 rows | — | no |
| **u600** | **−2.68% [−4.95, −0.59], 98 rows, 18/31 days** | — | yes |
| u1200 | −2.73% [−4.98, −0.71], 42 rows | — | yes |
| u600_v9 | −2.60% [−4.76, −0.56] | — | yes |
| m300 | — | −1.14% [−2.24, −0.12], 424 rows | yes |
| **m600** | — | **−1.22% [−2.27, −0.23], 99 rows, 16/31 days** | yes |
| **u600_m600** | −2.68% | −1.22% | **selected** |
| u300_m300 | −2.05% (interval crosses zero) | −1.14% | no |

Natural unmatched scopes (no-harm checks, 2–11 gated rows per month): January −0.17% [−0.72, +0.23], November −0.24%, December
−0.01%, July +4.4% on 6 rows with a wide interval. Gated rows' RMSE: thinned July 1,699 → 1,470 s; natural July matched 1,535 → 1,442 s.

## v45 (built 02:20 UTC; NOT uploaded)

`tools/export_lirf_gate.py --version 45 --base 40` → `runs/submissions/vigorous-whistle_v45.parquet`, SHA-256
`72d24155e6b6b8dbb4d6caf7a590679d3b242ad7c83793dcc3d28d5b88fd77fa` (byte-identical to the dry run), report `export_v45.json`, audit
`docs/export_audit/v45_audit.json` clean. Base **v40** (the clean model line): its LIRF non-population rows hold exactly v34's blend
(max 0.5 s rounding on the gated rows). 100 rows change, all LIRF: January 26 (mean +240 s), July 74 (mean +416 s, largest +1,126 s);
label-free firing rates 0.5% (July unmatched) against 0.7% in the thinned simulation. Sum of squared moves 1.8e7.

**Expectation, with its assumptions:** the development gated rows' per-row gain scaled to the ranking's 79 + 21 gated rows is
**−0.40 s**; if every move were noise, +0.12 s; the u300 precedent (a ranking-regime gate) transferred at ten times its development
mean, the tail stretch (a dev-fitted recalibration) at half.
The final-file runner gains a `v45` layer set only if the layer ships.

## v45 official: 224.2398 (01:29 UTC, 29 September) — −0.219 s against v40; the clean line's best

Uploaded at the user's word at 01:29:01 UTC, processed 01:29:47 UTC, 344,841 rows (`official_result_v45.json`). The gate transferred at
about half of its development expectation (−0.40 s): a model-based, declared, ranking-regime layer that paid on the ranking. Never re-upload v1–v45. No score enters any weight:
the realised transfer ratio is recorded as an outcome only.
