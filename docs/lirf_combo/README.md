# LIRF matched-row member (2 October 2026, after v54): the declared rule selects `mw40`

Declaration [`configs/lirf_combo_reread_v1.json`](../../configs/lirf_combo_reread_v1.json) (03:05:33 UTC, 1608a9d, before any
candidate score), scorer `tools/run_lirf_combo.py` (same commit), report [`results_v1.json`](results_v1.json). No new fit.

## Question

v53 / v54 blend the masked LIRF context network (`lm25_32`, 16 seeds 801–816) into the shipped LIRF network only on LIRF's
non-population rows **without** an ADS-B match. Outside LIRF, v51's combination member gained on matched rows by adding a second network
that differs from the pool. Does the same masked LIRF network, blended in on LIRF's **matched** non-population rows, help there:
network' = (1 − w) ens_ctx_lirf + w M inside v34's matched blend (v9 0.15 / network 0.40 / tree 0.45) and v45's matched gate m600?

Disclosed beforehand: the 1 October design report read four design-fold seeds of lm25_32 alone on matched LIRF rows (October 370.8 against
374.3 s, July 433.7 against 436.7 s, December 283.2 against 283.6 s for the shipped family's round-B seeds). The 16-seed development-fold
member had never been read on matched rows. The weight grid follows the unmatched round (0.25 / 0.40 / 0.55) plus 0.70.

## Sets and control

LIRF non-population matched rows (`lirf_m` of `runs/lomo_dev/rows.parquet`): July 2025 (eleven-month LOMO, 13,992 rows), November 2025
(forward fold, 7,401), December 2025 (January–November refit, 9,506; the month next to January 2026, the only ranking month with LIRF
ADS-B coverage). January 2025 has no matched LIRF row. Control check: every month's `lirf_net` equals the shipped network's 16-seed fold
mean (maximum 1.4e-13 s RMS).

## Result (primary: route rows with labels below 20,000 s; paired-day 95% interval)

| candidate | July 2025 | November 2025 | December 2025 | eligible |
|---|---|---|---|---|
| mw25 | −0.039% [−0.17, +0.10] | −0.140% [−0.39, +0.19] | −0.399% [−0.61, −0.21] | yes |
| **mw40** | −0.008% [−0.22, +0.23] | −0.348% [−0.69, +0.02] | **−0.630% [−0.97, −0.32]** (25/31 days) | **yes, selected** |
| mw55 | +0.052% [−0.24, +0.36] | −0.494% [−0.89, −0.10] | −0.927% [−1.39, −0.49] | no (July above 0) |
| mw70 | +0.056% [−0.29, +0.40] | −0.610% [−1.06, −0.16] | −1.121% [−1.71, −0.56] | no (July above 0) |

Standalone network RMSE on the primary rows (shipped / member): July 421.4 / 423.1 s, November 285.1 / 282.8 s, December 265.0 / 261.5 s.
The route's change as a share of the whole month's squared error (rows table's v33 column, scale reference): mw40 July −0.001%,
November −0.036%, December −0.089%.

**The declared rule selects `mw40`** (lowest mean of the three changes among eligible candidates). Stronger weights gain more in November
and December but turn July above zero; July is the month where the member alone is worse than the shipped network.

## Export (dry run, nothing uploaded)

`tools/export_lirf_combo.py --dry-run runs/lirf_combo/export_dry --version 55`: v54's model part
(`runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet`, hash-bound) with the ranking's 8,781 `lirf_nonpop_matched` rows (January 2026
8,747, July 2026 34) recomputed from the refit12 member at w 0.40: value = base + (candidate − control), floor −12. v54's model part holds
the control value on the route (3 rows above 1 s, max 3.7 s). 7,724 rows change, mean +0.02 s, mean absolute 3.7 s, max 134 s, squared
change 4.2e5; no row outside the route changes. SHA-256 `266b8c936f2625578e4980f1b11c3a26cc63594a13916510fa3fe86925ec800f`. The
reconciliation with the published daily totals follows last (`tools/export_daily_totals.py --base … --dry-run runs/daily_totals/dry_v55
--version 55`).

Expectation: January 2026's matched route holds about 8,747 × 285² ≈ 7.1e8 of squared error; December's −0.63% would be about −4.5e6,
about −0.03 s on 221.87 s. A small gain; building v55 and uploading it are the user's decisions.

Reconciled dry run (`runs/daily_totals/dry_v55/vigorous-whistle_v55.parquet`, 2 min 53 s), SHA-256
`d1c390d41225e22862c4536b7ee1ff7be1b42e7e046d52e592429460327d438b`: against the uploaded v54 only LIRF rows differ, 8,496 rows
(January 2026 8,420, July 2026 76; the reconciliation spreads the January days' LIRF change over the day's rows), squared change 4.6e5,
max 171 s. Not built under `runs/submissions`, not uploaded. To build v55 for real (only at the user's word): rerun the reconciliation with
the same `--base` and no `--dry-run`, compare byte for byte with the dry run, `tools/audit_export.py --version 55 --compare 54 53`.
