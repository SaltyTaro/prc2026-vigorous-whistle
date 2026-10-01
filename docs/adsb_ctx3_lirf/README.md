# LIRF context-network design (1 October 2026): no GO by the declared rule; masking helps LIRF where LIRF is dark

Declaration [`configs/adsb_ctx3_lirf_design_v1.json`](../../configs/adsb_ctx3_lirf_design_v1.json) (09:13 UTC, caf3836, before any fit;
amendment 09:16 UTC, 2d018de, before any new-run prediction: fold- and seed-matched baseline). Report
[`design_report.json`](design_report.json) (`tools/ctx3_lirf_design_report.py`). Nothing is exported or uploaded.

## Question

LIRF holds about 40% of the projected ranking error, and in July 2026 every LIRF row is unmatched (0.2% matched against 93.9% in July
2025). The non-LIRF networks moved to 32 epochs (v30) and gained a masked member routed to unmatched rows (v52); none of that reached
LIRF, whose network is still v29's round-1 `ctx_lirf` (16 epochs), and v52's member skips LIRF rows. Does a LIRF network built the new
way beat the shipped one on LIRF's ordinary rows, above all where LIRF has no ADS-B?

## Design

- Variants (`tools/run_adsb_ctx3.py --fold design`, the combination batch's 32-epoch reference with LIRF's non-population rows in
  fitting and monitoring): `lref32` (no masking) and `lm25_32` (masking 0.25), seeds 21–24, arcw0–3 CPU, two lanes per worker
  (09:13–12:58 UTC; each worker deallocated after its hash-checked mirror, `tools/mirror_ctx3_lirf.sh`).
- Items: **refresh** = lref32 on every LIRF non-population row; **refresh_mask** = lref32 on matched rows, lm25_32 on unmatched rows.
- Baseline B: the shipped family's four round-B design seeds (`ctx_lirf_s1..s4`, the same fit partition and October monitor).
- Readings (`tools/ctx3_lirf_predict.py`, the positional batch's predictor with LIRF rows only and a `--baseline` mode): October, and
  the design models' transfer readings on July 2025 natural and thinned to July 2026's coverage (coverage-regime attempt a13: LIRF
  dark) and December 2025. Every model reproduces its own October predictions (RMS ≤ 0.0002 s).
- L2 = 0.65 v9 + 0.35 X against the same with B (round B's blend), paired-day 95% interval.

## Result (declared rule: no GO)

| reading | rows (unmatched) | B | refresh | refresh_mask | L2 refresh | L2 refresh_mask | masking on unmatched (vs refresh) |
|---|---|---:|---:|---:|---|---|---|
| October | 14,027 (5,212) | 366.33 | 366.19 | 366.79 | −0.163% [−2.12, +0.97] | −0.101% [−1.56, +0.83] | +0.47% [−3.32, +6.54] |
| July natural | 14,856 (864) | 435.09 | 440.11 | 439.27 | **+0.475%** [+0.16, +4.05] | +0.402% [−0.09, +3.72] | −3.56% [−25.7, −4.5] |
| **July thinned (LIRF dark)** | 14,856 (14,771) | 441.42 | 445.17 | **437.30** | +0.118% [−0.88, +1.84] | **−0.467%** [−4.27, −0.02] | **−1.80%** [−14.7, −3.0] |
| December natural | 11,980 (2,474) | 298.65 | 304.48 | 302.26 | +0.329% [−0.46, +2.38] | +0.135% [−1.04, +1.75] | −2.55% [−16.4, −2.1] |

Standalone RMSE (s) on all LIRF non-population rows; December thinned equals natural (a13 thins EDDF only).

- **The 32-epoch base is not better than the shipped round-1 network on LIRF.** `refresh` is worse on natural July (interval above zero)
  and December. Neither item reaches the 0.3% October bar; both fail "not worse on natural July / December". **Nothing is selected.**
- **Masking helps LIRF's unmatched rows in every 2025 month read, most where LIRF is dark.** Against the same base it improves
  unmatched rows by 1.8–3.6% on July and December (intervals below zero). On thinned July — the July 2026 regime — `refresh_mask`
  beats the shipped four-seed family by 0.47% at the L2 level (interval below zero), and its four seeds read below even the shipped
  **16-seed** family (L2 421.01 against 422.70 s; reference, [`design_report.json`](design_report.json) `reference_16_seed_july`).

## Exploration after the reading (disclosed; not a selection)

The shipped network blended with, rather than replaced by, the masked network on unmatched rows only — X = B on matched rows,
(1 − w) B + w lm25_32 on unmatched rows — at the same L2 (scratch, no new fit):

| reading | w = 0.25 | w = 0.5 |
|---|---|---|
| October | −0.029% [−0.26, +0.06] | −0.035% [−0.44, +0.21] |
| July natural | −0.012% [−0.10, −0.01] | −0.020% [−0.17, +0.01] |
| July thinned (LIRF dark) | −0.213% [−1.51, −0.39] | **−0.362%** [−2.74, −0.51] |
| December natural | −0.033% [−0.25, +0.04] | −0.054% [−0.46, +0.12] |

Better on every reading, four times as large where LIRF is dark. This is the masked-member pattern of v52 carried to LIRF, and it is
the candidate for a separately declared development round on the clean line's footing (LIRF folds: leave-one-month-out January — LIRF
is naturally dark before May 2025 — and July with its thinned reading, the forward fold, the January–November refit, the twelve-month
refit), read inside the shipped stack's LIRF blends and gate. Nothing from this design ships directly.
