# LFPG EOBT_1 sensitivity (27 September 2026). Descriptive; nothing selected, exported for upload, or committed by the run

**Question.** In July 2026, LFPG's EOBT_1 changed meaning (see [`README.md`](README.md), finding 1). How much does that move v30, and
which way is right?

**Input.** `runs/final_lfpg_eobt/input/final.parquet` is a copy of `data/raw/ranking.parquet` with one edit. For the **7,510** LFPG
July-2026 departures (of 21,743) with EOBT_1 > IOBT, EOBT_1 is set to IOBT. The changed rows' EOBT_1 − IOBT was +300 s on 1,248
rows, +600 s on 2,814 rows and some other positive value on 3,448 rows. Every other value is byte-identical. The record is
`runs/final_lfpg_eobt/input/modification.json`.

The edit also undoes the genuinely late first filings. At 2025 rates that is about 6–7% of LFPG rows, so roughly 1,400 of the
7,510; this is a known cost.

**Run.** `tools/final_predict.py run --root runs/final_lfpg_eobt/run --threads 12 --workers 8 --version 901 --detail-ensembles
--vocab-cache runs/final_dryrun/run/leaves/vocabularies.json` ran from 06:43 to 08:21 UTC, about 1 h 38 min. The slowest stages were networks (4,176 s), adsb (617 s) and leaves (424 s).
Free RAM was never below 2.7 GB at the 9-minute checks.

**Comparison.** `tools/lfpg_eobt_sensitivity.py` → [`lfpg_eobt_sensitivity.json`](lfpg_eobt_sensitivity.json). It compares against
`runs/final_dryrun/run` (same code, models and machine), so float noise cancels.

## What moves

- **Features.** 22 columns change, only on LFPG July rows:
  - the EOBT_1 columns on the 7,510 edited rows (`nmclock_aobt_minus_eobt1_s`, `movement_minus_eobt1_s`, `eobt_*`, planned duration,
    `flight_duration_minus_plan_s`);
  - the EOBT ground-demand counts `gd_*_eobt_*` on 16,900–18,200 LFPG July rows.
- **No spill-over.** All 323,098 other rows are identical at every stage (max |Δ| = 0.0).

Mean per-row change in the prediction (modified − original, s) at each stage:

| stage | edited +300 | edited +600 | edited other | unedited LFPG July |
|---|---:|---:|---:|---:|
| v7 / v9 | +51 | +94 | +123 | 0 |
| committee `ens_v20_bag_q_all` | +50 | +92 | +117 | 0 |
| context nets e24 / e24b / e32 | +56 | +90 | +88 | +17 |
| tabular nets (nn, nnl, new) | +45 | +84 | +82 | 0 |
| residual stage | −0.8 | −1.8 | −3.6 | 0 |
| committee mix | +55 | +91 | +95 | +13 |
| **v30** | **+53** (RMS 60) | **+91** (RMS 101) | **+102** (RMS 134) | **+8.9** (RMS 14) |

- The unedited rows move only through the context networks' neighbour inputs.
- By ADS-B status, v30 on the edited rows moves +87 s when dark, +91 s when unmatched and +96 s at the gate (13 rows). LFPG July has
  only 51 gate rows.
- By carrier, v30 across all LFPG July rows moves: AFR +38, DAH +39, FDX +38, TAY +38, EJU +23, EZY +23 s.
- Total Σd² = **9.79e7**, which is 0.54% of v30's ranking SSE and 13% of the gap to first.

## Which way is right: 2025 delay-conditional bounds

`tools/lfpg_eobt_hypotheses.py` → [`lfpg_eobt_hypotheses.json`](lfpg_eobt_hypotheses.json).

LFPG predictions follow the NM proxy (MVT − AOBT_3). In the 2025 development labels (four recipe months), the label minus the proxy
grows with the **unannounced** part of the delay:

| total delay (AOBT − IOBT) | label − proxy, EOBT_1 = IOBT | label − proxy, EOBT_1 > IOBT ("announced") |
|---|---:|---:|
| 600–900 s | +142 s | +27 s |
| 900–1800 s | +228 s | +101 s |

v30 is calibrated on both 2025 groups (residuals +3 s and +6 s).

In July 2026, v30 − proxy on the edited rows is −36 s (+300) and −33 s (+600) as shipped, and +17 s and +58 s after the reset.

**Calibration check.** On the unedited July-2026 rows, the 2025 delay-conditional expectation sits within 2 s of shipped v30.

Expected label = proxy + the 2025 bin mean, under each reading. The table gives the bias-SSE change against shipped v30 on the edited
rows, as a share of ranking SSE (1% ≈ 1.15 s RMSE):

| hypothesis about the July-2026 EOBT_1 | expected − shipped | full reset | 75% | 50% |
|---|---:|---:|---:|---:|
| not a real announcement: labels follow total delay, as for 2025 rows with EOBT_1 = IOBT | +189 s | **−1.15%** (July-only table −1.01%) | −0.96% | −0.71% |
| announcement, behaving like 2025 late filings at the same total delay | +54 s | −0.20% | −0.25% | −0.23% |
| announcement, labels follow the remaining unannounced delay (the shipped model's own logic) | −8 to +17 s | **+0.66 to +0.82%** | +0.40 to +0.52% | +0.20 to +0.28% |

The unedited rows add between −0.07% and +0.18% of SSE, depending on the table.

**Reading.**

- The 2026 EOBT_1 on the edited rows sits within a median 120 s of AOBT_3. The 2025 late filings sat 510 s away. This looks like an
  update made close to off-block, not an early announcement, which favours the first two rows of the table.
- The third row is the case where the shipped model is right. A **50% blend** gains −0.71% / −0.23% of SSE under the first two
  hypotheses, about −0.8 s / −0.3 s. It loses +0.20 to +0.28% (+0.2 to +0.3 s) under the third.
- There are no labels for this regime. The development folds can't test it, because 2025 has no such EOBT_1. So any fix is a
  judgement call about uploading, and it belongs to the user.
- The rebuild's export (`runs/final_lfpg_eobt/run/export/vigorous-whistle_v901.parquet`) is a sensitivity artifact only. It is not a
  submission candidate as it stands: it is not audited and it carries the dry-run float noise.
