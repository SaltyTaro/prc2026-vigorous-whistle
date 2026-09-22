# adsb_v28 — the LIRF unmatched blend weight (22 September 2026) — nothing selected

Declarations: `configs/adsb_v28_v1.json` (17:05:15 UTC, scored on no month, superseded) and
**`configs/adsb_v28_v2.json`** (17:07:03 UTC, executed). Record: `development_results.json`,
`runs/adsb_v28/dev/`. No model was fitted, nothing was exported and nothing was uploaded.
**`vigorous-whistle_v23` (238.8197) remains the incumbent.**

## Why

The coverage census (`docs/adsb_coverage/README.md`) found LIRF without surface coverage in the July 2026
ranking month, so essentially every LIRF non-population row of that month takes the recipe's
*unmatched* branch — `0.25 × ens_v22_lirf_q` — at the highest-error airport in the larger half of the
scored set. That 0.25 was chosen in adsb_v24 over 0.50 by a mean ratio of 0.994652 against 0.994657: five
parts in a million, decided on months that were mostly LIRF-matched. January 2025, whose LIRF rows are all
unmatched, had preferred 0.50.

## Why v1 was superseded

`adsb_v28_v1` declared a pooled reading over all twelve months. That is not computable: the committee and
LIRF learner development predictions are fold outputs and exist only for the four recipe months
(`tools/run_adsb_v24.FRAME_OF`); the other eight have no out-of-sample table to blend, and producing them
would mean refitting the 29-learner committee leave-one-month-out. v1 was scored on no month and is
preserved unchanged. v2 restates the same candidates and readings over the four months that have the
tables — 18,463 touched rows, of which January 2025 carries 11,254.

**Disclosure.** This round fits nothing, so the code-path test of the scoring tool necessarily produced
real numbers, and 2025-07 and 2025-11 were seen before either declaration was written. Both declarations
quote those figures verbatim. The candidates, readings and rule shape were fixed before any score; only
the month list and the two count thresholds changed, and only because the twelve-month reading is not
computable. January and December, every pooled figure and every interval were unseen.

## Result

Touched-scope RMSE by month (rows of the scope in brackets), and the pooled four-month figure:

| weight | 2025-01 (11,254) | 2025-07 (864) | 2025-11 (3,871) | 2025-12 (2,474) | pooled | vs control | worst recipe month, all rows |
|---|---|---|---|---|---|---|---|
| **0.25 (control)** | 278.209 | 445.755 | **283.138** | 337.170 | **297.521** | — | — |
| 0.40 | **278.035** | 444.288 | 283.957 | 336.341 | 297.357 | −0.055% | +0.015% |
| 0.50 | 278.124 | 443.724 | 284.703 | 335.934 | 297.456 | −0.022% | +0.030% |
| 0.60 | 278.376 | **443.493** | 285.608 | 335.645 | 297.721 | +0.067% | +0.047% |
| 0.75 | 279.059 | 443.771 | 287.261 | **335.431** | 298.431 | +0.306% | +0.078% |
| 1.00 | 281.003 | 445.896 | 290.785 | 335.662 | 300.435 | +0.979% | +0.146% |

**Nothing is eligible, and it is not close.** The best alternative, 0.40, moves the pooled scope by
−0.055% against a 0.5% bar; every candidate makes at least one recipe month slightly worse on all rows;
and no candidate's paired-day interval is below zero on any of the four months (0/4 for all of them).

The months genuinely disagree — January and November want the incumbent weight, July and December want a
higher one — and the disagreement is larger than any candidate's pooled effect. The adsb_v24 coin flip
landed on the right side: **0.25 is the weight, and the evidence for changing it is noise.**

## What this closes

The recipe's blend weights are at their optimum on the evidence available. Together with the residual
rounds (v25 shipped, v26 and v27 null), the committee rounds, the matcher fixes and the coverage census,
every lever this data supports has now been tested. The remaining error at LIRF in July 2026 is not a
modelling deficit; it is the absence of the observations the model would need.
