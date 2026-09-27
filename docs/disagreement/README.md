# Component-disagreement audit and the disagreement-gate round (27 September 2026)

## Label-free audit (tools/disagreement_audit.py -> audit.json)

For every row, D = fold side - v9 side: committee mix (0.25 committee + 0.75 context networks) minus v9 + residual stage on non-LIRF rows;
the LIRF blend column minus v9 on LIRF non-population rows. Ranking rows come from `runs/adsb_ctx/export_v30/ranking_values.parquet`; no ranking
label exists or is read. Development rows are the fair set (`runs/fair_dev/rows.parquet`) with v34's weights.

| | Jan 25 | Jul 25 | Nov 25 | Dec 25 | **Jan 26** | Jul 26 |
|---|---|---|---|---|---|---|
| share of rows with \|D\| > 600 s | 0.15% | 0.26% | 0.11% | 0.10% | **0.55%** | 0.09% |
| share with \|D\| > 1200 s | 0.02% | 0.04% | 0.01% | 0.02% | **0.15%** | 0.01% |
| p99.9 of \|D\| (s) | 723 | 915 | 619 | 604 | **1404** | 585 |

January 2026 is outside every development month; July 2026 looks like 2025. The excess sits at **EHAM on 3-7 January 2026** (the
disruption): 29.9% of EHAM's unmatched rows have |D| > 600 s (0.5-1.7% in any 2025 month), mean D +332 s. 397 of the 430 unmatched EHAM rows
on those days have **no NM off-block clock**: v9 (no delay signal without the clock) predicts 1,523 s, the fold side (airport-wide ADS-B
context) 2,476 s, v34 2,190 s. Gate-matched EHAM rows on the same days (clock present) are tracked by v9 (docs/error_map).

Per-D-bin projections of the ranking RMSE from development cells move by only +0.4..+3.4 s; they do not explain the unexplained ranking SSE
on their own.

## Who is right when the sides disagree (fair development rows, labels)

Least-squares weight on the fold side (v34 uses 0.7 unmatched / 0.6 matched / 0.85 LIRF matched):

| scope | D < -1200 | -1200..-600 | -600..-300 | 300..600 | 600..1200 | > 1200 |
|---|---|---|---|---|---|---|
| unmatched non-LIRF non-LTFM | 0.14 | 0.67 | 0.59 | 1.07 | 1.20 | 0.78 (14 rows) |
| matched non-LIRF non-LTFM | 0.03 | 0.15 | 0.37 | 0.98 | 0.32 (33 rows) | - |
| LIRF non-pop matched (column) | 0.16 | 0.55 | 0.78 | 1.11 | 1.33 | 1.15 |

When the fold side is far ABOVE, the truth is at or beyond it; when it is far BELOW on matched rows, v9 + residual is right. For unmatched
rows with D > 300, the weight is 1.11 / 1.11 / 1.21 / 1.18 by month on clock rows and 1.03 overall on no-clock rows; the closest analog, EHAM
5 January 2025 (snow), gives 2.06 on its 35 no-clock unmatched rows (label 2,007 s against fold side 1,576, v9 1,171).

## Declared round (configs/disagreement_gate_v1.json, 13:50:53Z; tools/run_disagreement_gate.py -> gate_v1_results.json)

Control v34 (fair). Unchanged rule. **Nothing selected.** (A code-path test on the shipped-provenance rows printed candidate lines
before the fair run; never a result, disclosed.)

| candidate | Jan | Jul | Nov | Dec (equal footing) | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| u600 | -0.003% | -0.027% | -0.119% | -0.023% | -0.043% | 1/4 |
| **u300** | -0.010% | -0.055% | -0.117% | **-0.046%** [-0.163, -0.001] | -0.057% | 2/4 (Jul, Dec) |
| u600_m600 | -0.077% | -0.207% | -0.104% | +0.024% | -0.091% | 1/4 |
| u300_m300 | -0.078% | -0.211% | -0.137% | +0.060% | -0.091% | 1/4 |
| u600_m600_l600 | -0.077% | -0.244% | -0.105% | +0.017% | -0.102% | 1/4 |
| u300_m300_l300 | -0.078% | -0.262% | -0.136% | +0.038% | -0.109% | 1/4 |

The matched lean-down half wins only in the three months biased toward the v9 side (docs/fair_dev, fold asymmetry) and loses December: an
artefact, not a lever. The unmatched lean-up half (u300) is better in every month including December, like v34's lu00 was (-0.036%, 1/4
intervals; it transferred at -0.057%).

## u300 on the ranking (label-free exposure)

758 rows (675 January, 83 July), all raised: mean +244 s, max +709 s; EHAM 417 rows (+313 s, 408 without a clock), LFPG 152, EDDF 101.
Sum of D^2 over the touched rows is 6.7e8 (development: 2.4e8 over four months). With y = v9side + beta * D, the change in ranking SSE is
-sum D^2 * (0.6 beta - 0.51): break-even beta = 0.85; beta 1.0 -> -0.38 s, 1.1 -> -0.64 s, 1.2 -> -0.90 s; beta 0.7 -> +0.38 s.

Scratch export (tools/export_disagreement_gate.py --candidate u300 --dry-run): v31/v33/v34 floats round exactly to the uploads, 758 rows
change, independent rebuild within 1 s on 199 rows (float noise). Not uploaded; the rule did not select it (user decision).

## Round E (configs/adsb_ctx_v6.json) closed the same hour

Scored at 14:00 UTC after all 16 lanes' scoring folds: nothing selected (e_e32only mean -0.015%, July +0.051%, December -0.058%).
docs/adsb_ctx/development_results_v6.json.
