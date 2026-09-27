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

## Direct ADS-B check of u300 and the callsign fix (27 September, 14:00-15:30 UTC)

The 397 EHAM no-clock unmatched rows of 3-7 January 2026 have **no NM flight record at all**; the matcher keys them on `FLIGHT_mvt`, which
carries the IATA designator (`KL1787`) while ADS-B callsigns use the ICAO one, so they never match. Rematching them against the
raw adsb.lol surface files (label-free):

| subset | rows | observed ADS-B taxi | v9 | fold side | v34 | implied beta |
|---|---|---|---|---|---|---|
| IATA->ICAO mapped callsign, gate segment | 64 | 2,456 | 1,586 | 2,473 | 2,206 | 0.94 |
| takeoff-time match, unique candidate, gate | 205 | 2,852 | 1,598 | 2,595 | 2,296 | 1.16 |

(beta from observed taxi + 50 s; the takeoff-time method picks the callsign-matched aircraft only 76% of the time on a validation set,
so the mapped-callsign row is the precise one.) Both clear u300's break-even beta 0.85.

**Callsign fix round** (configs/callsign_fix_v1.json, 14:32:25Z; tools/adsb_rematch_nonm.py on arcw5 / arcw2 / arcw1;
tools/run_callsign_fix.py -> callsign_fix_v1_results.json). Scope: no-NM EDDF/EHAM departures whose existing key finds nothing but whose
mapped callsign yields a gate segment (development 38 / 87 / 35 / 38 rows per recipe month; ranking 115 January + 48 July). Observed taxi
offset from non-recipe months: EDDF +73.7 s, EHAM +6.5 s. Control v34 (fair). **Nothing selected.**

| candidate | Jan | Jul | Nov | Dec | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| cs50 | -0.004% | -0.012% | -0.016% | -0.011% | -0.011% | 4/4 (mean below the 0.02% bar) |
| cs75 | -0.004% | -0.006% | -0.019% | -0.013% | -0.010% | 3/4 |
| u300_cs50 | -0.014% | -0.067% | -0.136% | -0.057% | -0.068% | 2/4 |
| u300_cs75 | -0.015% | -0.060% | -0.139% | -0.058% | -0.068% | 3/4 (January fails) |

On the scope rows themselves v34's RMSE is 227 / 227 / 296 / 203 s by month; the 50% blend 143 / 127 / 191 / 118 s. The largest ranking
changes are EHAM disruption flights with 6-10 ADS-B stops and 1-1.4 h stationary (de-icing queues; observed taxi 3,500-7,400 s against v34
1,100-3,000 s); one July 2025 row shows the risk (ADS-B 4,778 s, v34 1,234 s, label 2,683 s), so the 50% weight is the robust choice.

Scratch exports (tools/export_callsign_fix.py --dry-run): u300_cs50 SHA f47a5c75..., u300_cs75 SHA 522f1653...; 861 rows change against
v34 (758 gate + 163 callsign, 60 overlap), v31/v33/v34 floats round exactly, independent rebuild within 1 s on 199 rows. Not uploaded.

## Screens closed and the queue-trust round (27 September, 15:00-15:30 UTC)

- **Callsign-free matching** (takeoff time + departure from the row's learned stand, four ranking days, 21,934 departures): 98.6% precise
  against callsign matches but only 36 new matches in four days; 5,886 of 6,614 key-unmatched rows have no ADS-B takeoff at all
  (LTFM, LFPG, LIRF July dark). Closed.
- **Extreme ADS-B taxi on regular matches**: rows whose ADS-B taxi exceeds v34 by > 1,500 s are artefacts (median 0 stops, 201 m path;
  the label follows v34 in 98%). Not a lever by itself.
- **Queue shape** (twelve 2025 months, gate rows with ADS-B taxi > 3,000 s): >= 5 stops over >= 5 km -> label within 120 s of the ADS-B
  taxi in 84% (non-queue 19%); with <= 75% stationary: 39 rows, ~95% within 120 s, none short by > 1,000 s.

**Queue-trust round** (configs/queue_trust_v1.json, 14:55:55Z; tools/run_queue_trust.py -> queue_trust_v1_results.json). Control: the
u300_cs50 stack. Scope rows: January 13, July 1, November 14, December 9. **Nothing selected.**

| candidate | Jan | Jul | Nov | Dec | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| qt50 | -0.006% | +0.005% | -0.004% | -0.010% | -0.004% | 3/4 |
| qt75 | -0.007% | +0.010% | -0.005% | -0.012% | -0.004% | 3/4 |
| qt100 | -0.008% | +0.016% | -0.005% | -0.013% | -0.002% | 3/4 |

On the regular-match scope rows the observed taxi (+ offset) has RMSE 78 / 68 / 19 s against the control's 442 / 219 / 369 s (January /
November / December). July's single scope row is a callsign-fix match (ADS-B 4,778 s, label 2,683 s): the uncertain part is the
IATA-mapped identity, not the queue logic. Ranking exposure: 161 regular + 25 callsign rows (EHAM January mostly); if the observed taxi is
right, -0.28 s (regular) and -0.17 s (callsign) at full weight.

**Stack exporter** (tools/export_stack.py): v34 -> u300 -> callsign 50% -> optional queue trust restricted to regular matches (a disclosed
post-hoc restriction after the July row). Dry runs: without the queue layer the file equals the u300_cs50 dry export (SHA f47a5c75...);
with queue trust 0.75 on regular matches SHA 75f4a4f4..., 1,022 rows change (u300 758, callsign 163, queue 161), independent rebuild
within 1 s. Not uploaded.

## Mix gate (27 September, 15:00-15:45 UTC)

Inside the committee mix, the same "higher side" screen: networks far above the committee -> LS weight on the networks about 1.0-1.2;
far below -> 0.06-0.64 in January / July but 0.8-0.95 in November / December at LTFM, EHAM and the other airports (the seven-month fold
networks are weaker than the trees; December's January-November networks are not). **Declared round** configs/mix_gate_v1.json
(15:00:52Z; tools/run_mix_gate.py -> mix_gate_v1_results.json), control = the current stack. **Nothing selected.**

| candidate | Jan | Jul | Nov | Dec | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| mu300 | +0.015% | -0.127% | -0.045% | -0.049% | -0.052% | 1/4 |
| mu600 | -0.004% | -0.037% | -0.004% | -0.040% | -0.022% | 0/4 |
| md300 | -0.113% | -0.312% | -0.116% | **+0.403%** | -0.035% | 1/4 |
| mud300 | -0.099% | -0.440% | -0.161% | **+0.354%** | -0.086% | 1/4 |

The lean-down half is the data-quantity artefact (December rejects it by +0.4%). mu600 is better in every month but weak; on the ranking it
moves 439 rows by about +200 s (sum of squared moves 2.0e7, at most +-0.13 s). Held out of the stack; to be re-read on the LOMO January / July
rows (eleven-month networks and committee). The stack itself (v34 + u300 + callsign 50% + queue trust 0.75 on regular matches) against v34 on
the fair rows: -0.021% / -0.067% / -0.141% / -0.069%. tools/export_stack.py gains --mix-up (dry run with 600: SHA 61e98464..., 1,268 rows).
