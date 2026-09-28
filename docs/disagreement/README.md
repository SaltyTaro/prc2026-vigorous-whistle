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

## LIRF lean-up on the stack (configs/lirf_up_v1.json, 15:03:38Z; tools/run_lirf_up.py -> lirf_up_v1_results.json)

Nothing selected: l300 January untouched, July -0.052% [-0.323, -0.019], November +0.001%, December -0.022% [-0.068, -0.009], mean -0.018%;
l600 mean -0.011%. 78 ranking rows at most (January). Not stacked.

Closed the same hour: an ADS-B "lower bound" (takeoff - segment_start on non-gate segments): 9% of labels (gate: 30%) lie below it, since
segment starts include earlier ground time; raising predictions to it multiplies SSE (+1e11 per month). The models already use these features.

## v35 official result (28 September 2026)

Uploaded at the user's "upload v35 now" at 00:57:18 UTC (the first attempt stopped before any write: the uploader required a
`selected_candidate` provenance field that the stack exporter does not write; tools/prc_submit.py now labels such exports by their
experiment id). Processed 00:57:59 UTC: **226.8989 on all 344,841 rows, -1.6944 s (-0.741%) against v34's 228.5933**, new official best.
The fair development months predicted -0.021 / -0.067 / -0.141 / -0.069%; the ranking gain is ten times the development mean, as the
label-free exposure analysis expected (January 2026 carries several times the development months' disagreement, EHAM 3-7 January). Official record: official_result_v35.json. Never re-upload v1-v35.

## Follow-up searches after v35 (28 September 2026, label-free on the ranking)

**Day-level ADS-B bias (closed).** Per airport-day, the median of (takeoff - motion start) - v9 on gate-tracked flights against the mean
residual of v33 on untracked flights of the same day, four fair months: slope -0.05 (per month -0.10 ... +0.36, correlations |r| < 0.15),
also -0.05 on the 59 most extreme days. It would not have found EHAM 3-7 January 2026 either: the tracked flights there show d9 -124 ... +87 s
while the untracked rows sit 300-660 s above v9. Binned by the observed taxi, EGLL July 2026 predictions lie 52 s below the 2025 label
relation on 9,601 gate rows (the helper's ~45 s, docs/adsb_shift), EDDF July +144 s on 784 rows, LSZH July +62 s on 2,393: all coverage
changes, each at most ~0.16% of SSE. Scratch scripts only (scratchpad daybias.py / egll_curve.py); nothing kept.

**Second disagreement pass** (tools/disagreement_audit2.py -> audit2.json; runs/disagreement2_{rank,dev}.parquet). Pairs: networks vs
committee (non-LIRF), LIRF network vs LIRF trees, residual stage vs v9 (matched). July 2026 has no excess on any pair. January 2026's
excess is EHAM (the known disruption: 141 untracked rows with nets > committee + 600) and LTFM 18-19 January (77 rows with nets < committee
- 600). Development side weights (least-squares weight on the second side, per sign and |D| bin, four months): nets above the committee
~1.0 in every month (supports mu600, 439 ranking rows); nets below the committee by > 1200 s 0.04 pooled, 0.53 December (vs 0.75 used;
~60 ranking rows); LIRF network below the LIRF trees by 600-1200 s -0.23 ... 0.27 in every month (vs 0.58 used on unmatched rows;
~100 ranking rows). The higher side wins again, but each touches <= ~0.1 s of ranking exposure.

**Snow days** (tools/snow_day_scan.py -> snow_days.json; METAR SN >= 8 or FZ >= 4 reports per UTC day). January 2026 is unusually
snowy: 45 snow airport-days, 18,931 ranking rows (6,221 untracked), against 33 snow days in the four development months. On the 2025
months the 12-month cross-fit v9 tracks snow where the airport's snow appears in other training months (EDDM +0, EHAM +14, LSZH -12 s)
but under-reacts at LTFM, whose 2025 snow all fell in February (+162 s mean, up to +485 s on 23 February; beta 1.19 overall). The full
v33 recipe on the fair months (out of sample): beta 1.087, +11 s on snow days (-2 s otherwise), **+56 s on untracked rows**, +4 s on
tracked rows, LFPG +68 s (17 January 2025 +110 s). LTFM 18-19 January 2026 (showers of snow and blowing snow) is predicted +754 / +1,011 s
over its median; LTFM's own snow response can only be read from February 2025, which no development month covers.

## Snow-day lift and higher-side gates (configs/snow_higher_v1.json, 01:31Z; tools/run_snow_higher.py -> snow_higher_v1_results.json)

Control: the v35 stack on the LOMO rows (January / July eleven-month footing, November / December fair); it reproduces the mix-gate
round's control exactly on November and December. **Nothing selected.**

| candidate | Jan (LOMO) | Jul (LOMO) | Nov | Dec | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| s10 | -0.011% | - | +0.014% | - | +0.001% | 0/2 |
| s20 | +0.023% | - | +0.062% | - | +0.021% | 0/2 |
| mu600 | -0.008% | -0.073% | -0.004% | -0.040% | -0.031% | 1/4 |
| nd1200 | -0.025% | -0.105% | -0.047% | **+0.020%** | -0.039% | 2/4 |
| lt600 | -0.003% | -0.085% | -0.012% | **+0.033%** | -0.017% | 0/4 |
| h3 | -0.036% | -0.264% | -0.063% | **+0.012%** | -0.088% | 1/4 |
| h3_s10 | -0.046% | -0.264% | -0.044% | +0.012% | -0.085% | 1/4 |
| h3_s20 | -0.010% | -0.264% | +0.011% | +0.012% | -0.063% | 1/4 |

- **Snow lift closed.** The fair rows' +56 s on untracked snow-day rows does not survive the eleven-month footing: s10 moves 1,555
  January rows for -0.011% (interval crosses zero) and worsens November; s20 worsens both. The residual was the fold side's shorter
  training, not a snow blind spot. LTFM's snow response stays untestable (its 2025 snow is February only), so a LTFM-only lift
  would be an unsupported bet.
- **Higher-side gates:** the networks-far-below (nd1200) and LIRF-network-far-below (lt600) gates win July (the month with the
  most such rows) but lose December. mu600 is better in every month again (8 of 8 month readings over this round and
  configs/mix_gate_v1.json) but has 1/4 intervals below zero. Ranking stake: 439 rows moved by about +200 s, at most about 0.13 s.

## Deep search for a ranking-only signal (28 September 2026, after the snow round)

Scripts: deep_search_20260928/ (scratch analyses; control_rows.py builds the v35-stack control on the LOMO rows, the others read it).
Everything below is label-free on the ranking; labels are read only on 2025 months. Nothing was declared or selected.

- **Where the error is.** On the LOMO / fair months the v35 stack's airport-day means carry 0.4-1.2% of SSE (so day-level
  corrections are nearly spent); 14-94 rows per month with |error| > 1 h carry 13-72% (January 26 rows = 72%). Of the 144 such rows,
  82 have no NM record, then LIRF one-day rows and schedule fills.
- **No-NM schedule fills outside LIRF: closed.** Over the twelve months, no-NM departures more than 1 h after schedule have their block
  within 15 min of schedule in 0-2% of rows (LTFM 11%, LIRF 50%); labels are ordinary (median ~1,000 s).
- **Arrival block-time source: closed.** The arrival fill rate (block = schedule) is inside its 2025 range at every airport in both
  ranking months. Runway shares: no ranking-month share leaves its 2025 range by 5 points. Month-level arrival taxi-in does not anchor
  departures (2025 correlations weak or negative; departures follow the de-icing season).
- **ADS-B observation trust: closed.** On 148k dev gate rows the least-squares weight on the raw observation beyond the stack is ~0 in
  every gap bin (also above +600 s). EGLL: dev residual by points per minute +4 ... +14 s; July 2026's density lies in that range, so
  the -45 ... -52 s against the 2025 label relation is a measurement change, not a model error (< 0.01 s). Stand stops before motion
  (`stop_seconds`) are in the label for 0-5% of long stops; residual ~0, weight 0 (already a feature).
- **NM AOBT_3 exactness: closed.** Block equals AOBT_3 within 1 s on 0.5-3% of rows (median |difference| 120-356 s).
- **Matched lean-up:** when the mix lies 150-300 s above the v9+residual side on matched rows the truth sits at the mix (weights 0.93 /
  1.01 / 0.52 / 1.02 against 0.6 used; 283-607 rows a month) -> another ~0.1 s higher-side item.
- **Peer congestion (new signal, mostly already exploited).** Per airport-hour, the mean NM proxy (MVT - AOBT_3, 0-7200 s, >= 5 peers)
  minus the airport-month median of hourly means. On the LOMO / fair months the stack under-predicts rows in hours > 300 s above
  normal in every month: no-clock untracked +32 s (300-600) / +137 s (> 600), no-clock tracked +168 / +286 s, clock untracked +13 /
  +67 s; the peer level itself carries no information beyond the stack (weights ~0). Cross-fit v9 on the other eight months: clock
  rows +58 s (600-1200) / +380 s (> 1200). January 2026 is outside the training range: 10.5% of rows in hours > 300 s (training 2.6-7.8%),
  1,052 no-clock rows (training at most 287 a month), 274 of them > 1,200 s (about 10 in the four development months). v35's layers
  already moved 462 of those rows (EHAM 377, +138 ... +301 s); the rest, 590 rows mostly at 300-600 s, is worth ~0.02 s on the
  development evidence. The extreme hours (EHAM 3-9 January, LFPG 3-7 January) have no development support.

## Higher-side bundle (configs/higher_bundle_v1.json 02:08Z, follow-up v2 02:09Z post-hoc; tools/run_higher_bundle.py)

Control: the v35 stack on the LOMO rows (reproduces the mix-gate round exactly on November / December). Peer congestion:
tools/peer_congestion.py. **Nothing selected in either round.**

| candidate | Jan (LOMO) | Jul (LOMO) | Nov | Dec | mean | intervals < 0 |
|---|---|---|---|---|---|---|
| mu600 | -0.008% | -0.073% | -0.004% | -0.040% | -0.031% | 1/4 |
| ml (matched lean-up 150-300 s) | +0.002% | -0.017% | **+0.050%** | -0.021% | +0.004% | 1/4 |
| pc (no clock, peer_c > 600: +100 s) | -0.001% | -0.009% | -0.042% | -0.016% | -0.017% | 0/4 |
| pcc (clock, untracked: +50 s) | -0.019% | -0.014% | -0.021% | -0.012% | -0.016% | 0/4 |
| bundle (all four) | -0.023% | -0.111% | -0.012% | -0.078% | -0.056% | 1/4 |
| **v2 mu_pc_pcc** | -0.026% | -0.096% | -0.066% | -0.059% | **-0.062%** | 2/4 (Jul, Nov) |
| v2 pc_pcc | -0.020% | -0.022% | -0.063% | -0.028% | -0.033% | 0/4 |

v2 is post-hoc (the three members were each better in all four months in v1; the parameters themselves came from exploratory
reads on the same months), so it is weaker evidence than a clean round. mu_pc_pcc is better in every month.

**Dry export** (tools/export_stack.py --queue 0.75 --queue-source regular --mix-up 600 --peer-lift --version 36 --dry-run DIR; the
patched exporter reproduces v35 byte for byte without the new flags): SHA-256 ed4bcd36..., independent rebuild within 1 s (204 float-noise
rows). Against v35: 2,388 rows change, all upward, mean +93 s, max +667 s, sum of squared change 3.57e7 (mu600 gate 439, u300 re-gated
541, no-clock lift 609, clock-untracked lift 1,476; LFPG 1,188, EHAM 510, LTFM 352, EGLL 134, EDDF 110). Ranking effect if the truth sits
at beta x the move: -(2 beta - 1) x 3.57e7 SSE, i.e. -0.23 s at beta 1, -0.46 s at beta 1.5, +0.23 s at beta 0; break-even beta 0.5.
Development implies beta ~1.3-1.4 for the lifts (+137 / +67 s residual against +100 / +50 s) and ~1.0 for mu600.
Report: dry_export_v36_mu600_pc_pcc.json. Not exported to runs/submissions; the user decides.
