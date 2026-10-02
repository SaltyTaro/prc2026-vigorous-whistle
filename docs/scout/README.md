# Scout lane, 2 October 2026: the published daily figure and the reconciliation's day shift

Exploration on 2025 data only; **nothing declared, nothing
exported, nothing uploaded, no official score used.** v54 (221.8686) is unchanged as the official best.

## 1. What EUROCONTROL's daily figure is, exactly

`tools/scout_published_definition.py` → [`published_definition.json`](published_definition.json). Against the supplied 2025 clocks,
per airport and UTC off-block day, `DLY_ALL_PRE_2 × 60` (All_Pre-Departure_Delay, the source behind v48) is

    sum over departures of max(round((block − schedule) / 60), 0) × 60

| Airports | Flight set | Hinge | 2025 days reproduced |
|---|---|---|---|
| EGLL, LIRF, EDDM, LEMD, LTFM | all | per-flight minute rounding | every day whose count equals ours: exact to the second |
| LEBL | **IFR only** | minute rounding | 359 of 359 equal-count days within 60 s |
| LFPG | all | minute rounding | 43 of 44 equal-count days exact |
| EDDF | all | exact seconds | median +112 s a day, robust sd 86 s |
| EHAM, LSZH | **IFR only** (`FLIGHT_RULE_mvt == "I"`) | exact seconds | median +139 / +73 s, robust sd 86 / 68 s |

- With VFR rows in the arithmetic (as in v48), EHAM's figure looks 2,343 s a day noisy and 86% of its days are off by more than ten
  minutes; v48 therefore gave it noise sd 2,412 s and wide share 0.86, i.e. it nearly ignored EHAM (ranking day shifts mean 0, sd 1.5 s).
  The ranking file holds 236 VFR departures at EHAM and 174 at LSZH.
- What remains comes with the count difference dn = `FLT_DEP_IFR_2` − our rows: LFPG's figure holds about two flights a day that are
  absent from the supplied rows (about 1,860 s each, heavy-tailed: EDDM 1 January 2025 has one missing flight with a 14.5 h delay).

## 2. The figure's own definition as the reconciliation's noise model — small, closed

`src/prc2026_scout/published_figure.py` (IFR-only arithmetic at EHAM / LSZH / LEBL; per airport offset c0 + mu |dn|, core sd
sqrt(s0² + |dn| s²), all from the other months), `tools/scout_daily_ifr.py --variant figure`, against the declared v48 run on the v46
development stack:

| Month | Change | Paired-day interval (s) |
|---|---|---|
| January 2025 | −0.0245% | [−0.19, −0.02] |
| July 2025 | +0.0015% | [−0.04, +0.05] |
| November 2025 | −0.0455% | [−0.16, −0.02] |
| December 2025 | −0.0175% | [−0.08, +0.04] |

Equal-footing mean −0.0135% (bar 0.02%), July flat: v48's rule would not accept it. EHAM −0.18 / −0.13 / +0.15 / −0.95%, LFPG better in
November (−0.39%), LSZH noisy. Twelve-month reading (v23-recipe predictor, all of 2025, against v48's procedure on the same table):
better in January −0.067%, July −0.107%, November −0.072%, December −0.065%, worse in the other eight months (+0.005% to +0.146%). Per
airport the figure model helps nearly everywhere except LIRF (+0.6% to +2.2% in most months): trusting the exact total at LIRF moves its
ordinary rows by discrepancies that its fill population carries. With an exact total the
reconciliation is limited by row-level noise, not by the figure's noise; this refinement is worth about 0.03 s. **Closed.**

## 3. The day shift is under-applied — real on average, not robust to exploit

On the development stack the labels confirm about 0.6 of the day shift the published total implies (least-squares slope of the moved
ordinary rows' mean residual on implied = −discrepancy / n_eff), while v48's posterior applies about 0.4 in every size band; on the
twelve-month table (corrected figure, dn = 0 days) the implied shift is nearly the truth at EHAM (corr 0.96, slope 0.90; 1.00 on its five
extreme days), EDDF 0.87, LEMD 0.83, LSZH 0.82, LFPG 0.74, LEBL 0.72, and unreliable at LIRF (corr 0.39: the fill population absorbs it).
The ranking months hold more extreme days than 2025 (17% of airport-days with |implied| > 40 s against 13%) and v48 applies 0.31 there
(EHAM 5–6 January 2026: implied +202 / +161 s, applied +0.2 s; LTFM 19 January: −231 against −26 s).

Every way of applying more of it fails at least one development month (all against the declared v48 run; scratch exploration in the
session scratchpad, tool variants in `tools/scout_daily_ifr.py`, prior in `src/prc2026_scout/shift_prior.py`):

| Variant | January | July | November | December | Failure |
|---|---|---|---|---|---|
| figure + tau × 1.5 | −0.028% | −0.096% | −0.048% | **+0.025%** | December |
| figure + scale-mixture prior (pi 0.03, k ~3.5, fitted on the fitting months) | −0.028% | −0.097% | **+0.269%** | **+0.242%** | month-end LTFM days (dn 62–78) and LFPG 31 Dec (missing flights) shifted in full |
| calibrated beta_a × implied, dn ≤ 2 | **+0.058%** | −0.173% | −0.046% | −0.056% | EDDM 1 Jan 2025 (one missing flight, 14.5 h) |
| calibrated, dn = 0 | **+0.028%** | −0.155% | −0.029% | −0.056% | January (LIRF 15 / 17 Jan carry half) |
| calibrated, dn = 0, only airports with fitting corr ≥ 0.9 | **+0.006%** | −0.148% | **+0.013%** | **+0.014%** | three of four months |

The last rule on the twelve-month table (other predictor, cross-fitted over eleven months): 8 of 12 months better, mean −0.052%; January,
May, November, December slightly worse; July −0.120% [−0.53, −0.14]. The mechanism pays in summer-type months and costs a little in
winter. A July-only use would be chosen after seeing these results and is not proposed as a declared candidate. **Closed as a general
rule.**

## 4. Open candidate: calibrated shift on extreme days only (not declared)

The same calibrated shift, applied only where v48's Gaussian prior bites: airport-days whose implied shift is at least 80 s, at airports
whose fitting-month correlation of need on implied is at least 0.9 (per fold; on all four months: EDDF, EDDM, EHAM, LEBL, LEMD, LTFM), on
days whose counts agree or whose count difference can only enlarge the discrepancy ("sign-safe": a published flight we lack adds delay to
the published total, a row of ours the figure lacks adds delay to the implied total). Everything else keeps v48. Scratch
`calib3.py` / `calib4.py` in the session scratchpad (to be moved into `tools/` if declared).

| Reading | Result |
|---|---|
| development stack (four months) | January +0.0005% (732 rows), July −0.093% [−0.42, −0.08], November 0, December −0.014% |
| twelve months, other predictor | no month worse; February −0.45%, April −0.13%, July −0.10%, October −0.04%, mean −0.068% |
| thresholds 60 / 100 s, correlation 0.85 / 0.95 (dn = 0 days) | worst month of 80 month-readings +0.0014%; at 60 s all twelve months better (mean −0.063%) |

On the ranking file (v54's model part with v48's reconciliation, label-free): 12 airport-days, 7,372 rows, sum of squared change 1.10e7,
mean change −16 s: about −0.07 s if 2026 behaves as 2025 (alignment 1), +0.07 s if the moves are noise. The largest item is EHAM
5 January 2026 (implied +184 s, v48 +0.2 s, candidate +138 s on 184 rows), the day the 27 September error map found under-predicted from
the ADS-B gate rows; the rest is LTFM's most extreme days (−50…−70 s against v48's −40…−50 s), EDDF 29 January (−94 against −1 s) and
EHAM 7 January (−72 against −1 s). Thresholds and gating were chosen in this exploration on all of 2025; no untouched 2025 data is left,
so a declaration would rest on the disclosed exploration as v48's did, with a rule written for a sparse layer (v48's "every interval
below zero" cannot hold for months without extreme days).

## Files

- `src/prc2026_scout/published_figure.py`, `src/prc2026_scout/shift_prior.py` (with π = 0 reproduces `solve_day` to 4e-11 s)
- `tools/scout_published_definition.py`, `tools/scout_daily_ifr.py` (the v48 variant reproduces the declared v48 run exactly),
  `tools/scout_export_daily.py` (ranking dry run of a variant; not run for an export)
- `tests/test_scout_published_figure.py`, `tests/test_scout_shift_prior.py` (10 tests)
- Runs: `runs/scout/` (not in the repository)

## 5. LTFM input drift 2025 → 2026 (label-free for 2026) — nothing actionable, closed

Scratch `ltfm_drift.py` / `ltfm_34l.py` in the session scratchpad. Not repeated here: the THY IOBT convention (docs/input_shift).

- AOBT_3 at LTFM is takeoff minus a fixed allowance per runway (about 1,140 / 1,200 / 1,440 / 1,500 s), in 2026 as in 2025; NM record
  share stable (98.5%); departures per day as in 2025.
- Runways: January 2026 uses 34L (4.3%) and 16R (2.4%), unused in January 2025 but used in July / December 2025 (4.6 / 5.2% for 34L).
  The 34L rows concentrate on the 18–19 January snow days (takeoff minus schedule median 3,358–4,017 s); the model's high predictions
  there (2,657–3,152 s) sit on disruption days, where the daily totals already pull LTFM down (v54 dry run: −34 s per row in January).
- Stands: 1 (January) and 5 (July) departures from stands unseen in 2025, against a 2025 leave-one-month-out baseline of 0–0.03%.
  Remote numeric stands carry 3,760 July 2026 departures against about 2,600 a month in 2025, all known stands; in July 2025 their mean
  taxi time was 1,117 s against 1,094 s overall.
- Takeoff minus schedule is much larger in January 2026 (median 1,615 s against 1,194 s in January 2025): winter delay, a level the
  reconciliation reads.

## 6. Declared round: extreme-day calibration — ACCEPTED by its rule (2 October 2026)

The user answered "YES" to declaring section 4's candidate. Declaration `configs/scout_extreme_shift_v1.json` (04:20 UTC, commit
9bfed66, before the governing run), thresholds fixed as found (80 s, correlation 0.9), selection history disclosed. Code
`src/prc2026_scout/extreme_shift.py`, `tools/scout_extreme_shift.py`, `tests/test_scout_extreme_shift.py`. Governing run →
[`extreme_shift_results_v1.json`](extreme_shift_results_v1.json):

| Reading (against v48's procedure) | Result |
|---|---|
| development stack | January +0.0005% (1 day), July −0.0929% [−0.42, −0.08] (7 days), November 0, December −0.0140% (1 day); equal-footing mean −0.0355% |
| twelve months, v23-recipe predictor (control = `runs/daily_totals/check12f`, reproduced exactly) | 8 months changed, all 8 better, none worse; February −0.45%, April −0.13%, July −0.10% |
| rule | no month worse than 0.005% ✓, mean ≤ −0.02% ✓, July's interval below zero ✓, twelve-month check ✓ → **accepted** |

The governing run confirms the exploration's arithmetic (section 4); its only independence is the twelve-month table's other predictor.

**Dry runs** (label-free, nothing under `runs/submissions`, nothing uploaded):

| Base (model part → v48's procedure → this layer) | File (SHA-256) | Against the reconciled base |
|---|---|---|
| v54: `runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet` + `runs/daily_totals/dry_v54` (= the uploaded v54 byte for byte) | `runs/scout/extreme_dry_v54base/vigorous-whistle_v56.parquet` (`78314a49be21e3be…`) | 7,372 rows, LTFM 6,495 / EDDF 503 / EHAM 374, S 1.098e7, mean −16.1 s, max 138 s |
| v55 candidate: `runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet` + `runs/daily_totals/dry_v55` | `runs/scout/extreme_dry_v55base/vigorous-whistle_v56.parquet` (`aa02ff597a619f22…`) | the same 7,372 changes (v55 differs from v54 at LIRF only) |

Receipts: [`extreme_dry_v54base.json`](extreme_dry_v54base.json), [`extreme_dry_v55base.json`](extreme_dry_v55base.json). Twelve
airport-days change: EDDF 29 January (−94 s day shift against v48's −1 s), EHAM 5 January (+138 against +0.2 s; dn +2, sign-safe) and
7 January (−72 against −1 s), and nine LTFM days (−51 … −70 s against v48's −30 … −52 s). Stake at v54's score: about −0.07 s if the
moves are right as in 2025 (alignment 1), 0 at alignment 0.5, +0.07 s if they are noise. The file version is assigned when the user
decides to build it (v55 if built on v54 alone, otherwise the number after the LIRF candidate).

## 7. v55 built (user: "build v55") and uploaded (user: "upload it"): 221.6941 official, the official best

- **`runs/submissions/vigorous-whistle_v55.parquet`, SHA-256 `aa02ff597a619f223888370726cdcbf5c6d055720ef2f95b0c93818133fd1ece`**,
  byte-identical to the combined dry run; record [`export_v55.json`](export_v55.json), receipt
  [`vigorous-whistle_v55.receipt.json`](vigorous-whistle_v55.receipt.json).
- Against the uploaded v54: 15,868 rows differ (LIRF 8,496 from the member step; LTFM 6,495 / EDDF 503 / EHAM 374 from this layer),
  S 1.14e7, RMS 5.8 s, no row beyond 600 s. Audit [`../export_audit/v55_audit.json`](../export_audit/v55_audit.json) clean: integrity
  checks pass, every flag (January airport-month means, daily outliers, disruption days) identical to v54's audit.
- Expected against v54 (221.8686): about −0.10 s if both parts transfer as their 2025 readings, about +0.10 s if both are noise.
- Upload only at the user's word:
  `.venv\Scripts\python.exe -X utf8 -B tools\prc_submit.py upload --file runs/submissions/vigorous-whistle_v55.parquet --expected-sha256 aa02ff597a619f223888370726cdcbf5c6d055720ef2f95b0c93818133fd1ece --calendar-day`

**Official result (2 October 2026).** Uploaded at the user's word ("upload it") at 05:09:10 UTC; processed 05:09:44 UTC:
**221.6941 on all 344,841 rows, −0.1745 s against v54 (221.8686): the new official best.** Record
[`official_result_v55.json`](official_result_v55.json). The squared-error change (−2.67e7 on a squared change of 1.14e7) corresponds to an
alignment of about 1.67 for the two parts together; that figure describes the outcome only and is not used to rescale anything. Never
re-upload v1–v55.

The first upload attempt stopped before anything was sent (the uploader reads `provenance.experiment_id`; the export tool had written
`experiment`). The tool was corrected, the never-uploaded local file and receipt were removed and rebuilt: the parquet is byte-identical
(SHA-256 `aa02ff59…`), only the receipt's provenance key changed.

## 8. Screens after v55 (2 October 2026) — closed, nothing declared

Scratch scripts in the session scratchpad (`rwy_ceiling.py`, `lfpg_screen.py`, `lfpg_exact.py`, `ltfm_screen.py`). No score is used.

- **Per-runway monthly taxi-out figures (AIU dashboard, never fetched).** Ceiling after v48's reconciliation on the development stack: if
  every (airport, runway, month) mean residual of labels in (0, 7,200] s were known exactly, the squared error would fall by 0.065% beyond
  the airport-month level (0.03–0.11% by month; RMSE about 0.03%); the biases involved are 12–25 s, the size of the published figure's own
  noise (`docs/clean_line_screens` section 4). Not fetched.
- **LFPG unmatched rows.** 76% of their development squared error sits in 15 rows without an NM record (labels up to 84,240 s). Elsewhere
  label minus NM proxy (AOBT_3 − block) is one smooth bell around 0 with an sd of about 4 minutes; the 21% within 60 s are its centre,
  not a separable population; no label-free field (operator, stand, runway, clock disagreements, market, hour, destination) carries a
  stable bias. LFPG block and takeoff times sit within ±6 s of a whole minute.
- **LTFM.** Residuals of the development stack carry no group bias (month × runway at most 33 s, July 2025) and no correlation with
  runway or airport queue counts from the known takeoff times (|corr| ≤ 0.011).

## 9. Declared round: count-day layer — ACCEPTED by its rule (2 October 2026)

**The finding.** EUROCONTROL's daily departure count `FLT_DEP_IFR_2` equals the number of in-figure departures whose recorded off-block
time, rounded to the minute, falls on that UTC day — on every non-month-end day of 2025 at LIRF, EGLL and LEBL, and all but 2 / 1 days at
LEMD / LTFM ([`count_exactness.json`](count_exactness.json)). The −1/+1 day pairs that looked like noise are departures blocked at
23:59:5x, which the figure counts on the next day. A prediction fixes a row's off-block day (takeoff − prediction); where the predicted
days disagree with the published counts, some row sits on the wrong side of a midnight. This is a hard constraint on each day, not a
classifier of label regimes.

**The procedure** (`src/prc2026_scout/count_days.py`, `tools/scout_count_days.py`, declaration
[`configs/scout_count_days_v1.json`](../../configs/scout_count_days_v1.json), 3744096, declared before the governing run): v48's own row
model gives each row's scenarios and their off-block days; rows with mass on two adjacent days are candidates; the posterior over the
month's midnights is exact (forward-backward), with the counts as near-exact constraints and the daily delay totals as v48's day-model
density for candidates whose two days differ by more than an hour of positive delay. Only midnights where a count disagrees move; a
candidate whose two days differ by more than an hour in mean label moves only when the posterior puts it across midnight; a moved row
takes the change of its posterior mean (never a hard assignment).

**Governing run** ([`count_days_results_v1.json`](count_days_results_v1.json); v48's declared run values as the base, cells from the other
development months):

| month | RMSE change | paired-day 95% interval | rows changed | squared error by airport |
|---|---:|---|---:|---|
| January 2025 | −0.0008% | [−0.0027, +0.0000] | 32 | LTFM −0.17e6 |
| July 2025 | −0.0057% | [−0.0181, +0.0015] | 99 | LTFM −1.48e6, LEMD +0.08e6, LEBL +0.02e6 |
| November 2025 | −0.0218% | [−0.0473, −0.0042] | 96 | LTFM −1.88e6, LEMD −0.43e6, LIRF −0.07e6 |
| December 2025 | −0.0449% | [−0.0968, −0.0091] | 66 | LIRF −3.45e6, LTFM −0.37e6, LEMD −0.28e6 |

Calibration over 315 flagged candidates (posterior's day against the recorded off-block day): confidence ≥ 0.99 — 236 candidates, 99.6%
right; 0.9–0.99 — 40, 95%; 0.7–0.9 — 20, 70%; 0.5–0.7 — 19, 84%; 2 candidates were on neither day. Crossings (the posterior puts a
row on the other side of midnight from its value): 46, of which 41 right. **All four rule checks pass: accepted.** 2025 holds no big
crossing (a row moved by hours across midnight): the type that carries the ranking stake is supported by the counts' exactness and the
calibration of confident posteriors, not by a 2025 case of its own.

**On v55 (label-free dry run, not built, not uploaded):** `runs/scout/count_days/dry_v55/vigorous-whistle_v56.parquet`, SHA-256
`7be7f3e13a7624bc701aacdf3e05748d1bbfb88b4e47ae63bcd6bf25dc64517b`, record [`count_days_dry_v55.json`](count_days_dry_v55.json):
173 rows differ from v55, sum of squared change 7.96e8 (about 4.7% of v55's squared error), max 24,148 s.

| airport | rows | squared change | what the counts say |
|---|---:|---:|---|
| LTFM | 124 | 8.0e6 | snow nights 13–19 January: fewer departures left before midnight than the values imply; near-midnight rows 100–680 s lower |
| LEBL / LEMD | 12 / 23 | 1.1e6 / 0.7e6 | near-midnight rows |

What is at stake: if every move is right, about −5 s on v55; if every move were wrong, about +5 s. The 5 July call rests on the counts being exact (as on every 2025 day at LIRF) and on the delay total favouring the
25,499 s row; if the other row were the one that left on the 5th, the layer would cost about 3.8 s and miss a larger gain.
Order: the layer runs last on v55's file; re-solving v48's day with the moved rows would shift the other rows of those days by a few
seconds (disclosed, not done). Build and upload only at the user's word (as the next unused version, after `tools/audit_export.py`).
