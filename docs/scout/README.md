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

### 9a. Amendment v2 — REJECTED by its rule (2 October 2026)

[`configs/scout_count_days_v2.json`](../../configs/scout_count_days_v2.json) (308110b, declared after v1's
governing run, ranking-side motivation only, rule unchanged) took the delay-total likelihood from each airport-month's own day residuals
(kernel density, the day left out) and moved big candidates whenever the counts raise the probability of the side opposite their value.

**Governing run** ([`count_days_results_v2.json`](count_days_results_v2.json)): January −0.0008%, July **+0.069%**, November −0.0218%,
December **+61.57%**; all four big crossings wrong; total squared error +7.44e9. **Rejected.** Cause: with the month's own wide residuals
the delay total no longer tells the hypotheses apart, and the counts were "explained" by moving two LIRF same-day fill rows of December 2025
(labels 14,471 and 39,120 s) to the row model's one-day outcomes (85,885 and 87,492 s) at posterior 0.98 and 1.0. v1's day-model
density kept them. The empirical density is not a safe replacement; v1 stays the accepted procedure.

What v1 means for 5 July on the ranking: v55 is wrong on one of the two rows if the counts are exact. Moving the 25,499 s row fully (v1)
has a lower expected squared error than v55 whenever the probability that it is the one that left on the 5th exceeds one half (expected
(1 − p) × 2.74e9 against v55's p × 5.8e8 + (1 − p) × 2.2e9). Every reading favours it: v48's row model about 3:1, the 5 July delay total
2:1 against the month's own residuals up to about 30:1 under the day model; the cost if wrong is about 3.8 s.

## 10. v56 built from v1 (user: "Build v56 from v1"), not uploaded

`tools/scout_count_days.py --base runs/submissions/vigorous-whistle_v55.parquet --version 56 --declaration configs/scout_count_days_v1.json`
(the accepted v1; v2 rejected).

- **`runs/submissions/vigorous-whistle_v56.parquet`, SHA-256 `7be7f3e13a7624bc701aacdf3e05748d1bbfb88b4e47ae63bcd6bf25dc64517b`**,
  byte-identical to the dry run; record [`export_v56.json`](export_v56.json), receipt [`vigorous-whistle_v56.receipt.json`](vigorous-whistle_v56.receipt.json).
- Audit [`../export_audit/v56_audit.json`](../export_audit/v56_audit.json)
  clean: integrity passes; airport-month, daily-outlier and disruption flags as in v55's audit (only the means of LIRF 5 January / 2 July
  and LTFM 12 / 18 / 19 January move by 1–27 s).
- Upload only at the user's word:
  `.venv\Scripts\python.exe -X utf8 -B tools\prc_submit.py upload --file runs/submissions/vigorous-whistle_v56.parquet --expected-sha256 7be7f3e13a7624bc701aacdf3e05748d1bbfb88b4e47ae63bcd6bf25dc64517b --calendar-day`

**Official result (2 October 2026).** Uploaded at the user's word ("upload it"); processed 06:41:43 UTC: **224.0957 on all 344,841 rows,
+2.4016 s against v55 (221.6941). v55 remains the official best.** Record [`official_result_v56.json`](official_result_v56.json),
submissionId `5d0fb9df-4ff7-4869-95b7-fac28b4f8b8e`. The squared-error change is +3.69e8 on a layer whose squared change is 7.96e8. No
row-level reading is drawn from the score: which moves were right is not inferred, nothing is rescaled or re-selected from it, and no
further version of this layer is built on it. The count-day layer is the fourth change touching LIRF population rows that read well on
2025 and lost on the scored months (v10, v43, v50). Never re-upload v1–v56.

## 11. After v56: screens that stay closed (2 October 2026)

Scratch scripts `day_ceiling.py`, `alloc_screen.py` in the session scratchpad; no official score is used.

- **No daily taxi-out figure exists.** EUROCONTROL's airport data page lists taxi-out (and taxi-in, ASMA) additional time monthly only;
  the daily series are traffic counts (now used by the count-day layer), arrival ATFM delay, ATC and all-causes pre-departure delay and
  slot adherence (all tried). After v48 the airport-day level of the development residuals is 0.10–0.59% of squared error, the ceiling a
  daily figure would have had.
- **EGLL's denser 2026 ADS-B traces are not a bias signal.** On gate-matched rows, prediction minus ADS-B taxi (takeoff − motion start)
  falls from a median 301–443 s in the development months to 206 s in July 2026 (47% gate rows); in every development month the
  prediction's offset tracks the label's (January 415 / 417, July 443 / 453, November 301 / 287 s) and the label's offset itself falls as
  coverage densifies (November, the densest 2025 month, 287 s). July 2026 continues that relation.
- **Allocating v48's day shift by row group** (leave-one-month-out slope of label − v48 on the implied day shift, half weight): predicted
  late vs on-time rows January −0.013%, July −0.045%, November −0.020%, December +0.004% (equal-footing mean −0.018%, under the 0.02% bar,
  and a variant of the general day-shift calibrations that failed in section 3); by hour bank, tracked / untracked, airport or airport ×
  hour bank worse. Closed.
- Guard for every later file: the two 4–5 July LIRF rows stay at their v55 values (25,499 and 47,705 s), since v56's score now bears on
  them; LIRF population rows are not touched by any scout layer.

## 12. Declared round: variance re-allocation of the day's discrepancy — ACCEPTED by its rule (2 October 2026)

Declaration
[`configs/scout_var_alloc_v1.json`](../../configs/scout_var_alloc_v1.json) (08:36 UTC, 028aeba); code `src/prc2026_scout/var_alloc.py`,
`tools/scout_var_alloc.py`, tests `tests/test_scout_var_alloc.py`; governing run [`var_alloc_results_v1.json`](var_alloc_results_v1.json).
No official score is used.

**What it does.** v48 moves the ordinary rows of an airport-day by one posterior day shift plus a tilt sized by the row's cell. This
layer, after the extreme-day layer, re-allocates the day's discrepancy I_d (the extreme-day layer's implied shift: in-figure rows, c0
offset) between rows by a learned, label-free weight w = exp(LightGBM of log r²) × P(late), each row moving by
clip(k · I_d · (w / day mean of w − 1), ±100 s); the changes of a day sum to zero before the cap, so the day's level stays the earlier
layers'. Heavy rows, rows of unread days, the LIRF population and the two 4–5 July guard rows never move. k (one slope) and both
models are fitted without the scored month.

**Disclosed before declaring:** the four development months' readings of this procedure (exploration in the session scratchpad: an
uncapped variance form, a sensitivity form, caps on the weight and on the change, heavy rows kept or not; the 100 s cap was chosen for
the least concentration), and a label-free ranking dry run. **Not seen before declaring:** any twelve-month reading (the exploration run
was stopped on its third fold; the governing run is the first).

| Development stack (control: extreme-day layer on v48) | change | paired-day 95% interval (s) | k | best ten rows' share of gain |
|---|---|---|---|---|
| January 2025 | −0.0440% | [−0.300, −0.067] | 0.238 | 33% |
| July 2025 | −0.1377% | [−0.563, −0.163] | 0.223 | 18% |
| November 2025 | −0.0613% | [−0.249, +0.009] | 0.260 | 59% |
| December 2025 | −0.1114% | [−0.301, −0.074] | 0.242 | 38% |

Twelve-month check (v23-recipe predictor, eleven-month fitting side): **all twelve months better**, mean −0.134%: January −0.042,
February −0.499, March −0.225, April −0.078, May −0.111, June −0.110, July −0.140, August −0.046, September −0.026, October −0.155,
November −0.083, December −0.097%; ten of twelve intervals below zero (April and September touch zero). Rule: no development month
worse (worst −0.044%), equal-footing mean −0.098% (bar −0.02%), every equal-footing interval below zero, twelve-month no month worse and
at least two thirds better (12 of 12): **accepted**. On the development stack 30 of 40 airport-months are better, LIRF in all four.

**Ranking dry run** (`runs/scout/var_alloc/dry_v57/`, [`var_alloc_dry_v57.json`](var_alloc_dry_v57.json); v55's chain reproduced byte for
byte first): 256,015 rows change, sum of squared change 3.03e7 (0.18% of v55's squared error), largest change 100 s, root-mean-square
9.4 s; largest cells LTFM January 2026 (7.4e6), LIRF July 2026 (3.9e6), LTFM July 2026 (3.6e6); guard rows and LIRF population unchanged.
Dry-run file SHA-256 `31d909ff2e5d1937e65404c396893b9e5524d082c07422be9390c75aad4daed8` (not under runs/submissions). Expected size:
across the sixteen 2025 readings the squared-error gain is 0.7–5.1 times the squared change (median about 2), so about −0.4 s if the
scored months behave like 2025 (−0.13 s at the weakest month's ratio), about +0.2 s if the change carries no information.

Build and upload only on the user's word, as the next unused version (v57):

```
.venv\Scripts\python.exe -X utf8 -B tools\scout_var_alloc.py --base runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet --recon runs/scout/v55_build/recon --version 57 --declaration configs/scout_var_alloc_v1.json
```

Final-file chain if shipped: runner `--layers v55m` → `tools/export_daily_totals.py --base` → `tools/scout_extreme_shift.py` →
`tools/scout_var_alloc.py`.

## 13. v57 built (user: "build v57"), not uploaded

`runs/submissions/vigorous-whistle_v57.parquet`, SHA-256 `31d909ff2e5d1937e65404c396893b9e5524d082c07422be9390c75aad4daed8` = v55's chain
→ the variance re-allocation layer of section 12, built by the command there from the accepted declaration; byte-identical to the dry run.
Against v55: 256,015 rows, sum of squared change 3.03e7, RMS 9.37 s, largest change 100 s; the two 4–5 July guard rows keep 25,499 and
47,705 s; no LIRF population row changes. Receipt [`vigorous-whistle_v57.receipt.json`](vigorous-whistle_v57.receipt.json), report
[`export_v57.json`](export_v57.json), audit [`../export_audit/v57_audit.json`](../export_audit/v57_audit.json): clean, the same flags as
v55 (January 2026 airport-month levels, the EHAM 2–7 January and LFPG 7 January disruption days). Upload only on the user's word:

```
.venv\Scripts\python.exe -X utf8 -B tools\prc_submit.py upload --file runs/submissions/vigorous-whistle_v57.parquet --expected-sha256 31d909ff2e5d1937e65404c396893b9e5524d082c07422be9390c75aad4daed8 --calendar-day
```

**v57 = 221.3289 OFFICIAL, the official best** (uploaded at the user's word, "upload it"; processed 08:53:14 UTC 2 October, submissionId
`9dc2c761-cbaa-4839-8ac3-1d521030e67e`; record [`official_result_v57.json`](official_result_v57.json)): **−0.3652 s against v55 (221.6941)**.
The squared-error gain on the scored months is 1.84 times the layer's squared change (5.58e7 against 3.03e7), inside the 2025 range
(0.7–5.1, median about 2): the layer transferred as its development readings said. Nothing is re-selected or re-tuned from the score. Never re-upload v1–v57. Final-file chain for this file: runner `--layers v55m` →
`tools/export_daily_totals.py --base` → `tools/scout_extreme_shift.py` → `tools/scout_var_alloc.py`.

## 14. After v57: refinements of the same mechanism — closed (2 October 2026, 2025 only, nothing declared)

Scratch `va_next.py`, `va_feat.py` in the session scratchpad; development stack, control v57's layer, month-blocked as its governing
run. No official score is used, nothing is re-read from v57's score.

- **Splitting v48's own day shift by the variance model** is v57 itself: its I_d is the whole airport-day
  discrepancy on every day with a published total, not the extreme-day residual.
- **One k per airport**: equal-footing mean −0.082% against the global k's −0.098% (in-sample k from 0.05 at EHAM to 0.41 at LIRF; the
  spread is noise at this sample size). Closed.
- **k by the size of I_d**: 0.54 / 0.28 / 0.26 / 0.19 for |I| in [0, 10) / [10, 30) / [30, 80) / [80, 600] s; the small-|I| rows move
  by a few seconds, so a size-dependent k has nothing to gain. A second pass on the same basis after the layer leaves a slope of 0.07.
- **Richer variance inputs** (runway, stand / aircraft-type / operator frequency, |I_d|): −0.0999% / −0.0998% against −0.0977%; the
  variance model is saturated on label-free inputs. Closed.
- **Day signal**: allocating v48's posterior day shift instead of I_d reads −0.067%, the shift in effect after the extreme-day layer
  −0.077%, against −0.098% for I_d (scratch `va_signal.py`). Closed.

**Chain step.** `tools/scout_var_alloc.py` now checks its rebuilt extreme-day step against the chain's previous file
(`--previous FILE --previous-sha256 SHA`, the output of `tools/scout_extreme_shift.py` on the same base and reconciliation; default the
uploaded v55), so the final-file chain can run it on a fresh runner output. With v55 given explicitly it rebuilds v57 byte for byte
(SHA-256 `31d909ff...`, `runs/scout/var_alloc/dry_v57_chaincheck/`).
- **A uniform extra day shift after v57** (beta2 x I_d on eligible rows, month-blocked): all days +0.0013 / −0.016 / +0.0005 /
  +0.032% (equal-footing mean +0.006%), |I| < 80 s only +0.007%: the general day-level calibrations of section 3 still fail with the
  allocation fixed (scratch `va_level.py`). Closed.
- **Allocating along member disagreement** (I_d x the day-centred committee − networks, v9 − mix or stack − v9 gap; development rows
  of `runs/lomo_dev/rows.parquet`): the slope changes sign between folds (committee − networks +0.16 / −0.35 / +0.06 / +0.05) and every
  variant is worse (equal-footing means +0.007 to +0.012%; scratch `va_member.py`). Closed.
- **Month-end days** are reconciled in the ranking (published counts equal our rows) except LTFM on 31 January 2026 (713 published,
  639 rows): one airport-day, worth about 0.003 s. Not pursued.

## 15. More cells after v57 — closed (2 October 2026, 2025 only, nothing declared)

No official score is used.

- **Fill blocks on LIRF rows with an NM clock (outside the population).** LIRF matched rows: RMSE 442 s against 75–183 s elsewhere; the
  recorded block sits a median 593 s before the first ADS-B motion; the 337 rows whose block is more than an hour before first motion hold
  30% of the cell's squared error; 20% of the rows have block = schedule (36.7% of the squared error). Delayed rows (takeoff − schedule
  > 1,800 s) hold 89% of the cell's squared error, with a fill share of 7.8% (stable by month); the operator shifts it from 0.5% to 19%.
  A month-blocked fill classifier reaches AUC 0.866 (timing + operator / stand / runway / type / destination; the stack's own hedge
  control / a alone 0.819). But the explicit mixture p · a + (1 − p) · t loses badly (July +0.59% at half strength, +2.32% at full,
  intervals above zero; the real-taxi model is worse than the stack on non-fill rows), and a residual correction with p as input is
  inconsistent (January +0.005% at half strength, the LIRF-dark month that resembles July 2026). Closed.
- **The extreme-day layer at the uncalibrated airports.** On the scored months only LFPG (7 January, I −91 s), LSZH (10 January) and
  LIRF (seven July days, where the 2025 analogue put the excess in the population rows) have |I| ≥ 80 s; EGLL none. At most about 0.01 s.
  Closed.
- **De-icing / cold mornings.** January and November 2025 residuals after v57's layer by temperature band × local hour stay within about
  ±10 s at EDDF, EDDM, LSZH, EHAM and EGLL (no cold-morning under-prediction); the weather inputs and the daily totals already carry it.
  Closed. (Already closed elsewhere and not repeated: the aircraft-rotation link, `fresh-lens` 28 September; test-time adaptation from
  matched rows, `docs/tta_screen/`.)

## 16. Declared round: refit the published-data steps on the shipped model part — REJECTED by its rule (2 October 2026)

Declaration [`configs/scout_refit_layers_v1.json`](../../configs/scout_refit_layers_v1.json) (10:06 UTC, 6365780), declared before any
layer ran on the table; result [`refit_layers_results_v1.json`](refit_layers_results_v1.json); code `tools/scout_refit_prep.py`,
`tools/scout_refit_run.py`. No official score is used.

Control = the shipped parameters (v48's cells and scales, the extreme-day betas and c0, v57's models and k, all fitted on the v46-era
months) applied to v55m's scored month; candidate = the same three steps refitted on v55m's own months. Month-blocked in both.

| Month | v55m model part | shipped chain on v55m (control) | refitted chain (candidate) | change | interval (s) |
|---|---|---|---|---|---|
| January 2025 | 322.5451 | 322.0588 | 322.0697 | +0.0034% | [−0.002, +0.034] |
| July 2025 | 257.4099 | 255.9946 | 256.0015 | +0.0027% | [−0.014, +0.029] |
| November 2025 | 184.6512 | 184.0052 | 183.9712 | −0.0184% | [−0.085, +0.007] |
| December 2025 | 167.0696 | 166.2473 | 166.2517 | +0.0026% | [−0.022, +0.031] |

Equal-footing mean +0.003% (bar −0.02%), no interval below zero: **rejected**. The refit moves v57's k by less than 0.01. The shipped
parameters are already right for the shipped model part, and the chain keeps its full gain on it (−0.15 / −0.55 / −0.35 / −0.49%
against v55m alone). Selection inheritance (the members were chosen on these months) and the missing twelve-month check are stated in
the declaration; neither matters for a null.

**Census of the shipped chain on the shipped model part** (development, `runs/scout/refit/ctl_v55m.parquet`, pooled RMSE 239.33 s;
scratch `census57.py`): the 78 labels above 3 h hold 35.3% of all squared error (three LFPG rows 26.6%, among them the two unidentifiable
January recording errors; 70 LIRF population rows 7.9%). Ordinary labels: LTFM unmatched 15.5%, LFPG unmatched 14.5%, LIRF matched 13.9%,
LIRF population 13.4% (492 rows, RMSE 2,605 s), EGLL unmatched / matched 7.1% / 5.9%, LIRF unmatched 6.1%, LEBL matched 4.8%; rows
without an NM record 1.2% of rows and 20.2% of ordinary squared error.

**v57's layer with v48's heavy rows (outside the LIRF population) made eligible**, same 100 s cap, on the v55m chain: −0.0002 / +0.0054 /
−0.0008 / +0.0032% (scratch `heavy_in.py`). Closed.

## 17. Systematic small-cell scans on the shipped chain's development proxy — closed (2 October 2026, 2025 only)

Control: the shipped chain on the shipped model part (`runs/scout/refit/ctl_v55m.parquet`). Scratch `bias_scan.py`, `low_clamp.py`.

- **Month-blocked group-bias corrections** (shrunk means of the other three months, k = 200 rows; LIRF population never changed) over
  18 groupings: airport, × ADS-B match, × take-off-minus-schedule band, × prediction band, × hour band, × weekday, × NM status, × wake
  category, × market segment, × runway, × stand area, × flight type, and prediction / delay bands alone. **No grouping improves all four
  months.** Best: airport × prediction band −0.056% equal-footing mean but January +0.035%; airport alone −0.025% with November +0.014%.
  The chain's residuals carry no simple group bias. Closed.
- **Prediction clamps** at the airport × match label quantiles of the fitting months: low side 0.1% / 0.5% / 1% / 2% neutral to worse
  (up to +0.085%); high side 99.5% / 99.9% badly worse (+0.3 to +5.1%). Closed.
- **Group-specific day slopes after v57** (residual on I_d by airport, airport × match, airport × prediction band, match × prediction band;
  month-blocked, shrunk; scratch `group_slope.py`): every variant worse or within noise (equal-footing means −0.004 to +0.051%). Closed.
- **ADS-B matching gaps in 2026** (label-free; scratch `match_rate.py`, `match_rate2.py`): Turkish Airlines' new alphanumeric callsigns
  did not cost matches outside LTFM (THY match rates at the other airports equal 2025's or better). EDDF's lower 2026 match rate
  (0.92 → 0.76–0.80) is uniform across its large carriers (DLH −18, CFG −19, OCN −18 points), i.e. receiver coverage, not matching; the
  cargo prefixes fall a little more (ACA, GEC, CCA −22..−26 points beyond the airport). Nothing to recover. Closed.
