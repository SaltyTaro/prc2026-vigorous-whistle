# Population signals round (28 September 2026, after v40; slots for the day used)

The user asked for a deep search for further RMSE reductions towards first place (v40 = 224.4588 official; first place 222.4726 at the
03:46 UTC snapshot; the gap is about 3.1e8 SSE, 1.8% of ours). Everything below is label-free on the ranking; labels are read only on
the twelve 2025 months. Scratch diagnostics ran in the session scratchpad; the declared round's code is in the repository.

## Where the squared error sits (v33 = v40 without the layers; LOMO/fair months `runs/lomo_dev/rows.parquet`)

| cell | Jan | Jul | Nov | Dec |
|---|---:|---:|---:|---:|
| LIRF population (v9 mixture rows; 60 / 337 / 52 / 88 rows, RMSE 2,400–5,100 s) | 6.1% | 27.8% | 23.9% | 10.6% |
| LIRF non-population: schedule fills' share of LIRF's own SSE | 39% | 34–50% | 42–43% | 48–51% |
| LTFM clock rows (no ADS-B) | 5.0% | 11.0% | 13.7% | 16.6% |
| LFPG unmatched | 6.4% | 8.0% | 10.7% | 14.4% |
| EGLL | 4–7% | 8–14% | 9–12% | 13% |

The ranking has 383 population rows (107 January, 276 July), 372 with a schedule interval at or below 43,200 s (the v9 mixture scope;
the 12 above it are the fill-pin rows, decided by v39–v41). Population rows sit **exactly at v9** in the final stack (v33 = v9 on all
537 development population rows; the layers v31–v40 never touch them), and v9 = 0.25 × v7 + 0.75 × mixture on the in-scope rows
(reproduced from `runs/lirf_mixture/export_v9/population_predictions.parquet`).

## Closed this session (scratch, nothing declared)

- **LIRF clock-present exact fills** (block = schedule ± 5 s): 16.3% of clock rows overall, but 4–9% where they cost (AOBT more than
  10 minutes after schedule; 34% when AOBT is within ± 2 minutes of schedule, where fill and proxy nearly coincide). Fills are 41% of
  July's LIRF clock-row SSE. A leave-one-month-out fill classifier on the costly rows has AUC 0.885; the new inputs below move it to
  0.894, and an explicit mixture is worse than the trees' implicit hedge at every blend weight (25% blend: Jan +1.2%, Jul +7.4%,
  Nov −1.8%, Dec −0.7% of the clock-row SSE). The v10 finding again, now with the new inputs.
- **Hedge bias by AOBT–schedule band at LIRF**: mean residual 0–30 s in every band and month; a per-band constant shift gains ~0.
- **Non-LIRF no-NM rows with long schedule intervals**: labels are ordinary (median 1,020–1,085 s in every band above 1 h; 0.8–2.6% of
  rows have a label above half the interval); the stack's band residuals are +30…+100 s and a band shift gains ~0. The large errors
  in that cell are selection on the label (disruption days: EGLL 16 March, 6 and 30–31 July 2025). LTFM's no-NM rows have a heavier
  tail (23–27% above half the interval beyond 1 h, 14% near-fills in the 1–2 h band) but only 125 such ranking rows; no separator.
- **Arrival taxi-in context** (mean taxi-in anomaly of arrivals completed 10–40 / 10–70 minutes before takeoff): correlation with the
  stack residual 0.004; ≤ 0.2% of any airport's SSE. The timeline bundle carries arrival counts and schedule delays but no taxi-in;
  it would not help.
- **LTFM block time versus the NM clocks**: block − AOBT_3 has median +296 s and MAD 189 s (AOBT_3 leads the block, an estimate);
  the stack's LTFM RMSE (187–239 s) is at that clock floor.
- **Block-source census** (which supplied clock the departure block equals within 5 s, all airports): only LIRF has schedule fills of any
  weight (15.5% of rows; 2.2 / 13.1 / 7.7 / 13.0% of monthly SSE); EOBT_1 / IOBT / LOBT fills are 0.3–4% of rows and ≤ 0.5% of SSE
  everywhere. No fourth label mechanism.
- **One-day rows below 43,200 s**: 1 of ~1,460 population rows in 2025; no hedge is warranted there.
- **LIRF unmatched blend in the dark regime** (January 2025, no ADS-B at LIRF before May 2025): v33 273.1 s against a least-squares
  optimum of 271.9 s; the current weights hold in the July-2026-like regime.

## The lever: new label-free inputs for the population classifier

v9's classifier reads the row's own clocks and the *timing* of the previous arrival in-block at the stand. Two things it does not read:

1. **What that arrival is.** When the previous arrival at the stand has a different operator prefix, the departure is a schedule fill
   78.5% of the time (177 rows) against 44.5% when it is the same operator (1,302 rows). Also its planned turnaround (departure
   schedule minus arrival schedule), whether its own in-block equals its schedule, and whether it has an NM record.
2. **The callsign's own fill history** (leave-one-month-out share among population rows; smoothed). Top quintile 94% fills, bottom 8%.
   Caveat: 279 of the 383 ranking population callsigns never appear in the 2025 population (35% of training rows are in that state),
   so this input is mostly the prior on the ranking.

The ranking population's feature distributions (interval, planned turnaround, hour, previous-arrival prefix share 0.875 vs 0.88) match 2025.

Declaration `configs/pop_signals_v1.json` (10:46 UTC): control = v9's seventeen features with the same protocol bagged over eight
seeds on twelve months; candidates `pa`, `pa2`, `cs`, `cs_pa`; leave-one-month-out over the twelve 2025 months; rule: at least 3%
lower twelve-month in-scope mixture SSE, better in at least 8 of 12 months, paired-month bootstrap 95% interval below zero.
Exploratory scratch reads on the same months are disclosed in the declaration. Runner `tools/run_pop_signals.py` →
`results_v1.json`; export `tools/export_pop_signals.py` (v42 = v40 + 0.75 × (mixture_new − mixture_v9) on the 372 rows, floor −12 s).
Module `src/prc2026_final/pop_signals.py`; tests `tests/test_pop_signals.py`.

**Transfer risk, stated up front.** The change is concentrated on 372 rows whose 2026 labels no development month can score. v10's
day-density classifier for the same rows passed the folds (−0.38% / −0.58%) and lost 0.52 s officially. The previous-arrival
identity signal is structural (a property of the stand's traffic, not of a month's density) and its ranking distribution matches 2025,
which is the argument for it; the callsign share is mostly the prior on the ranking, which is the argument against `cs`.

## Results (`results_v1.json`, 10:49–11:05 UTC; rows `runs/pop_signals/lomo_rows_v1.parquet`): nothing selected

Twelve-month leave-one-month-out mixture SSE on the 1,458 in-scope population rows (control 19.054e9; a perfect fill/normal
separation would be about 10.2e9):

| candidate | SSE (e9) | vs control | months better | paired-month bootstrap 95% | AUC (fill) |
|---|---:|---:|---:|---|---:|
| `pa` (previous-arrival prefix, planned turnaround, its fill flag, its NM record) | 18.285 | **−4.04%** | 7 / 12 | [−5.2%, +0.1%] | 0.918 |
| `pa2` (prefix + planned turnaround only) | 18.372 | −3.58% | 7 / 12 | [−5.1%, −0.3%] | 0.919 |
| `cs` (callsign fill share) | 18.627 | −2.24% | 7 / 12 | [−7.3%, +4.9%] | 0.926 |
| `cs_pa` | 17.967 | −5.71% | 7 / 12 | [−10.8%, +4.1%] | 0.928 |

No candidate reaches 8 of 12 months; `pa`'s interval touches zero (+0.1%). `selected_candidate` is null. Two readings behind the table:

- Excluding it, `pa` is −2.68%, `pa2` −2.77%, `cs_pa` −4.72% (row-level bootstrap for `pa` [−5.4%, +0.1%]).
  The whole gain is on normal-mode rows (8.17e9 → 7.85e9): the new inputs make the classifier less willing to call a normal
  departure a fill; fill rows are unchanged (3.73e9). About 250 of the 1,458 rows move by more than 300 s (mean |Δ| 180 s).
- **The gain lives in the heavy months.** `pa` by month: Jan +4.5%, Feb −1.2%, Mar +12.9%, Apr −6.2%, May −3.0%, Jun −5.3%,
  Jul −6.2%, Aug −6.5%, Sep −1.7%, Oct +8.1%, Nov +4.0%, Dec +0.9%. The months it loses carry 0.08–0.66e9 each; April–September
  carry 0.6–8.6e9 each. Weighted by SSE it wins; counted by month it is 7 / 12. The ranking's population is 72% July 2026
  (276 rows) and 28% January 2026 (107 rows).
- The callsign share (`cs`) raises the AUC most (0.926) but not the SSE: its swings (March +40%, September +16%) come from a few
  callsigns seen in one other month, and 279 of the 383 ranking callsigns are unseen anyway. Not a ranking lever.
- Scratch variants (not declared, disclosed): a small regression normal component instead of band medians is −2.3% alone and −6.6% with
  `pa`, but it loses January (+17% / +23%) and November (+14–17%), the light months and one of the ranking months; a stake-weighted
  classifier was a scratch bug (AUC 0.5) and is discarded.

**Dry-run export of `pa` on the ranking** (`tools/export_pop_signals.py --candidate pa --dry-run`, scratchpad only; SHA-256 of the
would-be v42 a8729c26…): 370 of the 372 in-scope rows change, 183 up / 187 down, mean |Δ| 204 s, largest +4,600 s (RYRR388G,
26 July, p_fill 0.55 → 0.75) and −2,189 s; January rows move −43 s on average, July +1 s; the mean
p_fill is unchanged (0.438 → 0.443). Sum of squared change 7.5e7, of which the twelve-month eight-seed refit of v9's own features
accounts for 2.3e7 and the new features for 2.9e7 (cross terms make up the rest). Scenario bounds on the changed rows: 223.98 if every
changed label equals the new value, 224.94 if it equals the old one; the leave-one-month-out reading implies about −0.5 s if the
population behaves as in 2025 (−2.7…−4% of a population SSE of roughly 2.9e9). `pa2` is the same picture (SHA-256 2ef11b10…).

## v42 built (11:25 UTC, user: "build v42 and keep searching"; NOT uploaded)

`runs/submissions/vigorous-whistle_v42.parquet`, SHA-256 `a8729c26565eae4ddc62cc8adc3f791f25efcadd8d5f08e29f61081863c55c41`
(byte-identical to the dry run): v40 + 0.75 × (mixture[`pa`, twelve months, eight seeds] − mixture[v9 shipped]) on the 372 in-scope
population rows, floor −12 s. Report `export_v42.json`; classifier bag and detail table `runs/pop_signals/export_v42/`; audit
`docs/export_audit/v42_audit.json` clean (370 rows changed, all LIRF population; every other row byte-identical to v40; the
airport-month flags are January 2026's known disruption/snow levels, unchanged from v40). Independent check: 370 changed rows, all in
the v9 scope. The report's `lomo_implied_rmse_change_s` (−1.27 s) uses the control's per-row SSE *including* the June one-day row and
the −4.04% total; the honest expectation excludes that row: about **−0.5 s** if the 2026 population behaves as 2025's, with a
comparable loss if the signal does not transfer (v10's precedent). The user uploads at the 00:00 UTC reset if they choose; a v42 runner
layer set is added only after an upload.

### Normal-component follow-up (scratch, on the declared LOMO rows with `pa`'s p_fill; not declared)

Replacing the band medians: band means −0.7% (7/12 months), a small LightGBM regression on interval/hour/weekday/month + the four
categoricals −3.4% (11/12), the same plus the turnaround features −4.6% (11/12), a 50/50 blend with the medians −2.3% (11/12); an L1
regression −1.2%. **Every regression variant loses January** (+3.5% … +14.7%): the "normal" class carries a heavy tail (p95 5,075 s,
p99 8,967 s) that a mean regression follows, and January's 34 normal-mode rows are short (median 844 s). Weighted by the ranking's
row mix (January rows carry a higher per-row SSE than July's), a candidate that loses January by 12% and wins July by 1% is a net
loss, so this is parked: no declaration, no export.

## Round 2: the month's arrival fill share (`configs/pop_signals_v2.json`, declared 14:58 UTC)

A label-free measure of the airport's record-capture state exists in every file: the share of LIRF *arrivals* whose in-block equals
their scheduled arrival (120 s). Across the twelve 2025 months it tracks the population's departure fill share with **r = 0.81**
(July 2025 is the lowest on both: 0.135 / 0.317; February–March and December the highest: 0.17–0.19 / 0.75–0.77). The ranking's
shares are 0.146 for January 2026 (January 2025: 0.148) and **0.156 for July 2026**, above July 2025's 0.135 and in the range of
August–December 2025, whose population fill shares were 0.43–0.77. The v9 classifier reads the *calendar month*, so on the ranking it
applies July 2025's low propensity to July 2026's 276 rows; a linear read of the arrival share implies about 0.52 instead of 0.32.
Day-level shares add little within a month (fill 0.44–0.53 across quintiles). Candidates: `pa` + share (month kept / dropped), the
share alone with the month dropped, and the day share; same rule as round 1.

**Results (`results_v2.json`, 15:00–15:50 UTC): nothing selected.** Against the same control (19.054e9): `pa_arrm` 18.228e9 (−4.34%,
8/12 months, bootstrap [−6.4%, +0.9%]); `pa_arrm_nomonth` 18.384e9 (−3.52%, 7/12, [−6.0%, +4.1%]); `arrm_nomonth` 19.100e9
(+0.24%); `pa_arrd` 18.380e9 (−3.54%, 5/12). The scratch month breakdown shows every arrival-share variant losing January by
+90…+180% (a few large-interval rows whose p_fill falls) and November by +10…+30%. Dropping the calendar month without a replacement
(`pa` minus month, scratch) costs January +57% and November +37%: the month feature carries the winter fill propensity.

**Why the r = 0.81 does not turn into a lever (calibration check on the leave-one-month-out rows).** The classifier's monthly mean
p_fill already tracks each month's actual population fill share with r = 0.91, through the rows' own features (interval, hour,
turnaround, operator). The *residual* miscalibration (actual − mean p) still correlates with the arrival share (r = 0.75), but its
size is small: July 2025 −0.07, January −0.13, March +0.15, December +0.08. On the ranking the v42 classifier's mean p_fill is
0.659 for January 2026 and 0.359 for July 2026; read through the residual relation, July 2026's arrival share (0.156) implies an
actual share only about +0.01…+0.03 above the classifier's mean, not the 0.52 of the naive month-level fit (which ignores that the
July 2026 rows are already scored by their own features). No calibration shift is warranted; `pa_arrm` gets the calibration right
(miscalibration correlation 0.49, July −0.04) but pays for it on the few January rows that dominate that month's SSE. Closed.

**Reading.** The previous-arrival identity is a real, structural separator (a −2.7% to −4% population-SSE effect, concentrated in the
summer months that resemble July 2026), but the declared rule was set for month-level consistency and it does not clear it. Shipping
it is a user decision with the odds above; the stake on the ranking is of the order of ±0.5 s (population SSE ≈ 16–26% of the total,
times 2.7–4%), on 372 rows.

## Closed after v42 (28 September, 17:55–18:15 UTC; scratch only, nothing declared)

- **UTC-day truncation of ADS-B tracks.** `tools/build_adsb_features.py` matches each departure against its own UTC day's surface
  file only, so a track that crosses midnight is cut at 00:00 (the rematch tool already loads the previous day). On the ranking this
  touches **55 matched rows** whose segment starts within 60 s of midnight (takeoffs 51 s to 3.5 h after midnight; the previous
  day's file holds 24–303 ground points of the same aircraft for the first twelve checked) and a handful of 23:45–24:00 departures
  whose airborne points fall in the next file. Not a lever at this size (well under 0.02 s); recorded for a future rebuild.
- **No-NM departures outside EDDF/EHAM.** Of the 5,290 no-NM ranking departures, 383 are the LIRF population and the IATA-format
  flight keys (1,276 unmatched rows) are all at EDDF/EHAM, which v35's callsign fix already covers. Every other airport's no-NM
  rows carry ICAO callsigns and match at the airport's normal rate (EDDM 94–97%, EGLL 76–99%, LEBL 98%, LEMD 85–91%, LFPG 19–25%,
  LSZH 26–28%, LTFM 0%). Nothing to rematch.
- **ADS-B observation as the population's normal component.** 714 in-scope 2025 population rows have an ADS-B match (May–December;
  none before May), 56 of the ranking's 107 January 2026 rows and 2 of July's. On normal-mode matched rows the observed taxi
  (takeoff − motion start) has a tight core (label − observed median 481 s gate / 628 s dark, MAD 187–359 s) but the label's heavy
  tail is not in the ADS-B: RMSE 2,874 / 3,108 s against the band median's 1,450 / 1,656 s. Replacing the band median by the
  observed taxi plus a LOMO-fitted offset worsens the `pa` mixture in every month (12-month SSE +13% at full weight, +3.5% at half).
  The population's "normal" labels are mostly block-time artefacts, not taxi observations. Closed.

## Round 3: operator descriptors and stake-weighted fits (`configs/pop_signals_v3.json`, declared 18:43 UTC)

Scratch reads after v42 (eight seeds, twelve-month leave-one-month-out, the scratch harness of rounds 1–2, control 19.007e9):

| variant (on top of `pa`) | SSE e9 | vs control | months | bootstrap 95% | Jan | Jul |
|---|---|---|---|---|---|---|
| `pa` (v42) | 18.210 | −4.19% | 9/12 | [−5.4, −0.6] | +4.4% | −7.0% |
| stake-weighted fit | 18.142 | −4.55% | 9/12 | [−8.0, −2.8] | −21.4% | −7.4% |
| isotonic calibration (inner OOF) | 18.155 | −4.48% | 10/12 | [−6.2, −1.1] | +11.5% | −7.9% |
| next-arrival fill / same prefix | 18.246 | −4.00% | 8/12 | [−5.2, +0.2] | +2.9% | −6.3% |
| stand / apron / 2-hour arrival shares | 18.591 | −2.19% | 5/12 | [−4.1, +2.2] | +3.6% | −6.6% |
| operator arrival fill share (smoothed) + count | 17.753 | −6.60% | 8/12 | [−13.0, −2.0] | −31.6% | −15.2% |
| + prefix no-NM / no-type / count | 17.900 | −5.82% | 8/12 | [−12.2, −0.4] | −30.0% | −16.1% |
| operator share + count, stake-weighted | 17.575 | −7.53% | 9/12 | — | −38.7% | −18.0% |

The operator descriptors are label-free per (month, prefix) over every LIRF movement of the file: they tell the classifier what kind of
operator a row belongs to (NM-filed and typed airline, or mostly unfiled operator), which the prefix categorical cannot for rare or
unseen prefixes. They help small and unfiled operators and the normal-mode rows of large carriers, and hurt on some large-carrier
fills (RAM, VLG, DLH in March/December: +19…+25% in those months). The stake weights make the fit spend its capacity where the
fill/normal decision moves the squared error (|interval − band normal| clipped to [600, 20,000] s), which is what the mixture is scored
on; they lower the AUC and raise the SSE reading. Isotonic calibration is not declared (loses January, the heavier ranking month);
next-arrival and stand-level capture-state features gave nothing.

Declared candidates: `pa_stake`, `pa_op`, `pa_op_stake`, `pa_opall`, `pa_opall_stake`; the v1 rule, unchanged; the report adds each
candidate's January/July readings and a ranking-composition read (January and July 2025 per-row SSE scaled to the ranking's 104 / 268
in-scope rows) for the user's decision. A selected candidate exports as **v43 = v40 + 0.75 × (mixture − v9 shipped)**, replacing v42.

**Round-3 results (`results_v3.json`, 18:50–19:35 UTC; the first run crashed after the control on a tooling bug in the informational
ranking-read block, fixed in commit 67a9cfd and rerun from scratch; rows `runs/pop_signals/lomo_rows_v3.parquet`). The rule selects
`pa_op_stake`.** Against the control (19.054e9, reproduced exactly from rounds 1–2):

| candidate | SSE e9 | change | months | bootstrap 95% | Jan | Jul | ranking read |
|---|---|---|---|---|---|---|---|
| `pa_stake` | 18.224 | −4.36% | 9/12 | [−7.6, −2.6] | −21% | −7% | −9.3% |
| `pa_op` | 17.826 | −6.45% | 8/12 | [−13.1, −2.0] | −32% | −15% | −17.8% |
| **`pa_op_stake`** | **17.646** | **−7.39%** | **8/12** | **[−16.1, −2.3]** | **−38%** | **−18%** | **−21.1%** |
| `pa_opall` | 17.958 | −5.76% | 6/12 | [−12.3, +0.0] | −31% | −16% | −18.1% |
| `pa_opall_stake` | 17.825 | −6.45% | 7/12 | [−14.8, −0.6] | −36% | −19% | −21.5% |

Three candidates are eligible; `pa_op_stake` has the lowest SSE. Its losing months are March (+29%), October (+5%) and December
(+13%), the large-carrier fills noted above; excluding the June one-day row its twelve-month change is −10.9%. The prefix departure
descriptors (`pa_opall*`) add nothing over the arrival share and fail the month count.

## v43 built (19:41 UTC; the rule's selection, NOT uploaded)

`runs/submissions/vigorous-whistle_v43.parquet`, SHA-256 `8c4a031b1319a0d5a78613a1ee63851ac0c9461b065abff0f065ce0439a73ff7`
(byte-identical to the dry run): v40 + 0.75 × (mixture[`pa_op_stake`, twelve months, eight seeds, stake-weighted] − mixture[v9
shipped]) on the 372 in-scope population rows, floor −12 s; it **replaces v42** (the `pa` candidate) rather than stacking on it.
Report `export_v43.json`; bag and detail `runs/pop_signals/export_v43/`; audit `docs/export_audit/v43_audit.json` clean (372 rows
changed, all LIRF population, every other row byte-identical to v40; the flagged airport-months are January 2026's known regimes,
unchanged from v40). Moves: January 104 rows, mean −129 s; July 268 rows, mean +93 s; mean |Δ| 440 s, largest 8,622 s; sum of
squared changes 2.6e8 (v42: 0.75e8; the two deltas correlate 0.57). Mean p_fill 0.438 (v9) → 0.445.

Expectation, stated with its assumptions: the twelve-month reading (−7.4% of population SSE, −10.9% without the one-day row) is
worth **−1.3 to −3.2 s** for a ranking population share of 16–26% of the total squared error; the January/July analog read scaled to
the ranking's rows says −3.8 s; if the moves do not transfer at all they cost about +1.7 s (v10's precedent: a population change that
passed its folds lost 0.52 s). The bounds in the report (222.76 / 226.14) are the extremes where every changed label sits at the new
or the old value. The user decides the upload at the 00:00 UTC reset; v42 stays built but superseded.

**CatBoost as the classifier learner (scratch, 4 seeds, twelve-month leave-one-month-out; closed).** Same inputs, depth 4, learning
rate 0.03, l2 5, early stopping on the same date-grouped inner folds: control 19.018e9 (LightGBM 19.007e9; AUC 0.926 vs 0.915) and
`pa` 18.106e9 (LightGBM 18.210e9; 8/12 months against its own control, March +11%, December +9%). A higher AUC, a 0.6% lower SSE
and the same month pattern: not a learner effect worth a round, and well above round 3's selected 17.646e9. Closed.

## v43 official: 227.6767 (00:41 UTC, 29 September) — 3.218 s WORSE than v40; v40 remains the official best

Uploaded at the user's word at 00:40 UTC (the first slot of 29 September UTC); processed 00:41:01 UTC, 344,841 rows
(`official_result_v43.json`). The squared error rose by 5.0e8 against a sum of squared moves of 2.6e8, i.e. the moves were aligned
*against* the residuals on average (Σ d·r ≈ +3.8e8), not merely noise (which would have cost about +1.7 s). No row-level attribution
is attempted (no labels). **Reading:** the second population-classifier change to pass its development protocol and lose on the
ranking (v10: folds −0.38% / −0.58%, ranking +0.52 s; v43: twelve-month LOMO −7.4%, bootstrap below zero, ranking +3.22 s). The 2026
population rows do not behave like 2025's under signals learned from 2025's labels; the twelve-month protocol, the seed bag and the
bootstrap did not protect against it. **The population-classifier lever is closed**: no v42 upload (its delta correlates 0.57 with
v43's), no round 4, no runner layer. Never re-upload v1–v43.
