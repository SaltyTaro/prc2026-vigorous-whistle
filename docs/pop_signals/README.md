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

**Reading.** The previous-arrival identity is a real, structural separator (a −2.7% to −4% population-SSE effect, concentrated in the
summer months that resemble July 2026), but the declared rule was set for month-level consistency and it does not clear it. Shipping
it is a user decision with the odds above; the stake on the ranking is of the order of ±0.5 s (population SSE ≈ 16–26% of the total,
times 2.7–4%), on 372 rows.
