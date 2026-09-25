# Coverage-regime development set (25 September 2026): the ranking regime costs 1.2–1.7%, mostly unavoidably

**Question.** The scored months (January and July 2026) have different adsb.lol coverage from the 2025 months every
selection was developed on (`docs/adsb_coverage/`). How much does the shipped v26 recipe lose when its ADS-B inputs
come from the ranking months' regime, and can training on that regime win part of it back?

**Answer.**
- **Penalty.** Thinning December 2025 to January 2026's coverage (EDDF only) raises the v26 development RMSE by
  **+1.24%** (172.87 → 175.02 s). EDDF's own RMSE rises **+39.5%** (86.8 → 121.1 s).
- **July.** Thinning July 2025 to July 2026's coverage (EDDF, LSZH, LIRF dark) raises it by **+1.72%**
  (266.03 → 270.59 s). EDDF +26.3%, LSZH +1.7%, LIRF +2.1%.
- **What this means for development scores.** Our development numbers at EDDF overstate what the ranking months can
  deliver by roughly a third.
- **Avoidable part.** A residual LightGBM trained on the same thinned regime recovers **16.8% (December) / 20.5%
  (July)** of the residual stage's same-row penalty on the thinned airports. At the recipe level that is
  **−0.01% / −0.02%** of month RMSE.
- **Rule.** The declared rule (recovery ≥ 25%) is not met, so no coverage-augmented committee or network round is
  recommended.
- **Extrapolation, not measured.** At the same recovery rate, retraining the committee and networks on thinned copies
  would be worth about 0.06–0.13% of whole-month RMSE.

Nothing was selected, exported or uploaded. Only local compute was used: no Azure worker, no AWS, no leaderboard read.

## Method

**Declaration:** [`configs/coverage_regime_v1.json`](../../configs/coverage_regime_v1.json), 03:43 UTC, before any
development label was read. Everything before it was label-free and is disclosed there: the regime statistics,
thinning attempts a1–a13 and the matcher reproduction checks.

**Thinning** (`src/prc2026_regime/thinning.py`, `tools/regime_thin_match.py`):
- Thinning can only remove ground points; airborne points are never touched.
- **Cell ratios.** Per airport, the tool counts ground points inside matched departure segments per departure on a
  100 m grid, in the 2025 source month and in the 2026 target month.
- **Shrinkage and calibration.** Each cell's target/source ratio is shrunk toward its 400 m cell and the airport
  ratio (prior 50). One factor per airport then scales the ratios so the expected kept total equals the target total.
- **Coverage versus rate.** Each ratio is split into two parts. Aircraft runs through a 400 m cell are kept with the
  cell's visit ratio, and points within a kept run with the remaining rate ratio. This separates lost coverage (areas
  2026 does not see) from a lower message rate (areas it sees less often).
- **Whole-aircraft drop.** A share of aircraft-days is removed whole: January 2026 has 8% of EDDF departures never
  seen on the ground at all.
- **Matching.** The frozen matcher (`prc2026_adsb.features.match_departures`) re-matches the thinned points under
  both frozen stand tables. The natural matcher reproduces the stored tables row for row (2025-12-03, 2025-07-10).
- **Attempts.** Thirteen attempts are recorded under `runs/coverage_regime/a*/`. Independent point thinning left
  segments too long and too many gate stops; whole-session drops were bypassed by the matcher's fallback to the
  arrival session. 100 m visits mistook a lower rate for lost coverage, and uncalibrated ratios lost the mass of cells
  that 2026 covers better.

| regime | airport | matched % (2025 / thinned / 2026) | gate % | points per segment | dark first point (s) |
|---|---|---|---|---|---|
| 2025-12 → 2026-01 | EDDF | 94.8 / 79.4 / 79.5 | 38.0 / 7.2 / 7.8 | 51 / 13 / 15 | 570 / 445 / 451 |
| 2025-07 → 2026-07 | EDDF | 91.7 / 64.9 / 76.1 | 16.7 / 4.3 / 3.8 | 32 / 7 / 10 | 480 / 496 / 457 |
| 2025-07 → 2026-07 | LSZH | 91.1 / 90.3 / 93.9 | 28.8 / 19.1 / 18.2 | 73 / 34 / 45 | 441 / 413 / 433 |
| 2025-07 → 2026-07 | LIRF | 93.9 / 0.6 / 0.2 | 6.8 / 0.0 / 0.0 | 34 / 4 / 9 | – |

- **December regime: matched.** Other airports stay natural; their January 2026 shares are within 4 points of
  December's.
- **July regime: EDDF is harsher than July 2026.** July 2026 sees parts of EDDF (runway areas) *better* than July 2025
  did: 54.7% of its departure-route points fall in cells whose ratio exceeds 1. Removal cannot recreate them, so about
  11 points of departures that July 2026 sees as dark are unmatched here. EDDF July penalties are an upper bound.
- **Airports left natural in July.** EDDM, EGLL and LEMD already have *less* coverage in July 2025 than in July 2026,
  so they cannot be moved toward it.

**Features** (`tools/regime_features.py`): the committee's ADS-B and neighbour table, the residual stage's table, and
the eleven v19 features, with every day's surface thinned exactly as in the attempt. The same code on natural inputs
reproduces the cached tables exactly: every v19 feature, the residual table, and the July committee table differ by
0.0.

## Step 2: what v26 loses in the ranking regime

`tools/regime_measure.py` re-predicts every ADS-B-dependent member of v26 twice, from natural and from thinned
features, with models that never saw the month:
- **Blend tables:** 22 committee trees, 32 committee networks, 7 LIRF trees and 16 LIRF networks. December uses the
  January–November refit models; July uses the seasonal-fold models.
- **Residual networks:** 16, from their leave-one-month-out models.
- **adj2:** refitted on the other eleven months with the v25 settings.
- **cat** (5% of the residual): its thinned value is approximated by its stored value plus `adj2`'s thinned-minus-natural
  change (declared).

The recipe is assembled with `tools/run_adsb_rnn_dev.py`'s own functions.

**Reproduction of the natural rebuild**
- The recipe reproduces the recorded v26 (`rnet_w90`) development RMSE: 172.87463 vs 172.87525 s (December) and
  266.02842 vs 266.02837 s (July), per airport within 0.003 s.
- December trees reproduce their stored predictions exactly.
- December networks differ on about 21 rows: their stored inputs were built on the VMs, and `nb_runway_prev`/`next` are
  not bit-reproducible (the unstable sort).
- The refitted `adj2` differs from the stored one by 2.5–2.7 s RMS (thread count). The same refit is used on both
  sides, so the comparison is unaffected.
- A few dozen rows at untouched airports change through that same neighbour-feature rebuild noise, at most 0.012% RMSE.

| month | whole month | non-LIRF | EDDF | LSZH | LIRF | penalty SSE share: blend tables / residual + routing |
|---|---|---|---|---|---|---|
| 2025-12 (→ Jan 2026) | 172.87 → 175.02 (**+1.24%**) | +1.74% | 86.8 → 121.1 (+39.5%) | – | – | 50% / 50% |
| 2025-07 (→ Jul 2026) | 266.03 → 270.59 (**+1.72%**) | +1.37% | 124.3 → 157.0 (+26.3%) | 137.7 → 140.0 (+1.7%) | 632.2 → 645.7 (+2.1%) | 83% / 17% |

Where the EDDF penalty sits (share of EDDF's SSE increase):

| natural → thinned class | December rows | RMSE | share | July rows | RMSE | share |
|---|---:|---|---:|---:|---|---:|
| gate → gate | 1,146 | 62.0 → 80.8 | 2.5% | 853 | 61.7 → 75.1 | 0.8% |
| gate → dark | 4,389 | 59.8 → 107.0 | 27.9% | 1,812 | 66.7 → 110.5 | 7.2% |
| gate → unmatched | 1,045 | 71.9 → 166.6 | 19.1% | 894 | 69.2 → 157.8 | 9.2% |
| dark → dark | 8,095 | 89.2 → 107.7 | 23.9% | 10,917 | 106.8 → 125.9 | 24.8% |
| dark → unmatched | 1,645 | 90.0 → 165.1 | 25.5% | 5,018 | 121.6 → 192.4 | 56.8% |
| unmatched → unmatched | 880 | 168.7 → 172.7 | 1.0% | 1,574 | 263.8 → 266.9 | 1.3% |

- **Losing the match or the stand stop** (gate → dark, → unmatched) is lost information; no model recovers it.
- **Rows that keep their class still get worse** (dark → dark +21%, gate → gate +30% in December). Thinner traces are
  read by models trained on dense ones; this is the part that training on the regime could address.
- **LIRF going dark costs little.** Its July error is dominated by schedule-fill rows that ADS-B barely helps with.

## Step 3: how much is avoidable (residual stage)

`tools/regime_avoid.py` thins the eleven other months toward the evaluated month's target regime with the declared
settings. Their EDDF lands at 77–80% matched / 6–8% gate (December regime), and 63–71% / 4–5% plus LSZH gate 19–28%
(July regime). It then refits `adj2` two ways:
- **mix:** natural rows plus a thinned copy of the thinned airports' rows;
- **thinned:** those rows replaced by their thinned version.

| month | trained on | thinned month: scope RMSE | thinned airports' scope | pure-adj2 recipe | natural month: scope |
|---|---|---:|---:|---:|---:|
| 2025-12 | natural | 124.381 | 113.534 | 175.648 | 120.710 |
| 2025-12 | mix | 124.184 | 108.680 | 175.630 | 121.163 |
| 2025-12 | thinned | 124.218 | 108.692 | 175.607 | 122.110 |
| 2025-07 | natural | 163.751 | 117.905 | 270.856 | 158.389 |
| 2025-07 | mix | 163.006 | 114.780 | 270.797 | 158.379 |
| 2025-07 | thinned | 163.027 | 114.829 | 270.805 | 159.362 |

**Recovery by `mix`, on the rows matched in both versions**
- **December, EDDF, 13,727 rows:** natural features 84.63 s, thinned features 113.52 s. `mix` takes back 4.85 s:
  **16.8%**.
- **July, EDDF + LSZH, 25,035 rows:** 102.18 → 117.42 s; `mix` takes back 3.13 s: **20.5%**.
- The recovered seconds are measured on the thinned scope, which differs from the same-row set by 27 and 199 rows.
- **Cost on the natural month.** `mix` costs +0.38% (December) / −0.01% (July) there; `thinned` costs +1.16% / +0.61%.
- **Recipe level.** The pure-`adj2` recipe moves −0.010% / −0.022%. The residual stage carries 0.3 of matched rows'
  predictions, and EDDF is a tenth of the rows.

## Decision and recommendation

**Declared rule:** (a) the penalty exceeds 0.3% in either regime: **yes**, 1.24% / 1.72%. (b) Step 3 recovers at
least a quarter of the residual stage's penalty on the thinned airports: **no**, 16.8% / 20.5%.

**Therefore no coverage-augmented training round is recommended.** The ranking regime's cost is recorded as mostly
unavoidable with the current data.

**Size of what such a round could buy (extrapolated, not measured).** At EDDF (and LSZH in July), the blend tables
carry 0.62% (December) / 0.38% (July) of whole-month RMSE of the penalty. Recovering 17–21% of it is **0.06–0.13%**;
the residual stage adds about 0.02%. It would need:
- the committee (22 trees) and 32 networks retrained on the VMs;
- thinned committee, v19 and residual tables for all twelve months under both stand tables.

That is comparable to recent seed and blend rounds, not a big lever.

**What is useful regardless**
- **Development scores at EDDF are optimistic by a quarter to a third** relative to the ranking months, and the
  whole-month development RMSE by 1.2–1.7%. Candidates whose gain comes from EDDF gate rows will transfer worse than
  their development reading.
- **The thinned December and July tables are a ranking-regime check for any future candidate.** They live under
  `runs/coverage_regime/a13_dec/`, `a13_jul/` and `data/external/adsblol/regime/`. Scoring a candidate on
  `recipe:thinned` from `tools/regime_measure.py`, next to its natural score, shows whether a gain survives the
  ranking coverage.

## Records and deviations

- **Results:** [`penalty_2025-12.json`](penalty_2025-12.json), [`penalty_2025-07.json`](penalty_2025-07.json),
  [`avoidability_2025-12.json`](avoidability_2025-12.json), [`avoidability_2025-07.json`](avoidability_2025-07.json),
  [`results.json`](results.json).
- **Row-level predictions:** `runs/coverage_regime/measure/<month>/predictions.parquet`. Tables print with
  `tools/regime_summary.py`.
- **Calibration cap (after the declaration).** Training-month preparation hit a source month whose total was below the
  target (June 2025, LSZH, 10 missing archive days). The calibration factor there is now a finite 1e9, keeping every
  point, instead of infinity. The declared attempts never reach that branch, so their keep tables are unchanged.
- **Out-of-memory incident.** Three thinning attempts were first launched in parallel (36 worker processes) and ran the
  15.7 GB workstation out of memory around 03:20 UTC. They were killed and rerun one at a time with 6 workers; later
  jobs ran with at most 12 threads and 3–6 workers.
- **Not measured.** The CatBoost residual member was not refitted (approximated, 5% weight). The committee and networks
  were not retrained on thinned data. LIRF is outside the residual stage, so its July darkness enters step 2 only.
