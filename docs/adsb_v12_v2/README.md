# v12: ADS-B residual without LIRF, with neighbour context, weight 1.0 (15 September 2026)

Declared at **09:52 UTC** in [`../../configs/adsb_v12_v2.json`](../../configs/adsb_v12_v2.json),
after v11 (official 248.4368, 4th) and the null result of `adsb_v12_v1`
([`../adsb_v12/README.md`](../adsb_v12/README.md)). Same residual recipe as v11
(LightGBM on label − v9, leave-one-month-out over Jan/Jul/Nov 2025, rounds from
a day-grouped inner split) with two changes declared a priori:

1. **LIRF is excluded from the residual scope** (its rows keep v9): the LIRF
   label is dominated by the documented BLOCK=SCHED fill artifact, the ADS-B
   pushback does not describe it, and v11 made LIRF worse on both months with
   coverage (+1.7% Jul, +0.4% Nov). A what-if on the recorded v11 development
   predictions with LIRF kept at v9 gave 0.00% / −0.74% / −0.15%.
2. Candidates with the v11 feature set (`v11`, 22 features) and with the eleven
   `nb_*` neighbour features added (`nb`, 33 features), at weights 0.75 and 1.0.

Baseline for selection: the recorded v11 development predictions
(333.914 / 277.689 / 200.421). Rule: not worse than v11 on any month and at
least 0.3% better on average; lowest mean ratio wins. Inputs: the v11 matches
and stands (`data/external/adsblol/matches_v11/`, hash-bound).

## Development ([`development_results.json`](development_results.json), `runs/adsb_v12_v2/selection.json`)

| Candidate | Jan 2025 | Jul 2025 | Nov 2025 | mean |
| --- | ---: | ---: | ---: | ---: |
| v11 baseline | 333.914 | 277.689 | 200.421 | — |
| `v11feat_no_lirf_75` | 333.320 (−0.18%) | 275.461 (−0.80%) | 199.265 (−0.58%) | −0.52% |
| `v11feat_no_lirf_100` | 333.044 (−0.26%) | 274.569 (−1.12%) | 198.663 (−0.88%) | −0.75% |
| `nb_no_lirf_75` | 333.382 (−0.16%) | 275.469 (−0.80%) | 199.138 (−0.64%) | −0.53% |
| **`nb_no_lirf_100`** | **332.993 (−0.28%)** | **274.585 (−1.12%)** | **198.544 (−0.94%)** | **−0.78%** |

All four candidates are eligible; **`nb_no_lirf_100`** is selected. Paired-day
bootstrap of the RMSE difference to v11 (descriptive): Jan [−1.57, −0.60] s,
Jul [−5.0, −1.15] s, Nov [−2.56, −1.33] s. With LIRF out of the scope, weight
1.0 beats 0.75 on every month (v11 had chosen 0.75 because of LIRF). Per
airport on Nov 2025 versus v11: EDDM −14.0%, EHAM −7.6%, EDDF −6.1%, LSZH −3.9%,
LEBL −2.7%, EGLL +1.1%, LEMD 0.0%, LIRF −0.4% (back to v9), LFPG/LTFM 0.
Refit rounds 1,173 (median of 1,152 / 1,455 / 1,173).

## Export ([`export.json`](export.json))

`tools/export_adsb_v11.py --config configs/adsb_v12_v2.json --selection
runs/adsb_v12_v2/selection.json --version 12`: v9 rebuilt from scratch
(rounding equals the uploaded v9 on every row); residual fitted on the 271,115
in-scope rows of the three months; applied to the **216,435 in-scope ranking
departures** (matched rows outside LIRF: EGLL 37,034, EHAM 36,040, LEMD 31,129,
LEBL 28,832, EDDF 28,163, EDDM 25,173, LSZH 21,747, LFPG 8,315, LTFM 2, LIRF 0).
Output `runs/submissions/vigorous-whistle_v12.parquet`, SHA-256
**`f3ae96a47a3f9c417d11d93285a393ccd5e7411c9d99b65849867980914471e6`**, 344,841
rows, min −12, max 120,549 (inherited). Versus v11: 222,555 rows differ, median
absolute change 10 s, 99th percentile 159 s, RMS of the change 43 s; the 8,843
LIRF rows revert to v9. Model and features under `runs/adsb_v12_v2/export_v12/`.

Honest expectation for the official score: **−0.5% to −1.0% versus v11
(≈ 246.0–247.2)**; third place is 247.0109.

## Verification ([`verification.json`](verification.json))

`tools/verify_adsb_v11.py --version 12 --config configs/adsb_v12_v2.json
--run-root runs/adsb_v12_v2`: receipts and parents re-checked, v9 reproduced
from scratch, the 27 ADS-B features reproduced from the saved matches, the
residual recomputed from the saved booster (difference 0.0), the feature list
identical between declaration, saved record and booster, rows outside the scope
equal the uploaded v9, and the independent per-row segment re-derivation agrees
on **300/300** sampled matched departures. Not uploaded at the time of writing.
