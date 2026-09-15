# v12_v1: neighbour taxi context in the ADS-B residual (15 September 2026) — null result, nothing exported

Declared at **09:34 UTC** in [`../../configs/adsb_v12_v1.json`](../../configs/adsb_v12_v1.json)
after v11 (official 248.4368, 4th). Same residual recipe as v11 (LightGBM on
label − v9, leave-one-month-out over Jan/Jul/Nov 2025) with eleven `nb_*`
features added (`neighbour_features` in `src/prc2026_adsb/features.py`: matched
neighbours' ADS-B taxi-out statistics on the same runway within ±10/±30 min,
airport-wide ±10 min, previous/next runway neighbour), two scopes — `matched`
(rows with a surface segment) and `context` (matched rows plus rows with a
matched runway neighbour within 30 min; 85% of November rows) — and weights
1.0 / 0.75 / 0.5. Baseline for selection: the recorded v11 development
predictions (333.914 / 277.689 / 200.421), margin 0.2% on each month. Inputs
are the v11 matches and stands, snapshotted under
`data/external/adsblol/matches_v11/` and hash-bound in the declaration.

## Result ([`development_results.json`](development_results.json), `runs/adsb_v12/selection.json`)

| Candidate | Jan 2025 | Jul 2025 | Nov 2025 | mean ratio |
| --- | ---: | ---: | ---: | ---: |
| v11 baseline | 333.914 | 277.689 | 200.421 | 1.0000 |
| `nb_matched_100` | 333.518 (−0.12%) | 278.178 (+0.18%) | 200.415 (−0.00%) | 1.0002 |
| `nb_matched_75` | 333.926 (+0.00%) | 277.309 (−0.14%) | 200.838 (+0.21%) | 1.0003 |
| `nb_context_100` | 333.898 (−0.00%) | 278.775 (+0.39%) | 201.013 (+0.30%) | 1.0023 |
| `nb_context_75` | 334.520 (+0.18%) | 277.419 (−0.10%) | 201.356 (+0.47%) | 1.0018 |
| weights 0.5 | worse on every month | | | |

**No candidate is eligible; nothing is selected or exported.** Extending the
scope to unmatched rows with neighbours (`context`) is worse than leaving them
at v9. Per airport the `nb_matched_100` model does improve the well-covered
airports on every month with coverage (Nov: EDDM −9.3%, EHAM −5.6%, EDDF −4.6%,
LSZH −2.2%, LEBL −2.1%) but the gain is cancelled by LIRF (+1.0% Jul, +1.2% Nov;
36–46% of the remaining squared error) and by EGLL/LEMD (+1.4% / +0.8% Nov).
This, together with v11's own LIRF degradation (+1.7% Jul, +0.4% Nov), motivates
the next declaration ([`../adsb_v12_v2/`](../adsb_v12_v2/README.md)): LIRF
excluded from the residual scope, with and without the neighbour features.

Post-v11 decomposition used to size the levers (no model, recorded here for
the record): on Nov 2025 the remaining squared error after v11 sits at LIRF 36%,
the six well-covered airports 20%, LFPG 20%, LTFM 15%, EGLL 10%; matched
gate-covered segments 76–105 s RMSE, dark segments 107–142 s, unmatched rows
180–300 s. adsb.lol ground coverage (median ground rows per day, thousands):
EGLL 8–29 in 2025 but 47 (Jan 2026) and 131 (Jul 2026); EDDM 0 in Jul 2025;
LEMD 0 in Jan–Feb 2025; LIRF 0 in Jan–May 2025 and Jul 2026; LFPG ≤ 6; LTFM 0.
