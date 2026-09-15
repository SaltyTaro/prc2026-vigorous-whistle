# v12_v5: v12 blended with the mean of the LightGBM and CatBoost full-data learners (15 September 2026)

Declared at **11:48 UTC** in [`../../configs/adsb_v12_v5.json`](../../configs/adsb_v12_v5.json)
after v12 (official 246.4831, 3rd) and v12_v4 (`blend_all_30` with the LightGBM
learner alone, exported as v13). Same rule as v12_v4 — the v12 residual model,
then a fixed convex blend on every non-LIRF row (or on matched rows only) —
with the partner replaced by the per-row **mean of the two no-LIRF label
learners** (LightGBM 1,306 trees, CatBoost 5,998 trees; correlation 0.976–0.981;
`tools/ensemble_adsb_full.py`, `runs/adsb_full_ens/`). Weights 0.3 / 0.4 / 0.5 /
0.6 on all non-LIRF rows and 0.5 on matched rows. Baseline: the recorded v12
development predictions; rule: not worse on any month and at least 0.3% better
on average. All learner records, predictions and code are hash-bound.

## Development ([`development_results.json`](development_results.json), `runs/adsb_v12_v5/selection.json`)

| Candidate | Jan 2025 | Jul 2025 | Nov 2025 | mean |
| --- | ---: | ---: | ---: | ---: |
| v12 baseline | 332.993 | 274.585 | 198.544 | — |
| v13 (`blend_all_30`, LightGBM alone) | 331.826 (−0.35%) | 273.881 (−0.26%) | 195.769 (−1.40%) | −0.67% |
| `ens_matched_30` | 332.047 (−0.28%) | 273.731 (−0.31%) | 196.328 (−1.12%) | −0.57% |
| `ens_matched_50` | 331.892 (−0.33%) | 273.516 (−0.39%) | 195.605 (−1.48%) | −0.73% |
| `ens_all_30` | 331.455 (−0.46%) | 273.225 (−0.50%) | 195.388 (−1.59%) | −0.85% |
| `ens_all_40` | 331.251 (−0.52%) | 273.061 (−0.55%) | 194.773 (−1.90%) | −0.99% |
| `ens_all_50` | 331.202 (−0.54%) | 273.043 (−0.56%) | 194.381 (−2.10%) | −1.07% |
| **`ens_all_60`** | **331.309 (−0.51%)** | **273.169 (−0.52%)** | **194.212 (−2.18%)** | **−1.07%** |

All six are eligible; **`ens_all_60`** is selected by the declared rule (lowest
mean ratio, 0.98932 versus 0.98934 for `ens_all_50` — a tie in practice; the
higher weight is also the safer bet for the ranking, whose coverage resembles
November, where the gain still grows with the weight). Paired-day bootstrap
intervals to v12: Jan [−3.32, −0.88] s, Jul [−2.08, −0.71] s, Nov [−6.63, −2.84] s.
Per airport on Nov 2025 versus v12: LEMD −10.7%, LEBL −5.1%, EDDM −4.1%,
LFPG −3.9%, EHAM −3.8%, EGLL −2.9%, EDDF −1.8%, LTFM −1.0%, LSZH −0.6%.
Exported as **v14**; it supersedes v13, which is not uploaded.
