# v11: ADS-B surface residual on v9 (15 September 2026; built and verified, awaiting upload approval)

Declared at **08:21 UTC** in [`../../configs/adsb_v11_v1.json`](../../configs/adsb_v11_v1.json)
after the exploratory work recorded in [`../opensky/README.md`](../opensky/README.md).
External data: **adsb.lol `globe_history` daily archives (ODbL)**; surface positions
at the ten aerodromes are extracted per day (`tools/adsb_extract_day.py`, part
hashes in each day's manifest), matched to departures by ICAO callsign and takeoff
time (`tools/build_adsb_features.py`; stand positions learned label-free from
covered segments of Jan/Mar/Apr/Jul/Nov 2025: 1,111 stands), and turned into 16
label-free features (`src/prc2026_adsb/features.py`). No departure label or block
time enters any feature; December is never read.

## Development (leave-one-month-out, `tools/run_adsb_v11.py`)

Residual LightGBM (label − v9) fitted on the matched rows of two of the three
scoring months that hold out-of-sample v9 predictions, applied to the matched
rows of the third; rounds from a day-grouped inner split; every other row keeps
v9. [`development_results.json`](development_results.json)
(`runs/adsb_v11/selection.json`).

| Held-out month | matched | v9 | `residual_matched_100` | **`residual_matched_75`** | `residual_matched_50` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jan 2025 | 50% | 343.873 | 333.548 (−3.00%) | **333.914 (−2.90%)** | 335.776 (−2.35%) |
| Jul 2025 | 53% | 288.133 | 278.799 (−3.24%) | **277.689 (−3.62%)** | 278.903 (−3.20%) |
| Nov 2025 | 71% | 219.875 | 199.991 (−9.04%) | **200.421 (−8.85%)** | 204.002 (−7.22%) |

All three candidates clear the 0.5% rule on every month; **`residual_matched_75`**
has the lowest mean ratio and is selected. Per airport on Nov 2025 (the month
whose coverage resembles 2026): EDDM −41%, EHAM −32%, EDDF −31%, LSZH −26%,
LEBL −22%, LEMD −16%, EGLL −10%; LFPG −1%, LIRF 0%, LTFM 0% (no coverage).
Jan/Jul 2025 are diluted by their artifact rows (LFPG 800 s RMSE in Jan) and by
lower 2025 coverage. Refit rounds: 873 (median of 873 / 1,186 / 639).

## Export and verification

- `tools/export_adsb_v11.py --version 11`: the float v9 is rebuilt from scratch
  (byte-identical v9 classifier; rounding equals the uploaded v9 on every row);
  the residual model is fitted on all 292,826 matched rows of the three months
  and applied to the **225,278 matched ranking departures** (65%: Jan 2026 69%,
  Jul 2026 63%; by airport EGLL 37,034, EHAM 36,040, LEMD 31,129, LEBL 28,832,
  EDDF 28,163, EDDM 25,173, LSZH 21,747, LIRF 8,843, LFPG 8,315, LTFM 2).
  Output `runs/submissions/vigorous-whistle_v11.parquet`, SHA-256
  **`98af50be5fb49d6cc42c8390b26be3220ba2ae24b693fc97c437ceb695c66b09`**,
  344,841 rows, min −12, max 120,549 (inherited). 223,979 rows differ from v9;
  median absolute change 20 s, 99th percentile 302 s. Model and features under
  `runs/adsb_v11/export_v11/`; report [`export.json`](export.json).
- `tools/verify_adsb_v11.py --version 11` ([`verification.json`](verification.json)):
  receipts and parents re-checked, v9 reproduced from scratch, features
  reproduced from the saved matches, residual recomputed from the saved booster
  (difference 0.0), unmatched rows equal the uploaded v9, and an independent
  per-row re-derivation of the surface segment agrees on **300/300** sampled
  matched departures.
- Tests: `tests/test_adsb.py` (two) pass with the existing suite.

Honest expectation for the official score: between the Jul and Nov figures,
about **−5% to −7%** (≈ 250–254), which would be fourth place; third is 247.0.
The ranking has the artifact rows of a seasonal set but 2026 coverage. Nothing
has been sent to the organisers; the external data is openly licensed (ODbL).
