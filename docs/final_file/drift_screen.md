# Drift screen of the final file's new months (10 October 2026)

Label-free check that February and June 2026 do not run outside what the models saw, before the single final upload of candidate 1
([`final_candidate.md`](final_candidate.md), SHA-256 `30e116a9...`). Tool `tools/final_drift_screen.py`; results
[`drift_screen_results.json`](drift_screen_results.json) (SHA-256 `704d019a...`). Run on arcw1 from a separate folder holding hash-checked
copies of the inputs. The workstation was not used because its free memory was 1.3 GB, with the user's other jobs running. arcw1 was
deallocated afterwards.

## Method

For each airport and each quantity:
- **New months.** February / June 2026 are compared with February / June 2025.
- **Reference ("drift we already survived").** January / July 2026 (scored board months) are compared with January / July 2025.
- **2025 range.** Each 2026 value is placed in the range of the twelve 2025 months.

**Flag rule.** A February / June 2026 value is flagged when both hold:
- it lies outside the 2025 range;
- it moved further from its 2025 analogue than both board months moved from theirs.

**Quantities:**
- departures per day;
- ADS-B match share and gate-traced share (the runner's definitions; 2025 from `matches2_2025-*`);
- NM record share (`ADEP_flt`) and NM clock share (`AOBT_3_flt` or `LOBT_flt`);
- the EOBT_1 > IOBT share. This is the repair detector's statistic; the 2026 values equal `repair/shares.parquet` exactly.
- LIRF fill-population departures per day (no NM clock and no airport aircraft type, `population_rows`), and its long-interval part
  (schedule interval > 43,200 s);
- shares of departures at stands or with aircraft types unseen in 2025;
- the distance of runway shares from 2025;
- the candidate's predicted distribution (mean, p50, p90, p99, share > 3,600 s) against the 2025 labels of the same calendar month.

## Result: 19 flags, none warrants a change before the upload

| flag | value (2025 analogue; 2025 range; board months' drift) | touches | matters? |
|---|---|---|---|
| **EDDF Feb ADS-B match** | 0.396 (0.964; 0.60-0.97; 0.157) | every ADS-B layer at EDDF in February | Uniformly lower all month: 21-54% per day, median 39%, no dead day; June is back at 0.83. About 60% of EDDF February departures take the unmatched route, which is how every unmatched departure is handled (coverage-thinned training). EDDF February will be less accurate than January. Nothing to change. |
| EGLL Jun ADS-B match / gate | 0.945 / 0.455 (0.113 / 0.115; 0.10-0.73 / 0.11-0.28; 0.742 / 0.327) | ADS-B layers at EGLL | Coverage rose, as in July 2026 (board month, already scored); June sits just beyond July's move. No change. |
| EDDF Jun unseen stands | 1.48% of departures (0.005%; max 0.03%; 1.44%) | stand features | The same new stands as July 2026 (board month). No change. |
| LEMD Jun departures per day | 644.5 (599.7; max 609.4; 43.0) | demand / queue features | +7.5% traffic, as in July 2026 (+43). The monthly taxi-out check passes for LEMD June. No change. |
| LEMD Feb EOBT_1 > IOBT share | 0.220 (0.176; max 0.206; 0.020) | EOBT_1-based features; the v31 repair rule | Above 2025, but below the repair detector's threshold (0.237; margin -0.018 in `repair/shares.parquet`). The declared rule leaves it unedited. No change. |
| EDDM Feb prediction tail (p99 2,887 s; share > 3,600 s 0.28%) | labels 2,219 s / 0.010%; 2025 max 2,579 s / 0.09% | predictions | 28 of the 30 rows above 3,600 s are on **19 February 2026**: EUROCONTROL's published pre-departure delay that day is 39.1 min per departure, against about 8 on a typical February day. The rows are ADS-B-matched, without an NM clock, and the model part already predicts them high (mean 4,237 s at v55m, 4,410 s final). A disruption day, backed by tracking and the published figure. No change. |
| LFPG Feb prediction tail (p90 1,257 s; p99 1,874 s) | labels 1,444 / 2,710 s; 2025 min 1,271 / 1,912 s | predictions | Predictions are narrower than labels, more so at LFPG, where coverage is thin (13% in February). The monthly mean check passes (gap 31.5 s, inside 2025). No change. |
| LEMD Jun predicted median | 1,024 s (994 s; max 1,013 s; 20 s) | predictions | Consistent with the +7.5% traffic. The monthly mean check passes. No change. |
| LEMD Feb ADS-B match | 0.967 (0; 0-0.967; 0.962) | — | **2025 artefact.** The 2025 match files have no LEMD departure in February 2025. 2026 sits at the top of the 2025 range. |
| LIRF Jun gate share 0 | — | — | LIRF June 2026 has no ADS-B coverage, a "dark" month like July 2026. The dark-month rule detects it from the file (0% match). |
| LTFM Jun ADS-B match | 0.018% (0) | — | About four LTFM rows matched; LTFM had none in 2025. Negligible. |
| unseen types / stands (EDDM Feb, EHAM Feb, LEMD Jun, LIRF Jun) | ≤ 0.05% of departures | — | Negligible (EHAM February has 0, below 2025's minimum). |

NM record and clock shares, the LIRF fill-population size and the runway distribution raise no flag.

## Note

The screen reads no 2026 label and no official score. The published daily delay is quoted only to explain the EDDM day; it changes
nothing. Candidate 1 stays as built.
