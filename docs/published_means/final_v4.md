# Final candidate 6 (12 October 2026): candidate 4 + the published-means layer (`configs/published_means_v4.json`), NOT uploaded

**What it is.** Final candidate 4 (`docs/sixmonth/final_candidate4.md`, `8fc8828d...`; its January / July reading
`docs/sixmonth/devmonths.md` keeps it in every month) with `configs/published_means_v3.json`'s `pm_np` procedure applied last, on all four
months of the final file. Declared as a **disclosed override** in [`configs/published_means_v4.json`](../../configs/published_means_v4.json)
(3008494) before the file was written: v3's procedure was **not accepted** on 2025 (January +0.0006%, one LEBL cell overshooting; July
-0.0013%, November -0.0015%, December -0.0050%). The cells it would select on candidate 4 were explored before the declaration (disclosed there).
No official score is used. Tool: `tools/run_published_means_v4.py apply --base <candidate 4> --out <candidate 6>`.

| file | SHA-256 | rows |
|---|---|---:|
| **candidate 6** `runs/final_blind/final_candidate6/chain_out_pm.parquet` | `a7ac1f60c5ddd79ef6927887b50471b3160748c386ed8186d99095bee6b63a81` | 670,790 |
| base: candidate 4 `runs/final_blind/final_candidate4/chain_out_cap.parquet` | `8fc8828d1f5338bda08972f16af61c0f5f288b5d50992c1750d4d768d89b0a42` | 670,790 |
| fallback: candidate 3 `runs/final_blind/final_candidate3/chain_out_cap.parquet` | `b673791dfdaf8e909175001387183946236075280c9b3457822b52157c9cebc8` | 670,790 |

## Cells (receipt [`final_v4/chain_out_pm.receipt.json`](final_v4/chain_out_pm.receipt.json))

Level = our mean minus the mapped published mean (non-fixed rows; LIRF without its fill population), less the method bias; sigma = the
airport's mapping / method noise; eligible if |level| >= 3 sigma (and sigma <= 5 s, u inside the 2025 range).

| cell | effective level (s) | sigma (s) | shift (s) | rows moved |
|---|---:|---:|---:|---:|
| EGLL February 2026 | +2.4 | 0.6 | -2.2 (rounded -2) | 17,718 |
| EGLL June 2026 | +8.1 | 0.6 | -8.05 (-8) | 19,661 |
| EGLL July 2026 | +8.5 | 0.6 | -8.44 (-8) | 20,553 |
| LIRF July 2026 (non-population) | +17.5 | 4.7 | -16.26 (-16) | 15,555 |

No other cell is eligible (on candidate 4, LIRF January / February / June read -10.7 / -9.6 / -8.2 s, below 3 sigma; EDDF, EDDM, LFPG, LTFM
have sigma above 5 s; January EDDF / EDDM / LSZH have u outside the 2025 range). Expected squared-error gain 7.6e6 s^2 if the estimates hold
(about -0.026 s on the four-month RMSE, about -0.04 s on its January / July part); the same loss if every move is noise. On 2025, EGLL's
shifts tracked the true level error within 1.2 s in every development month (v2, v3); LIRF July 2025 under v2: shift 19.2 s against a true
12.4 s.

## Checks (label-free)

- 670,790 rows, exact `final_submitting.parquet` order, int32, no NaN, range -12 to 111,666.
- Against candidate 4: 73,487 rows change, all in the four cells (EGLL -2 / -8 / -8 s, LIRF July -16 s, each a constant), squared change
  6.63e6, max 16 s; 0 rows outside the cells; 0 of the 619 LIRF fill-population rows; guard rows 213142471 / 213142469 keep 25,391 / 51,222;
  cap rows 213145579 / 213152353 keep 111,666 / 88,392.
- Uploader rehearsal `tools/prc_submit.py upload-final --check-only`: `checked_not_uploaded`, 670,790 rows, organiser ETags unchanged, no
  final object in the team bucket ([`final_v4/vigorous-whistle_final.check_c6.json`](final_v4/vigorous-whistle_final.check_c6.json)); the
  shared `runs/submissions/vigorous-whistle_final.check.json` was restored byte for byte afterwards (`e7633168...`).

## Upload (only at the user's word; one upload, never overwritten)

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate6/chain_out_pm.parquet --expected-sha256 a7ac1f60c5ddd79ef6927887b50471b3160748c386ed8186d99095bee6b63a81

Without the published-means layer (candidate 4):

    .venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py upload-final --source runs/final_blind/final_candidate4/chain_out_cap.parquet --expected-sha256 8fc8828d1f5338bda08972f16af61c0f5f288b5d50992c1750d4d768d89b0a42
