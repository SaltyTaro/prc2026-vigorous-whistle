# Final verification of the final file before upload (13 October 2026)

**Verdict: PASS.** The file is candidate 6, `runs/final_blind/final_candidate6/chain_out_pm.parquet`, SHA-256
`a7ac1f60c5ddd79ef6927887b50471b3160748c386ed8186d99095bee6b63a81`. It is ready to upload as `vigorous-whistle_final.parquet`, and the
upload is the user's decision. The checks are scripted in [`tools/verify_final_file.py`](../../tools/verify_final_file.py). The cap and
published-means checks recompute those two steps from the supplied files without the code that built them. The tool also re-runs both
tools from the committed code. The full report is [`final_verification.json`](final_verification.json).

    .venv\Scripts\python.exe -X utf8 -B tools/verify_final_file.py --rerun-dir <empty scratch directory>

## What was checked

| check | result |
|---|---|
| **inputs** | `final_ranking.parquet` `4b0692a7...`, `final_submitting.parquet` `b3445d80...` and the EUROCONTROL workbook `82d119ed...` carry the hashes the chain recorded. |
| **contract** | 670,790 rows. Schema equals the template's (`MVT_ID_mvt` double, `TAXITIME_SEC_mvt` int32). IDs are in template order and unique, and every ID is a departure of `final_ranking`; no departure is left without a prediction. No nulls. Range -12 to 111,666 s: none below the -12 s floor, 73 at or below zero (72 LSZH, 1 LTFM), 155 above 7,200 s, 37 above 20,000 s. |
| **lineage** | Every stage file has its recorded hash: model part `cdb81e08...`, early chain `4c393769...`, early chain with the six-month stacker `d5834fcb...`, late chain `ab25a196...`, candidate 4 `8fc8828d...`, candidate 6 `a7ac1f60...`. Each receipt names the previous stage's hash as its base and records the final ranking file and workbook used. |
| **published means** (independent) | From candidate 4, round(max(value - shift, -12)) with the receipt's four shifts, on non-fixed rows valued in (0, 7,200] s of each eligible cell, equals candidate 6 on every row. The cells: EGLL February -2 s (17,718 rows), EGLL June -8 s (19,661), EGLL July -8 s (20,553), LIRF July non-population -16 s (15,555). 73,487 rows change in total, squared change 6.63e6. No row changes outside these cells, and no fixed row changes. |
| **re-run from committed code** | `tools/run_lirf_cap.py apply` on the late-chain output and `tools/run_published_means_v4.py apply` on its result reproduce candidate 4 and candidate 6 **byte for byte**. |
| **code** | HEAD `0736e03`. Every tool and declaration of the chain (list in the tool) is tracked and unmodified. |
| **tests** | 457 of 460 pass; the other 3 need drive E: (below) |
| **uploader** | check-only passes at 15:57 UTC (below) |

## January / July against the uploaded v76

All 344,841 v76 rows are in the file. |candidate 6 - v76| has median 6 s, 90th percentile 21 s, 99th percentile 64 s and 99.9th
percentile 165 s; the RMS difference is 25.2 s and the mean difference -1.4 s. The differences come from:

- the ±12 s jitter of the final file;
- the pins being off;
- the six-month stacker, second and third passes;
- the cap;
- the published-means shift.

Ten rows differ by more than 1,000 s. Nine are LIRF fill-population rows: seven pin rows, ITYTY672 and RYRR388G. The tenth needed an
explanation:

- The no-NM ADS-B rematch learns its
  flight-number → callsign map from the NM rows of the movement file. The match is a gate-traced track at a known stand: 126 points, 703 s from first motion
  to take-off. Through the stages: model part 2,100 s, after the own-points member 1,464 s, after the rematch 1,084 s, final 964 s.
  On the ranking file the member moved this row up instead (2,107 → 2,264 s). Whether the member's change also comes from the new
  match was not traced.
  On January / July only three rows gain a match this way, all at EHAM, with a squared change of about 1.1e6. No row loses a match.
  The 343 rows matched on both files show observed taxi times equal to within 12 s, which is the jitter. This is the declared procedure
  applied to a larger file, not a fault.

Its NM clocks fall one day before take-off. Its value of
85,960 s matches every upload since v55 to within 15 s (85,975).

## Monthly level against EUROCONTROL's published means

`tools/final_month_means.py` on candidate 6 flags 3 of 40 airport-months: **January 2026 at EDDF, EDDM and LSZH** (gaps 179 / 255 / 189 s
against 2025 maxima 53 / 164 / 84 s). January 2026 was the snow month. v76 shows the same three on the board months, so the flags come
from the model's reading of those months, not from the final-file pipeline. **No February or June airport-month is flagged.** The LIRF
July flag of candidates 1-4 (gap 44 s) is gone, because the published-means shift brings that gap to 28.5 s.

Against the same calendar months of 2025, the predicted level (mean of ordinary predictions in (0, 3,600] s) moves within the year as
the published figures do. The largest February / June-versus-January / July contrasts are January 2026 at EHAM / LSZH / LTFM / EDDF /
EDDM (+119 to +150 s above January 2025): the snow and disruption days that the daily published delay totals anchor.

## Tests and uploader

- **Test suite, 13 October:** 460 tests are collected. **457 pass.** Three tests in `tests/test_final_runner.py` fail with
  `FileNotFoundError`: `test_adj2_needs_an_explicit_refit_for_other_months`, `test_v9_stage_reproduces_the_recorded_ranking_v9_on_a_sample`
  and `test_adj2_refit_for_a_new_month_is_saved_and_loadable`. They read `runs/adsb_v25/export_v23/fit_record.json` and
  `runs/azure_arcw0_surface_refit/.../pipeline.json`, which sit behind junctions to the USB drive E:, and E: was not connected. Nothing the
  final file or the upload reads lives on E:. Re-run these three when the drive is back.
- **Uploader** (`tools/prc_submit.py upload-final --check-only`, 15:57 UTC): `checked_not_uploaded`, 670,790 rows, no final object in the
  team bucket. `final_ranking.parquet` / `final_submitting.parquet` carry the ETags of our download.
- Verifier run at 15:57 UTC: PASS, no failures; both tool re-runs byte-identical.

## What was not re-run

The model part, members, early chain and late chain (steps 1-18) were not rebuilt today. Their outputs carry the recorded hashes, and
the chain reproduced v69-v76 byte for byte on `ranking.parquet` (`lane_E.md`, `lane_L.md`). Rebuilding them takes hours of VM time.
