# Stage 3 dry run (27 September 2026): v30 rebuilt from a copy of the ranking file — pipeline works

`tools/final_predict.py run --root runs/final_dryrun/run --threads 12 --workers 8 --version 900 --detail-ensembles` on a copy of
`data/raw/ranking.parquet` (runs/final_dryrun/input/final.parquet), then `tools/final_compare.py` →
[`dryrun_compare.json`](dryrun_compare.json). Workstation only; features 170 s, leaves ~25 min, 128 + 16 networks 4,023 s, residual
285 s, recipe and export about a second: **~1 h 55 min end to end**, free memory never below ~3.4 GB. The file is written only inside
the run root (version 900); nothing was copied to runs/submissions or uploaded.

**Result: v30 is reproduced to 0.031 s RMS (p99 0.022 s)**; the exported integers differ from the uploaded v30 on 1,141 of 344,841
rows, by at most 7 s. Exact or within float tolerance: ADS-B matches (both months, both consumers), v19 features, context tokens
(content hash equal), the residual adj2 and cat members (0.0), the residual networks (RMS 1.8e-4 s), 331 of 332 feature columns, scopes.

**Every remaining difference traces to floating-point noise in two inputs that flips tree splits:**
- `wx_change60_east_wind_mps` is 4.4e-16 in the frozen bundle and exactly 0.0 in the rebuild on 30 rows. Those 30 rows contain all
  11 rows where v7/v9 differ (max 0.46 s) and 30 of the 104 rows where the committee differs.
- Four ADS-B distance columns differ by at most 1.8e-12 m (a different summation order); split thresholds sitting exactly between
  repeated values flip the remaining 74 committee rows.
- The networks take the committee and v9 tables as inputs, so ~70 rows per ensemble follow (max 14–24 s on those rows); the e32
  table also carries the known GPU→CPU reload difference (RMS 0.055 s).
- All 46 rows where v30 moves by more than 0.5 s (max 6.3 s) are committee-differing rows.

Neither value is more correct than the other; for a genuinely new file this is irrelevant. For a pixel-exact reproduction of v30 the
two computations would have to follow the frozen code's arithmetic order, which is not worth doing.

**Readiness:** with the saved models, a new movement file for already-extracted dates takes about 2 h on this workstation; new dates add
the adsb.lol extraction on Azure (~1 h per month) and the IEM pull (docs/final_file/runner.md). Copying
`runs/final_dryrun/run/leaves/vocabularies.json` into a new run root skips the twelve-month vocabulary rebuild.
