# Final blind round: inputs pre-staged before the dataset (8 October 2026)

The final dataset's two months are not known yet. Everything the runner needs that does not depend on the movement file itself is fetched
or fitted now for every 2026 month that could be in it (February-June, August-September; January and July 2026 are already on disk), so the
day the file arrives only the file-specific stages remain. Plan and decisions: [`final_round_plan.md`](final_round_plan.md).

| input | candidate months | where | status |
|---|---|---|---|
| IEM observations (`tools/pull_prc_weather.fetch`, the runner's own pull) | 10 stations x 7 months | `data/external/iem_final/prestage_2026/` + `inventory.json` (pass with `--weather-inventory`) | **done: 70 objects, 0 failures** (LFPG 2026-02 needed one retry) |
| adj2 boosters (`final_predict.py refit-adj2`, the declared ~5-minute refit; January 2026 refitted as a control against the saved booster) | 7 months | arcw5-7 `runs/final_prestage/adj2/residual/adj2/`; copy into the final run root's `residual/adj2/` | queued |
| adsb.lol data-link archive (`tools/oooi_pin_extract.py list/run --months`, `oooi_pin_pack.py`) | 7 months | arcw3 `runs/oooi_final/` | queued |
| EUROCONTROL All_Pre-Departure_Delay workbook | — | the online file is unchanged (57,854,598 bytes, Last-Modified 18 September 2026) and ends 31 July 2026; re-check on the dataset's day | no newer release |

Notes
- **adsb.lol has no archive for 6 May 2026** (no `v2026.05.06-planes-readsb-prod-*` release, 8 October); if May is a final month, that
  day's departures stay unmatched (the runner records it as a warning).
- The worker copies of `scratch/v24`, `runs/adsb_v24/features` and `data/external/adsblol/matches_v11` (the adj2 refit's inputs) and of
  `residual.py`, `common.py`, `run_adsb_v24.py`, `run_adsb_v25.py` match this workstation's SHA-256 (44 common files; the workstation also
  keeps 21 match manifests the refit does not read).
- `pull-weather` writes into `data/external/iem_final/<root name>/`: give the final run root a distinctive last segment (the December
  stand-in's root `runs/final_standin/dec2025/run` wrote `data/external/iem_final/run/`).
- **December 2025 stand-in** (`runs/final_standin/dec2025/`): `movements.parquet` (December 2025 in the ranking file's exact schema,
  departure labels and block times removed, SHA-256 `f06190f9...`), `template.parquet` (165,677 departures, the official template's dtypes),
  `labels.parquet` kept apart. `inspect` and `external` pass (all surface days present; ten station-months pulled). December is inside the
  twelve-month refits, so the stand-in tests plumbing and timing only, not accuracy.
