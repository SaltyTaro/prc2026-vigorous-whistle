# After v12 (246.4831, 3rd of 144; 2nd 245.2932, 1st 245.0207): plan updated 15 September 2026 ~10:25 UTC

Gap to 2nd: 1.19 s (0.48%); to 1st: 1.46 s (0.59%).

## Done today

- v11 (`configs/adsb_v11_v1.json`): v9 + ADS-B residual on matched rows → 248.4368 (4th).
- v12_v1 (`configs/adsb_v12_v1.json`): neighbour features on matched/context scopes → null.
- v12_v2 (`configs/adsb_v12_v2.json`): LIRF excluded, neighbour features, weight 1.0 →
  **246.4831 (3rd)**; the development mean (−0.78%) predicted the ranking gain (−0.79%).
- Extraction complete for every available day (no adsb.lol release 28 May–10 Jun 2025);
  pass-2 matches with 1,332 stands for all months on the workstation
  (`data/external/adsblol/matches/`); v11/v12 inputs frozen in `matches_v11/`.

## Step 2 (in progress) — full-data learner stacked into the residual

`tools/run_adsb_full.py`: LightGBM on the timeline bundle's 276 model features plus
the 27 ADS-B features, target = label, seasonal and forward folds (arcw1/arcw2),
then a refit on the eleven months at the median best iteration. Its out-of-sample
prediction (`adsb_full`) becomes an extra residual feature in a v12_v3
declaration (`extra_feature_tables`, `attach_extra` in `run_adsb_v11.py`;
scopes `matched_no_lirf` and `all_no_lirf`; baseline = v12 development
predictions, column `residual_matched_no_lirf|nb`, weight 1.0). Rationale: the
residual model sees only 2–3 months of matched rows; per-airport/stand offset
distributions (LEMD/EGLL/EDDM have months with zero coverage) need all eleven.

## Step 3 — pushback timing failure modes (measured, not yet modelled)

- Gate rows: 12–24% have the block time 200–400 s before the first received
  movement (transponder switched on late, or "ready but waiting"); the model
  hedges. Signature: `adsb_first_gs_kt` 2–3 kt, no slow start, no taxi stop.
- Dark rows: offset grows with `adsb_first_dist_stand_m` (EDDF 37 s at < 50 m →
  460 s at 1–3 km); stands unknown for 41% (LEMD) / 36% (LSZH) of dark rows —
  name-based stand imputation would fill most of them.
- Segments whose motion start is hours before takeoff (a separate earlier
  movement): EHAM/LSZH 1–3 km bins with offsets of −4,000 s; a "last long stop"
  boundary would fix them but risks legitimate long holds.

## Housekeeping

arcw3–arcw5 are idle; deallocate once `pull_surface` has copied their surface
files. arcw0 is
idle (disk 99% full: other projects' data; do not touch).
