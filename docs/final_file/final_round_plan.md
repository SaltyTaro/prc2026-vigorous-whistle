# Final blind round: plan (8 October 2026)

The organisers will publish a final dataset with two additional months for one blind submission, due **14 October 2026 12:00 CEST
(10:00 UTC)**; the repo / docs deadline is unchanged and the model design should not change.
The final file must come from the documented, frozen pipeline run on the new months.

## The user's decisions (8 October 2026)

1. **Full v76 chain.** Parameterise every step after the runner for a new movement file; prove it by rehearsing on `data/raw/ranking.parquet`
   until it reproduces v76 (`a2ff395d...`) byte for byte, then run the same code on the new months.
2. **v40's nine LIRF fill pins are OFF in the final file** (their interval bands were chosen after v39's official score). They stay on only
   in the rehearsal, which must reproduce v76. The guard rows do not exist in a new file.
3. **Published-delay layers are kept where the data exists.** Pull the newest EUROCONTROL workbook; if it covers the final months the
   reconciliation, extreme-day and variance re-allocation layers run as on the board, otherwise they change nothing (as designed).

## Readiness at the start (read-only survey, 8 October)

- Runner (`tools/final_predict.py`, `docs/final_file/runner.md`): new raw file -> v55 model part; networks predicted on CPU; needs an adj2
  refit for the new months; refuses months overlapping January-November 2025.
- Step 1 (own-points member, v66): manual; no CPU prediction path documented; `build_ownpoints_features.py` hard-codes the ranking block.
- Steps 2-10: take `--raw / --detail / --published-table / --template` (`tools/final_chain_overrides.py`); gaps: the relearned
  flight-number -> callsign key map is not passed to `run_stack_resid_v3`, `run_opdi_stack`, `run_own_motion_v2`; v69's step 6
  (`run_stack_weight`) reads the shipped step-1..5 directory and `ranking.parquet`; OPDI features read `ranking.parquet`.
- Steps 11-18 (data-link pin, neighbour state / dark, second / third passes, two W25 passes, key memory): hard-wired to the shipped vN files
  (hash checks) and to the 2026-01 / 2026-07 matches. The W25 layers only act on January 2026 (nothing on other months, by design).
- External data for new months: adsb.lol surface days (pre-extraction of February-June and August-September 2026 started on arcw3-7,
  `data/external/adsblol/surface_final/` there), IEM weather pull, EUROCONTROL workbook (the 16 September release ends 31 July 2026), OPDI
  (fetch windows end August 2026), the adsb.lol data-link archive (OOOI pin, `--months`).

## Lanes (8 October)

- **Lane E (early chain):** step 1 on a new file (CPU prediction of the 16 own-points nets, own-point features for any movement file);
  steps 2-10 gaps (key map pass-through, step 6 in its v69 form from a work directory, OPDI features for a new file).
- **Lane L (late chain):** steps 11-18 runnable on (movement file, base file, work directory) without the shipped-file hash checks.

## Progress

- **Lane L done (315236f, `docs/final_file/lane_L.md`):** `tools/final_late_chain.py` runs steps 11-18 on (movement file, step-10 base,
  work directory). Rehearsed on `ranking.parquet` from the shipped v69: v70, the nbr_state dry run, v71-v74, the v75 dry run and v76
  (`a2ff395d...`) reproduce byte for byte, with all five intermediate ranking tables and the four refitted ranking bags (arcw1).
  A July-only cut runs end to end; its only difference from the full-file run is the declared transfer guard evaluated on the file
  being predicted (EHAM nbr_state ratio 0.89 on July alone vs 2.07 on the full file; LEMD 0.51, near the 0.5 edge). The W25 layers
  change nothing outside January 2026 (February / March 2026 would need a new declaration). Interface for a new file: `--base` (step-10
  output, fill pins off), `--recipe`, `--stacker-rank` (Lane E: `run_stack_weight.rank_inputs()` on the new file), `--matches-dir`,
  `--features-own-dir`, `--messages` (data-link messages; without them the LFPG pin passes through).
- **Main:** runner `--no-fill-pins` (13d8b52). Pre-staged inputs: `docs/final_file/prestage.md`.
