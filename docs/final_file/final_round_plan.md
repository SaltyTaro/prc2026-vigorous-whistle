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
- **Lane E done (76da2ed, `docs/final_file/lane_E.md`, `lane_E_rehearsal.json`):** `tools/final_own_member.py` (own-points member on CPU,
  32 nets ~17 min on a D16; per-month `features_v19own` for Lane L) and `tools/final_early_chain.py` (steps 0-10 on a new file; v69's step 6
  writes `stacker_rank.parquet` for Lane L). Shipped inputs + every new-file redirection: v69 `192c4a30...` and all nine intermediates
  byte for byte. Full new-file path with the CPU member and a real step-6 refit: 17,351 rows differ from v69 (95% within 2 s; squared
  change 4.4e4 s^2, about 0.0003 s of RMSE), 1,015 s. Gaps: steps 3-10 read ~8 GB of 2025 development tables that exist only on this
  workstation; `--no-published` (steps 3-6 skipped) drops the reconciliation and the stacker (333,120 rows differ from v69) and leaves
  Lane L without stacker inputs; OPDI for the final months may not be published (the OPDI learner then moves nothing).
- **Next lanes (8 October):** **Lane P** measures, on the 2025 development months, the chain without published figures (step 6 with NaN
  published-day features / a stacker without those columns / skipping it, and what the late passes do in each case) — the likely case if
  the final months are after July 2026 (the workbook ends 31 July; the next release is expected after 14 October). **Lane V** syncs the
  development inputs to arcw0 and runs the whole pipeline (runner -> early chain -> late chain, `--no-fill-pins`) on the December 2025
  stand-in there, timing every stage.
- **Lane P done (83634a0, `docs/final_file/lane_P.md`, `lane_P_results.json`, `lane_P_final_p2_manifest.json`):** without published figures
  (2025 development months, Jan/Jul/Dec mean, complete-month RMSE): no stacker (P3) is +0.77% against the full published chain; the shipped
  stacker with the published-day features NaN (P1) is **harmful** (+0.60% against P3; LightGBM 4.7 predicts NaN as 0.0 on columns never
  missing in training); the same learners / seeds / weight refitted without the nine published columns (**P2**) is **-0.29% against P3, every
  month and interval below zero** (about 40% of the gap); on P2 the second pass reads -0.105% and the third -0.030% (intervals below zero).
  Implemented: 30 frozen P2 models (`runs/final_nopub/final_p2/`, fitted on arcw1, hash-checked) and `final_early_chain.py --published-missing
  {off,auto}` (auto: steps 3-5 pass through at zero coverage - they used to crash; P2 on rows without a published day, v69's stacker on
  the rest); default path still reproduces v69 byte for byte. **For the final file: `--published-missing auto`** whatever the coverage; never P1.
- arcw2 (like arcw6) ran about ten times slower than its twin on identical work: avoid arcw2 / arcw6 for timed final-day runs.

- **Deadline:** code, docs and the final file are all due **14 October 2026 10:00 UTC**; the organisers review code up to the last commit
  before it. The board on `ranking.parquet` stays open until then (five uploads a day; the last upload is still v76 of 8 October).
- **The model may change until the deadline** ("you can change your model till the deadline"). This lifts the freeze this plan was written
  under (and the final-week schedule's "10-11 October no new lever"). The three decisions above were made under the freeze and still stand
  unless the user changes them. A new layer can still enter the final file if it is declared, accepted and run through the final chain
  before the deadline. Three of the four final rankings include February / June 2026, so a new layer should also be read on February /
  June 2025.
- **Final file name and upload:** `vigorous-whistle_final.parquet`, one upload ("unique final submission"), to the team bucket, unscored.
  `tools/prc_submit.py upload-final --source <late-chain export> --expected-sha256 <sha> [--check-only]` stages a byte copy under
  `runs/submissions/`, validates it against `final_submitting.parquet` (670,790 rows, exact order and types), checks that the organiser's
  final objects still carry the ETags of our download, refuses an existing final object, uploads with `IfNoneMatch` and reads the object
  back. `--check-only` stages nothing and uploads nothing. A check-only rehearsal on a dummy 670,790-row file passed on 10 October (organiser
  ETags unchanged, no final object in the bucket); its log was removed. Tests: `tests/test_submission.py` (17 passing).
- **Data:** the 8 October 11:04 UTC `final_ranking.parquet` / `final_submitting.parquet` are definitive; the jitter is intended.
  Their live ETags, sizes and timestamps equal our download on 10 October, so nothing is re-downloaded.
