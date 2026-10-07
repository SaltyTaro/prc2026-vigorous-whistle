# The v69 chain (reproduction record, 7 October 2026)

v69 = **215.8835 official** (uploaded 04:30 UTC on 7 October at the user's word), `runs/submissions/vigorous-whistle_v69.parquet`,
SHA-256 `192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665`. This page extends [`v66_chain.md`](v66_chain.md) (kept
unchanged as the v66 record): steps 1-5 and 7-10 are v66's, unchanged; **step 6, the guarded stacker, is replaced twice**: by v68's
OPDI-at-LSZH stacker and then by v69's weight 0.65 / ten-seed form of it. Nothing here uploads or writes under `runs/submissions`.

## What changed after v66, in chain order

| version | official | change | step | record |
|---|---|---|---|---|
| v67 | 216.1021 | mix_own member (user override; worse than v66) | 1 | `docs/ctx_own2/` — **not in this chain** |
| v68 | 215.9449 | the OPDI learner (stack_resid_v3's learner with the OPDI columns, guarded) on LSZH rows with an OPDI match (`configs/opdi_stack_lszh_v1.json`, post-hoc LSZH restriction of `configs/opdi_stack_v1.json`) | 6 | `docs/opdi_stack_lszh/` |
| v69 | 215.8835 | the same three stacking learners, ten seeds, at weight 0.65 instead of 0.5 (`configs/stack_weight_v1.json`, candidate `w65_bag10`) | 6 | `docs/stack_weight/` |

## Commands (ranking file)

    tools/export_ownpoints_weight_override.py --dry-run --version 66   # steps 1-10 of v66 (v66_chain.md); steps 1-5 feed step 6 below
    tools/run_opdi_stack_lszh.py rank                                  # v68: step 6 = step_stack_opdi(..., ['LSZH']), then steps 7-10
    tools/run_stack_weight.py --rank-identity --rank-fit --rank-check  # v69 step 6 inputs: v68 identity, one model per seed x ten seeds
    tools/run_stack_weight.py --rank-dry                               # v69: step 6 at weight 0.65 with the ten-seed mean, then steps 7-10
    tools/rehearse_v69.py [--step5 rehearse|shipped]                   # all of the above into runs/final_rehearsal_v69/, byte checks

## The steps, in order (ranking file)

`S` = the shipped step-1..5 directory `runs/ctx_ownpoints_weight/rank/dry_v66/steps` (equal to v66's build and to the v66 rehearsal).

| # | step | function / tool (declaration) | reads | v68 output SHA-256 | v69 output SHA-256 |
|---|---|---|---|---|---|
| 1-5 | model part M' -> no-NM rematch (B) -> v48 reconciliation -> extreme-day -> variance re-allocation (C) | v66_chain.md steps 1-5, unchanged | see v66_chain.md | `S/rematch/base/...` `e0ffdbc4...`; `S/Y/vigorous-whistle_v66.parquet` `12a50ea0...` | same |
| 6 | **guarded stacker with the OPDI learner at LSZH** | v68: `run_opdi_stack_decl.step_stack_opdi(B, C, S/Y/rows.parquet, S/model_part/member_delta.parquet, out, 68, ['LSZH'])` with `run_stack_resid.SEEDS = [20261003, 11, 12]`, `WEIGHT = 0.5`. v69: `run_stack_weight.rank_inputs` (the same tables) -> each learner fitted once per seed for the ten seeds 20261003, 11, 12, ..., 19 (`run_stack_resid.fit_bag` with one seed; a seed's model does not depend on the others) -> `run_stack_weight.rank_value(P, 0.65, 10)` (value = max(chain + 0.65 x ten-seed mean, -12)) | B, C, `S/Y/rows.parquet`, `S/model_part/member_delta.parquet`, `runs/ctx_ownpoints_weight/chain/own_all_w100/{ctl,rows_all,dmp}.parquet`, own-point caches `runs/stack_resid/v3_features/{dev,rank}.parquet`, OPDI columns `runs/opdi_stack/features_{dev,rank}.parquet` | `312fab7aacc6aae0df61acfd2fbaaec8ccb515bfe75817d70c710f344c1d6b35` | `467ccb7459ca561d26fcd2b661af44be3ecd82496d890fc9d4ca3e1ddc188c94` |
| 7-8 | intruder stand bound + identity extension | `run_stack_resid_v3.chain_from_stack` -> `run_intruder_identity.dry_run` (as v66) | step 6, `runs/intruder_identity/ids.parquet` | `77ee5d1d2f102472958b09e328788f3d093942c06c9fd166ca2270ef779552a5` | `e0b2419619b38d6d4dc45c234aee319a86ca65dd5847990307c21ab5418877f8` |
| 9 | own-motion bound v1 (EDDF / EGLL / LFPG) | `run_own_motion.dry_run` inside `chain_from_stack` (as v66) | step 8, identity table, adsb.lol matches and surface days | `a57c88bf78d681710cb1c76ed2f183899af277404e127c87f02f1327ce08bb5a` | `f6ab8d8b960930270e0e314978333d4a9f9ef63a683b19ce5a2d8f72fb1a26f6` |
| 10 | own-motion bound v2 (EHAM / EDDM / LSZH / LEMD / LEBL), last | `run_own_motion_v2.dry_run` (as v66) | step 9, `runs/dig/position/{ids,park}_ranking.parquet`, adsb.lol surface days | **v68** `d41d8705e75a71e865fe21a8d6ad2f62044a5ed07f125fd208067914b2b10eab` | **v69** `192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665` |

Step 6 in detail (both versions; `tools/run_stack_resid.py`, `tools/run_stack_resid_v3.py`, `tools/run_opdi_stack.py`):

- **Learners.** v2: the 46 label-free inputs (`configs/stack_resid_v2.json`); v3: those 46 + 19 own-point columns (`configs/stack_resid_v3.json`);
  OPDI: the 65 + the OPDI columns (`run_opdi_stack.CORE` + the four `q_*_vs_pred` differences). Target label − chain, clipped at ±3,600 s.
  LightGBM `run_stack_resid.PARAMS`: 31 leaves, min 400, learning rate 0.03, feature fraction 0.5, bagging 0.7 / freq 1, L2 50, max_bin 127,
  4 threads, `deterministic=True`, `force_col_wise=True`; at most 800 rounds, early stopping 60 on fitting rows whose take-off day of month is a
  multiple of 5. Fitted on all four development months (January, July, November, December 2025) of the rebuilt own_all_w100 tables.
- **Routing.** LIRF rows take v2; LSZH rows with an OPDI match not held by the OPDI guard take the OPDI learner; every other row v3.
- **Guards.** v2's support guard (seven features, per-airport 0.1-99.9% range of the fitting rows); v3 / OPDI add the same guard on their
  new columns (`run_stack_resid_v3.guard_new`).
- **Value.** max(chain + w x mean of the seeds' predictions, −12): v68 w = 0.5 with seeds 20261003, 11, 12; v69 w = 0.65 with seeds
  20261003, 11, 12, 13, 14, 15, 16, 17, 18, 19. The stored per-seed ranking predictions of v69 are `runs/stack_weight/rank/preds.parquet`
  (`8083ffcfdf861e278ac8693fb2320872cd95378f3f9ac861decf23c9a6163772`); recombined at w 0.5 with three seeds they equal `step_stack_opdi`'s
  values exactly (`docs/stack_weight/results_v1.json`, `rank_recombination_vs_step_stack_opdi_max_abs` 0.0).

## Inputs added after v66 (SHA-256)

| input | SHA-256 | used by |
|---|---|---|
| `runs/opdi_stack/features_dev.parquet` (OPDI columns, 2025 development rows) | `0f0278d8e39e06a995df6daf6eb1ec95248820f6037640dd86f5e18ca6dd7380` | 6 |
| `runs/opdi_stack/features_rank.parquet` (OPDI columns, ranking rows) | `eb9896f89df5254c5e6cff5909aaffb8cab6508f4729da6e1be6e6118de2b333` | 6 |
| `runs/stack_resid/v3_features/dev.parquet` / `rank.parquet` (own-point columns) | `11f739731ae9ba48a8eddc8bcd91a06612d19689fd2263b946ece9c7e17a48ed` / `b16be93d992ae7ec881643543aed1b30d53993038aba01acaf39c45e188b8bbc` | 6 |
| `runs/ctx_ownpoints_weight/chain/own_all_w100/ctl.parquet` / `rows_all.parquet` / `dmp.parquet` | `462fc5f0...` / `9389b3d9...` / `5a5c3373...` | 6 |
| `runs/rematch_nonm/key_map.parquet` (flight number -> callsign; also the OPDI match key) | `79276c48511e587ccd476c5c9ec85a7ced55503924d5701ba126b98cbacac02c` | 2, 6 |
| `data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet` | `24434d40306efa952471d38f17f6892456902245244686bb8c1cf74c3d0bbad0` | 3-6 |
| `data/raw/ranking.parquet` / `data/raw/submitting.parquet` | `9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86` / `0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b` | all |
| configs: `opdi_stack_v1.json`, `opdi_stack_lszh_v1.json`, `stack_weight_v1.json`; results `docs/opdi_stack_lszh/results_v1.json`, `docs/stack_weight/results_v1.json` | git blobs at the rehearsal commit | 6 |

## Pinned settings and determinism

- The stacking learners (step 6) are pinned in `run_stack_resid.PARAMS` (`deterministic=True`, `force_col_wise=True`).
- The variance re-allocation of step 5 runs **as shipped** in the ranking chain (its LightGBM row/column-wise choice is not pinned there);
  the v66 rehearsal reproduced step 5 byte for byte. The **development** readings of every lever after 6 October (including v68's and
  v69's) used the pinned form (`deterministic=True`, `force_col_wise=True`, `runs/pubrefit/chain/ship`, `tools/run_pubrefit.py`), because
  unpinned refits drift up to 0.024% of a development month. The pin changes development readings only; no shipped ranking value.

## Known caveats (disclosed)

- **Own-point cache keyed on existence.** `run_stack_resid_v3.own_points` loads `runs/stack_resid/v3_features/<which>.parquet` whenever it exists;
  the `raw` argument and the row set are not checked. For a new movement file, or any
  change of the raw input or row set, **delete `runs/stack_resid/v3_features/rank.parquet` (and its receipt) first**; v66_chain.md's
  sentence that a changed row set rebuilds it is wrong. The OPDI columns (`runs/opdi_stack/features_rank.parquet`) are likewise built once by
  `tools/run_opdi_stack.py --features rank` and must be rebuilt for a new file.
- **Development stacking table built from the pre-rematch control.** `run_ctx_ownpoints_chain.dev_table_from` reads `ctl.parquet` (the chain
  without the no-NM rematch); the ranking table is built from the rematched model part. 430 development rows differ. Kept as declared since
  `configs/stack_resid_v2.json`; on the ranking side the stacker does not undo the rematch (coordinator check). Not a scoring change.
- **v40's nine LIRF fill pins** (`configs/fill_pin_v2.json`) are inherited through the model part (the runner layers inside the shipped v55
  model part `runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet`).
- The LIRF population and the two guard rows keep the model part's value through every step (checked at each dry run).

## Rehearsal (7 October 2026, `tools/rehearse_v69.py` on Azure arcw4) — v66, v68 and v69 reproduce byte for byte

Summary [`v69_rehearsal.json`](v69_rehearsal.json); full reports `runs/final_rehearsal_v69/<run>/rehearsal.json` (mirrored, hash-checked).

| run | what is rerun | v66 | v68 | v69 | intermediates vs the shipped dry runs | wall |
|---|---|---|---|---|---|---|
| `full_chain` (06:59-07:21 UTC) | all ten steps from the step inputs (v66's chain), then v68's and v69's step 6 and steps 7-10 | `b4ea727e...` identical | `d41d8705...` identical | `192c4a30...` identical | all equal (9 v66 step files incl. the model part, 4 + 4 for v68 / v69) | 1,324 s |
| `from_shipped_step5` (06:40-06:56 UTC) | v68's and v69's step 6 and steps 7-10 on the shipped step-1..5 files | — | identical | identical | all equal | 963 s |

- v69's per-seed ranking predictions (30 columns: three learners x ten seeds) equal the shipped `runs/stack_weight/rank/preds.parquet`
  exactly (same file SHA-256 `8083ffcf...`), so v69's step 6 is reproducible from the declared seeds alone.
- Every recorded input has the same SHA-256 before and after (`inputs_unchanged`); text inputs on the VM are the git blobs (LF), which is
  why their hashes differ from this Windows checkout's CRLF copies.
- The unpinned variance re-allocation of step 5 gave the same bytes on Linux as the shipped Windows run.
- Environment: Linux (Azure Standard_D16ds_v7, glibc 2.39), Python 3.12.3, LightGBM 4.7.0, numpy 2.4.2, pandas 3.0.0, pyarrow 23.0.1,
  scikit-learn 1.7.2 (the shipped files were built on Windows, Python 3.14.2, the same LightGBM / numpy / pandas / pyarrow). Driver peak
  RSS 3.6 GiB (step tools run as child processes and are not included). Tool SHA-256 `8eb24098...`.
- Two earlier attempts of the full run stopped on inputs missing from the VM (`runs/adsb_rematch/out_arcw5.parquet`,
  `runs/rematch_nonm/matches_2025.parquet`); they were copied from this workstation hash-checked and the run restarted from scratch.
  The partial attempts are kept under `runs/final_rehearsal_v69_attempts/`. No shipped file was changed.
