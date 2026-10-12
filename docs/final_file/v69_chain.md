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

## Step 11 (v70): data-link OUT pin at LFPG

After step 10, `tools/run_oooi_pin.py` (declaration `configs/oooi_pin_v1.json`, w50): LFPG departures matched to an Air France 2N or
easyJet 8D data-link report from the adsb.lol Aircraft Data Links archive (CC0; `tools/oooi_pin_extract.py` / `oooi_pin_pack.py` on a
VM, `tools/oooi_pin_join.py` locally) and passing the source guard take value = max(0.5 x (take-off - OUT + offset) + 0.5 x v69, -12),
offsets from July / November / December 2025 (2N -62.26 s, 8D -22.84 s), np.rint. 775 rows; every other row equals v69. v70
SHA-256 `f1d17747...`, 215.8099 official.

**Rehearsal of v70 (7 October 2026, [`tools/rehearse_v70.py`](../../tools/rehearse_v70.py) on Azure arcw4, Linux): v66, v68, v69 and
v70 reproduce byte for byte.** Summary [`v70_rehearsal.json`](v70_rehearsal.json); full reports `runs/final_rehearsal_v70/<run>/rehearsal.json`
(102 files mirrored, hash-checked). Run `full_chain` (12:47-13:07 UTC, 1,234 s): `rehearse_v69.py`'s stages unchanged (all ten steps of
v66, then v68's and v69's step 6 and steps 7-10; every intermediate equal to the shipped dry runs, v69's 30 per-seed predictions exact), then
step 11 from the v69 it had just rebuilt: `run_oooi_pin.main()` imported unchanged, only its paths redirected into the rehearsal directory,
reading the stored filtered message tables `runs/oooi_pin/messages_unique.parquet` (`13a089c2...`) and `runs/oooi_pin/fresh/messages_unique.parquet`
(`bb297728...`, the governing run's recorded inputs). It re-parses and re-joins them, refits the offsets (2N −62.26 s, 8D −22.84 s), re-reads
clauses 1-5, selects w50 again and writes **v70 `f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f`, identical to the
uploaded file** (775 LFPG rows differ from v69, nothing else). The parsed and joined tables equal the governing run's byte for byte (the two
`join_stats.json` differ only by CRLF), `docs/oooi_pin/results_v1.json` equals on all 345 compared values (time stamp and code hashes
excepted), the dry-run receipt on all 26, and the published `data/external/adsblol_links/target_origin_messages.csv.gz` is rewritten
identically (decompressed; CRLF as on Windows). Run `from_v69_rehearsal` (15 s): step 11 alone from the v69 rehearsal's verified output, after
re-checking all 60 input and code hashes that rehearsal recorded; same bytes. A first step-11-only attempt also gave v70's bytes but
compared the CSV with Linux line endings; it is kept under `runs/final_rehearsal_v70_attempts/`. **The extraction is not rerun.** It was
`tools/oooi_pin_extract.py list --out runs/oooi_pin` (months 2025-07, 2025-11, 2025-12, 2026-01, 2026-07), `... run --kinds acars,hfdl` and
`... run --kinds vdl2`, then `tools/oooi_pin_pack.py runs/oooi_pin` on arcw0; the same with `--out runs/oooi_pin/fresh --months
2025-08,2025-09,2025-10` for the fresh months. The receipts [`receipts/archive_files.csv.gz`](../oooi_pin/receipts/archive_files.csv.gz) and
[`receipts/fresh_archive_files.csv.gz`](../oooi_pin/receipts/fresh_archive_files.csv.gz) (with the listing-page hashes in `listing_receipt.json`
and `fresh_listing_receipt.json`) bind 25,499 + 17,995 archive files by URL, size and SHA-256, each equal to GitHub's digest. The rehearsal
checks this, and checks that every one of the 257,969 + 132,178 stored messages comes from a receipted file. Same environment as the v69
rehearsal (Python 3.12.3, LightGBM 4.7.0, numpy 2.4.2, pandas 3.0.0, pyarrow 23.0.1); all inputs unchanged before and after. arcw4 was
deallocated after the copy back.

## Steps 12-13 (v71): neighbour ADS-B ground state on unmatched and dark rows

After step 11, two label-free layers, both applied last ([`docs/nbr_state/README.md`](../nbr_state/README.md)):

- **Step 12, `tools/run_nbr_state.py`** (declaration `configs/nbr_state_v1.json`, w75): for each departure, x = mean over ADS-B-matched
  departures of the same airport within 900 s (at least 3, the row itself excluded) of (value - v9), minus the airport-day mean of that
  difference; unmatched, non-fixed rows take max(value + clip(0.75 x b_airport x x, +-120 s), -12), b_airport a least-squares slope through
  the origin fitted on the four development months (v70's development proxy). EHAM keeps v70 (transfer guard). Dry run 7a0b1431...
- **Step 13, `tools/run_nbr_dark.py`** (declaration `configs/nbr_dark_v1.json`, dark_pool_w75): matched rows whose pushback ADS-B did not
  trace (gate False in `matches2_<month>`) take max(value + clip(0.75 x B x x, +-120 s), -12) with ONE pooled slope B = 0.1172 from the
  development months. No airport excluded.

Inputs: the uploaded v70 file, `runs/final_dryrun/run/recipe/detail.parquet` (ranking v9 and routing scope), `data/raw/ranking.parquet`
(take-off times), `data/external/adsblol/matches/matches2_2026-*.parquet` (gate flags), and for the slopes the v70 development proxy
`runs/oooi_lhr/dev_v70_proxy.parquet` (51aaa6a6...), `runs/lomo_dev/rows.parquet` (v9) and the 2025 raw month files. Commands:
`.venv\Scripts\python.exe -X utf8 -B tools\run_nbr_state.py` then `... tools\run_nbr_dark.py`; the second writes
`runs/nbr_state/rank/dry_v72/vigorous-whistle_v72.parquet`, which `docs/nbr_state/build_v71.py` (CHOICE "state+dark") copied to
`runs/submissions/vigorous-whistle_v71.parquet`. **v71 SHA-256 `2dfb3b9f53284c7ae3a02882d19cfd61b19116b1b00504176551bcae8b82e6ba`,
215.6232 official.**

**Reproduction on Linux (7 October 2026, Azure arcw1, during stack_nbr_v1's identity stage; [`docs/stack_nbr/HANDOFF.md`](../stack_nbr/HANDOFF.md)):**
starting from the v70 that the stacker path had just rebuilt byte for byte, the two layers' code reproduced nbr_state's dry run 7a0b1431...
and **v71 `2dfb3b9f...` byte for byte**, with the same exclusions (step 12 EHAM, step 13 none). Both layers are pandas / numpy arithmetic
with no random state.
## Step 14 (v72): second guarded residual pass at the seven high-coverage airports

After step 13, **`tools/run_stack_second.py`** (declaration `configs/stack_second_v1.json`, candidate sp_hicov7;
[`docs/stack_second/README.md`](../stack_second/README.md)): `--build` writes the development table (stacker inputs X79 prefixed `s_`,
neighbour columns nx / nxk / nn / x_nmerr, queue columns, clock columns, kind, ap; target clip(label - v71, +-1800)) and the ranking table
from the uploaded v71; `--fit DIR` (Azure) fits five-seed LightGBM bags (seeds 7, 11-14, 400 rounds, deterministic) on the non-fixed rows of
EDDF / EDDM / EGLL / EHAM / LEBL / LEMD / LSZH; `--score` writes max(v71 + clip(0.5 x bag, +-120 s), -12) on those rows (every other row
byte-identical) as `runs/stack_second/rank/dry_v72/vigorous-whistle_v72.parquet`, which `docs/stack_second/build_v72.py` copied to
`runs/submissions/vigorous-whistle_v72.parquet`. **v72 SHA-256 `4280c399aa148847c4be4588dfc3d98d4adfd1a5e64981035b4e0fa3ee802f80`,
214.927 official.** Predictions copied back from arcw0 with matching SHA-256 (`runs/stack_second/vm_preds.sha256`).
## Step 15 (v73): third guarded pass with raw ADS-B inputs at EDDF / EHAM / LSZH

After step 14, `docs/audit_v72/raw_build.py` writes the raw ADS-B input tables (`runs/audit_v72/raw_{dev,rank}.parquet`: features_v19own
26 columns + matches2 17 columns, times as seconds before take-off), then **`tools/run_stack_third.py`** (declaration
`configs/stack_third_v1.json`, candidate t3_ro_3; [`docs/stack_third/README.md`](../stack_third/README.md)): `--build` joins them with
runs/stack_second's tables, the second pass's predictions and the uploaded v72 (checking that v72 = v71 + sp_hicov7 exactly); `--fit DIR`
(Azure) fits five-seed LightGBM bags (seeds 7, 11-14, 400 rounds, deterministic) on the non-fixed rows of EDDF / EHAM / LSZH; `--score` writes
max(v72 + clip(0.5 x bag, +-120 s), -12) on those rows (every other row byte-identical) as `runs/stack_third/rank/dry_v73/vigorous-whistle_v73.parquet`,
which `docs/stack_third/build_v73.py` copied to `runs/submissions/vigorous-whistle_v73.parquet`. **v73 SHA-256
`649e99725ee3f700041b07a231dcd2fa656fdbeef3aa8db5b5b9c9fc341af3ba`, 214.5851 official.**
## Step 16 (v74): winter-season (W25) pass, January 2026 at EGLL / LIRF / LEBL

After step 15, **`tools/run_season_pass.py`** (declaration `configs/season_pass_v1.json`, candidate w25_e3;
[`docs/season_pass/README.md`](../season_pass/README.md)): `--build` takes runs/stack_third's tables and t3_ro_3's predictions (checking that
the uploaded v73 = v72 + t3_ro_3 exactly); `--fit DIR` (Azure) fits one five-seed LightGBM bag per airport (EGLL, LIRF, LEBL) on that airport's
non-fixed November + December 2025 rows (target clip(label - v73, +-1800)); `--score` writes max(v73 + clip(0.5 x bag, +-120 s), -12) on the
airport's non-fixed January 2026 rows (every other row byte-identical) as `runs/season_pass/rank/dry_v74/vigorous-whistle_v74.parquet`, which
`docs/season_pass/build_v74.py` copied to `runs/submissions/vigorous-whistle_v74.parquet`. **v74 SHA-256
`85e57d05322009d886d7423ba818ec6435f023bb8fd054463dd8adca6983f761`, 214.4666 official.**
## Steps 17-18 (v76): second W25 pass at EDDF / EDDM / EHAM / LSZH / LEMD, then the W25 flight-number memory

After step 16, **`tools/run_season_pass2.py`** (declaration `configs/season_pass_v2.json`, candidate w25b_5;
[`docs/season_pass2/README.md`](../season_pass2/README.md)): `--build` adds the stacker / neighbour / queue inputs of runs/stack_second to
runs/season_pass's tables (checking that the uploaded v74 = v73 + w25_e3 exactly); `--fit DIR` (Azure) fits one five-seed LightGBM bag per
airport on its non-fixed November + December 2025 rows (target clip(label - v74, +-1800), 151 features); `--score` writes max(v74 + clip(0.5 x
bag, +-120 s), -12) on the airport's non-fixed January 2026 rows as `runs/season_pass2/rank/dry_v75/vigorous-whistle_v75.parquet` (SHA-256
`a227c51ca90589bf6966f49ae94e2113394ee8f508bee0794d20871a800a6036`; never uploaded on its own). Then **`tools/run_key_memory.py`**
(declaration `configs/key_memory_v1.json`, candidate km_9; [`docs/key_memory/README.md`](../key_memory/README.md)) takes that file as its
base: per airport + FLIGHT_mvt, the mean of clip(label - base, +-1800) over November + December 2025 non-fixed rows (labels below 20,000 s,
base = v74's development values + w25b_5's out-of-month bags), c = clip(0.25 n/(n+30) mean, +-120 s), new = max(base + c, -12) on non-fixed
January 2026 rows at the nine airports other than EGLL, written as `runs/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet`, which
`docs/key_memory/build_v76.py` copied to `runs/submissions/vigorous-whistle_v76.parquet`. **v76 SHA-256
`a2ff395db4fdf22be86e1001f5dd97c8f27adc7b836fd7b4902690a576f9725c`, 214.3353 official.**

## Final-file step 19 (final candidate 3, 12 October 2026): LIRF fill-population dominance cap

After step 18 (and the six-month third pass of `docs/sixmonth/final_candidate2.md`), **`tools/run_lirf_cap.py apply`** (declaration
`configs/lirf_cap_v1.json`, [`docs/lirf_cap/README.md`](../lirf_cap/README.md)) caps LIRF fill-population rows at max(schedule interval,
88,392 s); every other row byte-identical. On the ranking file it changes no row (v76 unchanged); on the final file it moves two July 2026 rows.
