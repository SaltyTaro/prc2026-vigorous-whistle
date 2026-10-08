# Lane P: the chain without EUROCONTROL's published figures (8 October 2026)

Plan: [`final_round_plan.md`](final_round_plan.md). If the final blind months are after July 2026, the published daily figures are missing
for every row (the workbook ends 31 July 2026, the online file was last modified 18 September, and the next release is expected after
14 October). Steps 3-5 (v48 reconciliation, extreme-day, variance re-allocation) then change nothing by design. Step 6, v69's guarded
stacker (three learners x ten seeds at weight 0.65, plus the LSZH OPDI learner), reads nine published-day columns. The second pass reads all 79 of
the stacker's inputs. Lane E's `--no-published` skipped steps 3-6 and left the late passes without inputs.

Tool: [`tools/final_nopub.py`](../../tools/final_nopub.py). Readings: [`lane_P_results.json`](lane_P_results.json). Development months:
January, July, November and December 2025, complete-month RMSE on every row, with paired-day 95% intervals (`run_intruder_bound.interval`,
seed 20261004). No official score and no ranking label is used.

## What "missing" does to step 6

- The nine published-day columns are `p_pub`, `p_n_pub`, `p_disc_new`, `p_disc_m`, `p_disc_new_pr`, `p_disc_m_pr`, `p_pub_pr`,
  `p_n_ratio` and `p_complete`. `p_n`, `p_own_share` and `p_pos_new` are computed from the file and stay.
- None of the nine was ever missing in training (NaN share 0.0 on all 672,428 development rows). For a feature never missing in training,
  LightGBM 4.7 predicts NaN exactly as 0.0 (checked on a toy model). So the shipped stacker would read "zero published delay, zero
  published departures" on every row.
- Without published figures the chain entering step 6 is the model part plus the no-NM rematch (B). The stacker's own base is, as
  shipped, the rematch-free model part.

## Step 6 variants (v69 level, after steps 7-10 run exactly as `run_stack_weight.dev_score`)

- **P1:** the shipped learners (month-blocked refits on the published chain, as shipped) predicting from the no-published features.
- **P2:** the same learners, seeds and weight, refitted month-blocked on the no-published chain with the nine columns dropped, using
  P2's guards (the support guard's TOP features minus the dropped ones).
- **P3:** no step 6.

Identity: P1's refits predict the published features of each held-out month exactly as the shipped development predictions
(`runs/stack_weight/dev/preds_<M>`): max |diff| 0.0 over all 30 learner-seeds.

| reading (RMSE change) | January | July | November | December | Jan/Jul/Dec mean |
|---|---:|---:|---:|---:|---:|
| P3 vs the full chain with published figures | +0.255 [+0.10, +0.90] | +0.851 [+0.62, +1.16] | +0.830 [+0.57, +1.17] | +1.197 [+0.94, +1.46] | +0.768 |
| P1 vs P3 | +0.025 [-0.05, +0.16] | +0.126 [-0.05, +0.30] | **+1.089 [+0.74, +1.55]** | **+1.657 [+1.26, +2.08]** | +0.603 |
| **P2 vs P3** | **-0.114 [-0.40, -0.04]** | **-0.146 [-0.21, -0.09]** | **-0.376 [-0.54, -0.25]** | **-0.608 [-0.79, -0.44]** | **-0.289** |
| P2 vs the full chain | +0.140 | +0.703 | +0.451 | +0.582 | +0.475 |

The shipped stacker on missing figures (P1) is worse than no stacker, with November and December intervals above zero. **P2 improves
every month, with every interval below zero.** It recovers about 40% of what the published layers are worth (0.77% -> 0.48% of the
equal-footing gap).

## The late passes on the P2 chain

Steps 11-13 (the LFPG data-link pin, neighbour state and dark rows) read no published figure. For these readings their shipped
development deltas are carried onto the P2 chain, an approximation of rerunning them on the new base.

| reading (RMSE change) | January | July | November | December | mean |
|---|---:|---:|---:|---:|---:|
| **second pass as shipped (L1, published `s_*` NaN) vs no second pass (L3)** | -0.052 [-0.18, -0.02] | -0.056 [-0.08, -0.03] | -0.143 [-0.23, -0.08] | -0.207 [-0.28, -0.15] | **-0.105** |
| second pass refitted without the published `s_*` (L2) vs L3 | -0.052 [-0.19, -0.02] | -0.023 [-0.05, +0.01] | -0.120 [-0.21, -0.06] | -0.229 [-0.30, -0.17] | -0.101 |
| **third pass as shipped (T1, on P2 + L1) vs no third pass** | -0.018 [-0.07, -0.01] | -0.022 [-0.03, -0.01] | -0.064 [-0.10, -0.04] | -0.050 [-0.06, -0.04] | **-0.030** |

The second pass is robust to the missing columns: as shipped it is at least as good as a refit, with every interval below zero. The
third pass reads no stacker input. **Both run as shipped**, with no code change in Lane L's chain: the early chain still writes
`stacker_rank.parquet`, with the published columns NaN. The W25 season passes and the key memory act only on January 2026 rows.

## Recommendation

| case | final file |
|---|---|
| published figures cover the final months | default (`--published-missing off`, as shipped; the rehearsal path) |
| **no published airport-day in the period** (expected if the months are after July 2026) | **`--published-missing auto`**. Steps 3-5 are written as their pass-through: the step-5 file is B and `Y/rows.parquet` gives the off-block day. The step-6 value comes from P2's frozen learners on every non-fixed row; v69's own fits are skipped. The late chain is unchanged. |
| published for part of the period | `--published-missing auto`. Steps 3-5 run, and read complete airport-days only. Rows whose off-block day has a published figure keep v69's stacker value; the other rows take P2. The mixture is not read as such on development; each part is. |

Do not run v69's stacker on missing figures (P1), and do not use `--no-published` (P3) while the frozen P2 learners are available.

## Implementation

- **`tools/final_nopub.py`:** `build` / `fit` / `score` (the readings above); `pass2-*` / `pass3-*` (the late passes); and
  `fit-final`, which fits P2's ranking learners: 3 learners x 10 seeds on all four development months (~fixed rows, the shipped
  day-of-month validation split). They are saved under `runs/final_nopub/final_p2/` (models and `manifest.json`, with the SHA-256 of
  every model and of the input tables; fitted on Azure arcw1, mirrored hash-checked). Its no-published inputs (the model part, `a`,
  `bday`, `cls`, the components) are identical in the pinned `runs/pubrefit/chain/ship` copy and in the shipped
  `runs/ctx_ownpoints_weight/chain/own_all_w100` that `run_stack_weight.rank_inputs` fits on. Only the post-reconciliation `new` differs
  (at most 2.9 s), and P2 never reads it.
- **`tools/final_early_chain.py --published-missing {off,auto}`** (default `off`: the shipped path, unchanged):
  - `published_covers()` asks whether the published table holds any airport-day within the file's departure days;
  - `passthrough_3_to_5()` writes the step-5 outputs when it does not (`export_daily_totals.py`'s report fails on an empty day table:
    `KeyError: 'airport'`);
  - `published_missing()` adds P2's value on rows without a published day;
  - v69's own fits are skipped when no row has one.
  - The chain records the flag, the rows replaced and the P2 prediction file's SHA-256.
- The shipped tools (`export_daily_totals`, `scout_*`, `run_stack_*`) are unchanged.

## End to end on the ranking file without published figures

`tools/final_early_chain.py --step1-source shipped --members shipped --published-table runs/final_nopub/pub_until_2025.parquet
--published-missing auto` ran on this workstation (the chain's development tables live here). The published extract was cut at 31 December
2025 (SHA-256 `6cb12ff0...`), so nothing in January or July 2026 is published.

- Steps 3-5 were written as their pass-through.
- 344,437 of the 344,441 non-fixed rows took P2's value. The other 4 have an off-block day of 31 December 2025, which the extract covers,
  so they kept v69's stacker (the partial rule); because of them v69's own fits ran too (step 6 took 279 s).
- Steps 7-10 then ran. Total 685 s; record `runs/final_nopub/e2e_auto/chain.json`.
- File `d0b0a6ec...`: template order, int32, minimum -12. The frozen learners' manifest is
  [`lane_P_final_p2_manifest.json`](lane_P_final_p2_manifest.json) (`b70b3f52...`).
- Against v69 (the published chain), 328,208 rows differ (RMS 28.3 s): the size of what the published layers do.
- 66 LIRF population rows keep B where v69's steps 3-5 had moved them (the published layers do act on population rows they do not hold
  as heavy). The stacker moves no fixed row in either chain, and the guard rows are unchanged.

**The default path is unchanged:** `--published-missing off` with the shipped inputs and the shipped per-seed predictions
(`--stack-preds runs/stack_weight/rank/preds.parquet`) reproduces v69 byte for byte (`192c4a30...`; `runs/final_nopub/rehearsal_off/`).

## Runs

- Development fits on Azure:
  - P1 on arcw1 (4 months x 30 learner-seeds);
  - P2 on arcw2;
  - the second-pass fits (L1 on arcw1; L2 moved from arcw2 to arcw1);
  - the frozen P2 learners and the third pass, both on arcw1.
- arcw2 ran about ten times slower than arcw1 on identical work (the same CPU type), as arcw6 did earlier the same day. It was stopped and
  deallocated; arcw1 was deallocated after the hash-checked mirrors.
- Everything else ran on this workstation: matrices, guards, steps 7-10 and the end-to-end check, because the chain's development
  tables live only here.
