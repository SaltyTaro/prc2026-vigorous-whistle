# The v66 chain (reproduction record, 6 October 2026)

v66 = **216.0926 official** (uploaded 08:15 UTC on 6 October at the user's word), `runs/submissions/vigorous-whistle_v66.parquet`,
SHA-256 `b4ea727e1644b6add104f44832898bfecfd3d5607f7d6a9ce0471efe581322e4`. This page lists every step that builds it, what each step
reads, the tool that runs it and the SHA-256 of every intermediate. It extends [`v61_chain.md`](v61_chain.md) (kept unchanged as the
v61 record): steps 2-5, 7 and 8 are v61's, step 1 replaces one member of the model part, step 6 is the refitted stacker and steps 9-10
are the two own-motion bounds. Nothing here uploads or writes under `runs/submissions`; a submission build stays the user's word.

## What changed after v61, in chain order

| version | official | change | step | record |
|---|---|---|---|---|
| v62 | 218.4174 | own-motion bound at EDDF / EGLL / LFPG (`configs/own_motion_v1.json`) | 9 | `docs/own_motion/` |
| v63 | 217.9413 | guarded stacker refitted with the own-aircraft turnaround points (`configs/stack_resid_v3.json`; LIRF keeps v2's learner) | 6 | `docs/stack_resid/`, `docs/dig/pregap.md` |
| v64 | 217.8569 | own-motion bound at EHAM / EDDM / LSZH / LEMD / LEBL (`configs/own_motion_v2.json`; user override) | 10 | `docs/own_motion/` |
| v65 | 216.8814 | the context-network member replaced by `own_all` (dl32own matched / m25dl32own unmatched, 16 seeds) at 0.40; stacker refitted on the rebuilt components | 1, 6 | `docs/ctx_ownpoints_dev/` |
| v66 | 216.0926 | `own_all` at weight 1.00 (user override of `configs/ctx_ownpoints_weight_v1.json`) | 1, 6 | `docs/ctx_ownpoints_weight/` |

## One command on the ranking file

The whole chain is `tools/export_ctx_ownpoints.chain(candidate, out, version, identity)`, called by
`tools/export_ownpoints_weight_override.py` with `CHAIN = runs/ctx_ownpoints_weight/chain`, `DECLARATION = configs/ctx_ownpoints_weight_v1.json`
and the candidate `own_all_w100`:

    tools/export_ownpoints_weight_override.py --identity --version N   # own_all_w40 through the same path must give v65 byte for byte
    tools/export_ownpoints_weight_override.py --dry-run --version N    # -> runs/ctx_ownpoints_weight/rank/dry_vN/
    tools/rehearse_v66.py                                              # the same chain into runs/final_file/rehearsal_v66/, compared step by step

`--build` (writes `runs/submissions/`) needs the user's decision text and refuses to overwrite.

## The steps, in order (ranking file)

`O` = the output directory (`runs/final_file/rehearsal_v66/steps` for the rehearsal). Hashes: shipped dry run = shipped build = rehearsal
(see the last section).

| # | step | function / tool (declaration) | reads | output | SHA-256 |
|---|---|---|---|---|---|
| 1 | **model part M'** | `export_ctx_ownpoints.step_model_part`: the shipped v55 model part with the non-LIRF rows of the v52m step recomputed (`export_network_member.stacks` on the v34 rebuild, `export_disagreement_gate.rebuild_v34`): the committee's network table moves to `(1 - w) table + w member` with `own_all` at w = 1.00, then the masked member at its shipped w40; the candidate's float where it differs from the combination's float, the shipped integer elsewhere; LIRF rows (v53m-v55m) unchanged (`configs/ctx_ownpoints_weight_v1.json`, candidate `own_all_w100`) | `runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet` (266b8c93...), the 16 twelve-month member tables `runs/ctx_ownpoints_dev/member/s601..s616/ranking.parquet` (hashes in `steps.json`), the combination / mask member tables of `configs/adsb_ctx3_combo_dev_v1.json` / `configs/adsb_ctx3_mask_dev_v1.json`, the runner's dry-run detail | `O/model_part/vigorous-whistle_vN.parquet` + `member_delta.parquet` | `73eef8daa3c48500db13497491bee2926498c06334f29743d403969e0ede470c` |
| 2 | no-NM ADS-B rematch | `tools/run_rematch_nonm.py --matches --out O/rematch`, then `--ranking-base --base M' --base-sha256 ...` (`configs/rematch_nonm_v1.json`) | M', 2025 matches/offsets of `runs/rematch_nonm/`, adsb.lol surface days | `O/rematch/base/vigorous-whistle_v55m_rematch.parquet` (= B) | `e0ffdbc4da3292d6d2f6013230317d32413fb54fde9ee4903a8c01ea84a480e8` |
| 3 | v48 reconciliation | `tools/export_daily_totals.py --base B --dry-run O/R --no-incumbent-report` (`configs/daily_totals_v1.json`) | B, published daily totals | `O/R/vigorous-whistle_vN.parquet` | `cf3b9f8eec0cac90cc044c0059c3542837a3f35882c98e8349a9a9835e0661df` |
| 4 | extreme-day layer | `tools/scout_extreme_shift.py --base B --recon O/R --dry-run O/X` (`configs/scout_extreme_shift_v1.json`) | B, step 3, published daily totals | `O/X/vigorous-whistle_vN.parquet` | `02a3e44c83a801838059afe9310fa96c41876be3d6457d5534519b87070e3f79` |
| 5 | variance re-allocation | `tools/scout_var_alloc.py --base B --recon O/R --previous O/X/... --dry-run O/Y` (`configs/scout_var_alloc_v1.json`) | B, steps 3-4, published daily totals | `O/Y/vigorous-whistle_vN.parquet` (= C) + `O/Y/rows.parquet` | `12a50ea0011164f6ecdf2df1f227f930a9130b7f669eb523bb311d9e8771bad5` |
| 6 | guarded stacker, refitted | `export_ctx_ownpoints.step_stack`: stack_resid_v3's learner (46 v2 inputs + 19 own-point columns, `run_stack_resid_v3.own_points` / `new_columns` / `guard_new`) and v2's learner refitted on the candidate's rebuilt development tables; ranking table `run_stack_resid.ranking_table` on C with the v51m-v54m components moved by the member's delta (non-LIRF rows); LIRF rows take the refitted v2 learner (`configs/stack_resid_v2.json`, `configs/stack_resid_v3.json`) | C, B, `O/Y/rows.parquet`, `O/model_part/member_delta.parquet`, `runs/ctx_ownpoints_weight/chain/own_all_w100/{ctl,rows_all,dmp}.parquet`, the own-point caches `runs/stack_resid/v3_features/{dev,rank}.parquet` | `O/stack/vigorous-whistle_vN.parquet` (= D) | `2ae5ba770d03e74628cedc72758dd8dee6151b36b38bb6d1302d19a3fd7bb2e3` |
| 7-8 | intruder stand bound + identity extension | `run_stack_resid_v3.chain_from_stack` -> `run_intruder_identity.dry_run` (no v60 check) (`configs/intruder_bound_v1.json`, `configs/intruder_identity_v1.json`) | D, identity table `runs/intruder_identity/ids.parquet` (3097dd06...), arrival blocks / stands of the ranking file | `O/identity/vigorous-whistle_vN.parquet` | `8b85e477ff0ee7ed5d23e0f5cac571465d40c8cb50c485657fd978949d532cc2` |
| 9 | own-motion bound v1 (EDDF / EGLL / LFPG) | `run_own_motion.dry_run` inside `chain_from_stack` (`configs/own_motion_v1.json`; offsets from the twelve 2025 months) | step 8, the identity table, `data/external/adsblol/matches/matches2_<month>.parquet`, adsb.lol surface days | `O/own/vigorous-whistle_vN.parquet` | `46f49ad3619670cd451af21fb9097dec694b96a81e51b04e150df1b7f3703131` |
| 10 | own-motion bound v2 (EHAM / EDDM / LSZH / LEMD / LEBL), last | `run_own_motion_v2.dry_run` (`configs/own_motion_v2.json`) | step 9, identity tables `runs/dig/position/ids_ranking.parquet` / `park_ranking.parquet`, 2025 triggers (cached), adsb.lol surface days | `O/om2/vigorous-whistle_vN.parquet` = **v66** | **`b4ea727e1644b6add104f44832898bfecfd3d5607f7d6a9ce0471efe581322e4`** |

The LIRF population rows and the two guard rows keep the model part's value through every step (checked by
`export_ctx_ownpoints.compare`).

## Inputs and external data

Fixed inputs (SHA-256 at the rehearsal; the full list with hashes is `runs/final_file/rehearsal_v66/rehearsal.json`, `inputs`):

| input | SHA-256 | used by |
|---|---|---|
| `data/raw/ranking.parquet`, `data/raw/submitting.parquet` | in `rehearsal.json` | every step |
| `runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet` (shipped v55 model part) | `266b8c936f2625578e4980f1b11c3a26cc63594a13916510fa3fe86925ec800f` | 1 |
| `runs/ctx_ownpoints_dev/member/s601..s616/ranking.parquet` (16 twelve-month `own_all` refits) | listed in `steps.json` -> `model_part.member_tables` | 1 |
| `runs/ctx_ownpoints_weight/chain/own_all_w100/{ctl,rows_all,dmp}.parquet` (rebuilt development tables) | in `rehearsal.json` | 6 |
| `runs/stack_resid/v3_features/dev.parquet` / `rank.parquet` (own-point columns, cache keyed on the row set) | `11f73973...` / `b16be93d...` | 6 |
| `runs/intruder_identity/ids.parquet` (identity table) | `3097dd067aaa165fe2510310e05bcccae269fec7ee75a088af5e7285846b7dba` | 8, 9 |
| `runs/dig/position/ids_ranking.parquet` / `park_ranking.parquet` | `b8cf4ce0...` / `28471b34...` | 10 |
| `data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet` (published daily totals) | in `rehearsal.json` | 3, 4, 5, 6 |
| `data/external/adsblol/matches_v11/stands.parquet` | `63502c9465e3742d537fca30bd3c03622623a3be047eb01ba068292a249d995d` | 2, 8 |

External data by step: **adsb.lol** (ODbL; surface day files `data/external/adsblol/surface/`, matches, the own-point features
`data/external/adsblol/features_v19own/` behind the member tables and the step-6 cache) in steps 1 (through the member), 2, 6, 8, 9, 10;
**EUROCONTROL published daily pre-departure delay** (the user's decision of 29 September) in steps 3-6; IEM weather and OSM geometry only
upstream, inside the model part's members. No step reads an official score.

## Determinism

The shipped dry run (`runs/ctx_ownpoints_weight/rank/dry_v66/`, 07:16 UTC) and the build (`build_v66/`, 08:12 UTC) were two separate runs of
the chain and agree on every intermediate. Fork B found (6 October) that LightGBM's automatic row/column-wise histogram choice is decided by a
timing test, so a refit under different load can differ (up to 0.024% of a development month for the variance re-allocation's models).
The chain fits LightGBM models in steps 5 (variance re-allocation, month-blocked on 2025) and 6 (the stacker bags) without pinning
`force_col_wise`. The rehearsal below shows whether that choice moved on this machine; the shipped values are never changed.

REHEARSAL_RESULT

## For a new movement file

The ranking-file chain above calls its tools with their default inputs; `export_ctx_ownpoints.chain` itself has no `--raw` / `--template`
flags. For a new file, run the steps one by one with the override flags of `v61_chain.md` (steps 2-5, 7-8 unchanged there) and these additions:

- **Step 1**: the 16 `own_all` twelve-month refits must predict the new file (`tools/run_adsb_ctx_lomo.py` with the new file's token build and
  `--extra-features` from `tools/build_ownpoints_features.py` on the new file), then `export_network_member.stacks` with weight 1.00 on the
  runner's v55m detail. Not wrapped in one tool; manual.
- **Step 6**: `run_stack_resid_v3.own_points("rank", ids, raw=F)` rebuilds the own-point columns when the cache's row set differs (the cache is
  keyed on the row set); the development tables stay those of `runs/ctx_ownpoints_weight/chain/own_all_w100/`.
- **Step 9**: `tools/run_own_motion.py --dry-run --base ... --raw F --ids ... --matches-dir ... --out ...` (flags exist; `v61_chain.md` step 9).
- **Step 10**: `tools/run_own_motion_v2.py --dry-run --base ... --raw F --ids-dir ... --build-ids <tag> --out ...` (flags exist).
- If EUROCONTROL has not published the period's daily figures, steps 3-6 are skipped as in `v61_chain.md`.

## Extending this page by one step

A later layer (v67 onwards) adds one row to the step table after the step it follows, its output SHA-256 from its own dry run, and a
rehearsal tool `tools/rehearse_v<N>.py` written like `tools/rehearse_v66.py` (rerun the shipped export path into
`runs/final_file/rehearsal_v<N>/`, compare `steps.json` with the shipped dry run and build, record inputs, code hashes and environment).
If the layer replaces a member (step 1) or refits the stacker (step 6), every later hash changes and the whole table is re-recorded.
