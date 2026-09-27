# Rebuilding v30 on a new movement file — stage 1 inventory (27 September 2026)

Insurance against a 2025-style separate final test file (the 2025 prize ranking used a hidden final file released late;
the 2026 site says nothing of the kind). This traces every column of `runs/adsb_ctx/export_v30/ranking_values.parquet`
back to the tool that first computed it for `data/raw/ranking.parquet`, and records what a different input file needs.
Read-only investigation; nothing was run beyond file listings. Machine-readable twin: [`inventory.json`](inventory.json).

**Bottom line.** Every *model* behind v30 is saved on disk (tree boosters, CatBoosts, 144 network `model.pt` files, the v7
nodes, the v9 classifier is refittable from 2025 labels in seconds), so a new file needs **features + inference, not
refits** — with one exception: the residual stage's `adj2` LightGBM is fitted per *target month* (2026-01, 2026-07 only),
so any other month needs a ~5-minute refit. The work is plumbing: ~20 tools hard-code `data/raw/ranking.parquet`,
`RANKING_MONTHS = ["2026-01", "2026-07"]` and "must reproduce the uploaded vN" guards. The December campaign
(`src/prc2026_december/`, `tools/build_december_features.py`, `scoring.py`) already solved the feature side for a new month
and is the template.

## 1. Column map (v30 detail → first producer)

v30 = max(0.3·(v9 + residual) + 0.7·mix, −12) on non-LIRF rows (mix = (1−0.75)·committee + 0.75·network table); LIRF
non-population rows use the round-B blends of (v9, tree = `ens_v22_lirf_q`, net = `ens_ctx_lirf`); LIRF fill population
rows (`fixed` scope) keep v9. Arithmetic: `tools/run_adsb_v11.candidate_values`, `run_adsb_nn2_dev.lirf_blends/spec_for`,
`run_adsb_nn_dev.mixed`, `run_adsb_rnn_dev.combine`.

| column | what | first computed for the ranking by | saved models to re-apply |
|---|---|---|---|
| `v9` | where(NM clock, max(v7₁₂, −12), v9_frozen); v9_frozen = v7 frozen + 0.75 LIRF mixture on the fill population (interval ≤ 43,200 s) | `tools/export_adsb_v25.py` (v23 export, 22 Sep) — itself v22's `export_adsb_v24.py`; recorded in `runs/adsb_v25/export_v23/ranking_values.parquet`, carried unchanged v24→v30 | v7 frozen: `runs/azure_arcw0_surface_refit/20260914T185207842456Z_refit_full_timeline_v6_100/pipeline.json` (+v6/v5 fallback inside); v7₁₂: `runs/adsb_v24/v7/refit/` (nodes: 10 airport LGBs + 2 globals); mixture classifier refit from Jan–Nov 2025 labels by `run_lirf_mixture.fit_population_model` (byte-identical, seconds) |
| `residual` | 0.05·adj2 + 0.05·cat + 0.9·rnet on `matched_no_lirf`, 0 elsewhere | adj2/cat: `export_adsb_v25.py`; rnet: `tools/refit_adsb_rnn.py` (16 seeds, Azure) on `runs/adsb_rnn/ranking_nb.parquet` (`tools/rnn_ranking_table.py`); combined in `export_adsb_rnn.py` (v26) | `runs/adsb_v25/export_v23/residual_adj2_2026-01.txt`, `residual_adj2_2026-07.txt`, `residual_cat_all.cbm`; `runs/adsb_rnn/rnb_s5??/refit12/model.pt` |
| `committee` | `ens_v20_bag_q_all` = mean of finite twelve-month leaf predictions over its flattened member list | `tools/run_adsb_learner_dec.py` per leaf (Azure arcw0–7, 19 Sep, 39.3 h fit total) → `export_december_refit.py` table means | 29 leaves `runs/adsb_v23/<dir>__<name>/refit/{model.cbm|model.txt, run.json}` |
| `ens_v22_lirf_q` | LIRF tree table (same leaf mechanism, LIRF-inclusive LGB group + 4 LIRF CatBoosts) | same as committee | same leaves |
| `ens_nn`, `ens_new` | 16+16 nnb MLP seeds (s101–132) | `tools/run_adsb_nn.py --fold refit12` (Azure) | `runs/adsb_nn/nnb_s1??/refit12/model.pt` |
| `ens_nnl` | 16 LIRF-inclusive MLP seeds (s201–216) | `tools/run_adsb_nn2.py` (Azure) | `runs/adsb_nn2/nnl_s2??/refit12/` |
| `ens_ctx` | 16 context nets (s101–116, 16 epochs, CPU) | `tools/run_adsb_ctx.py --fold refit12` (Azure) | `runs/adsb_ctx/ctx_s1??/refit12/` |
| `ens_ctx_e24`, `ens_ctx_e24b` | 16+16 e24 context nets (s117–148, CPU) | `tools/run_adsb_ctx.py` (Azure) | `runs/adsb_ctx/ctx_e24_s1??/refit12/` |
| `ens_ctx_e32` | 16 e32 context nets (s401–416) | `tools/run_adsb_ctx3.py` on AWS L40S (TF32) | `runs/adsb_ctx_gpu/ctx_e32_s4??/refit12/` (E: junction); CPU reload within 0.09 s RMS per net |
| `ens_ctx_lirf` | 16 LIRF context nets (s301–316, CPU) | `tools/run_adsb_ctx.py` (Azure) | `runs/adsb_ctx/ctx_lirf_s3??/refit12/` |
| `committee_mix`, `lirf_mix:*`, `v29`, `v30` | arithmetic above | `tools/export_adsb_ctx4.py` | — |
| `scope`, `airport`, `month` | scopes from `run_adsb_v11.SCOPES` (+`lirf_nonpop_unmatched` in `run_adsb_v24`) on proxy_s, schedule_interval_s, population, adsb_matched | `export_adsb_v11.ranking_table` | — |

## 2. Stages, in dependency order

**S0 Input contract.** Today: `data/raw/ranking.parquet` (DEP + ARR rows, 30 raw columns, arrivals keep taxi-in and in-block),
template `data/raw/submitting.parquet` (MVT_ID order/types). A new file must also carry arrival rows (turnarounds for v9,
arrival tokens, timeline surface context). If it does not → blocker B1.

**S1 Phase-1 row features + movement context.** First: `src/prc2026/phase1.py` `prepare` (validates raw hashes against download
receipts → refuses a new file) writing `data/phase1/58d621b245de3f5e12dd/{features,context}/ranking.parquet`. For a new
file use the functions directly (`prc2026.features.movement_context/ranking_view`, `prc2026_phase3.row_features`, which
reproduces the Phase-1 names), as the December builder does. Local, seconds. Label-free.

**S2 Feature layers phase3 / phase4 / phase5R / weather / nmclock / timeline (335 cols, 276 predictors).** First: each frozen
bundle's `prepare` (timeline: 108.66 s, 6.87 GiB, arcw0). Guards: every `bundle.py` checks `ranking_months == ["2026-01",
"2026-07"]` and the ranking IDs hash; five feature modules carry one "reserved/undeclared period" `raise`. Template:
`src/prc2026_december/patched.py` (loads the five modules with that single guard replaced, frozen files untouched) +
`features.build(root, month, ...)` (`tools/build_december_features.py --month M [--verify]`), which reproduces the frozen
November rows exactly (331 exact columns, 4 cyclic NM-clock float64 columns within 1.11e-16). Needs the full movement set
as context: the frozen ranking rows used context = Jan–Nov 2025 + the ranking file's 2026-01/07 (headway features look back
to the previous n same-runway departures however old). Local. Label-free (labels only read to save them — make optional).

**S3 External weather/runways.** IEM ASOS (public domain) for the ten stations: `tools/pull_prc_weather.py` (frozen 30 objects
cover 2025 + Jan/Jul 2026) and `tools/pull_december_weather.py` (separate directory + inventory, the pattern to copy);
one request at a time, provider cooldown. OurAirports runways static (`docs/competition/weather_20260914/`). A new date range
→ new pull + inventory passed as `extra_weather_inventories`.

**S4 ADS-B (adsb.lol globe_history, ODbL).**
- Extraction `tools/adsb_extract_day.py --day YYYY.MM.DD` (streams the day's GitHub release tar parts, 2–4 GB/day;
  `globe_history_2026` repo for 2026; ~33 s/day with 14 processes on an Azure worker). The workstation already holds 413
  extracted days — 351 of 2025 (a few May/June days were never downloaded) + all 62 of Jan/Jul 2026 (`data/external/adsblol/surface/`). New dates → extraction (Azure or local).
- Matching, two tables with FROZEN stands: learners/nets/tokens use `data/external/adsblol/matches` (1,332 stands),
  the residual stage uses `data/external/adsblol/matches_v11`. `tools/build_adsb_features.py` maps 2026 months to
  `ranking.parquet` (hard-coded `raw_path`) and refuses December; `tools/build_december_matches.py` is the frozen-stands
  copy to generalise. Local, minutes per month.
- v19 extra features: `tools/build_adsb_v19_features.py --months ...` → `data/external/adsblol/features_v19/`.
- ADS-B caches (27 learner features per raw file): built inside `run_adsb_nn/learner/ctx build_tables` from raw + matches;
  keyed `"ranking.parquet"` with `RANKING_MONTHS`. Exports check the cache hash `1dcf5c7f…` (v29/v30 guard).

**S5 Context tokens.** `tools/build_ctx_tokens.py` (`SOURCES` hard-codes the split `("ranking", RANKING_MONTHS,
"ranking.parquet")`; `--only`), `data/ctx/tokens_v1` (1.7 GB, ~25 s per VM). Needs every movement (arrivals + departures,
ADS-B matches) in [t0−60, t0+30] min. Exports compare each seed's `inputs.tokens.files` with the declaration's
`tokens_manifest_content["ranking"]` → a new split key is needed.

**S6 v7 / v9.** v7 frozen: `prc2026_surface_refit.inference.pipeline_prediction(run, pipeline.json, features[TIMELINE_NAMES])`
(as `prc2026_december.scoring.build_scratch`). v9_frozen: `run_lirf_mixture.fit_population_model(movements(Jan–Nov 2025))`
+ `score_population(model, component, <new movements>, ids)` + `apply_candidate(w=0.75, threshold 43,200 s, floor −12)`.
v7₁₂: `tools/run_v7_dec.py --fold refit --resume` (prediction stage from saved nodes; the fit took 1,968 s / 17.8 GB on a
16-core Azure VM; reads the ranking rows from the timeline bundle). Guards in `export_adsb_v25`: `predicted_ids_sha256`,
uploaded-v9 reproduction, v22 recorded v9.

**S7 Committee + LIRF tree tables (29 leaves).** Re-apply with `prc2026_december.scoring.learner_prediction` (fold `refit`)
or the verifier path of `tools/verify_december_refit.py` (twelve-month fitting-row vocabularies; it re-predicted 3,000
ranking rows per leaf with difference 0.0). Frame = timeline features + learner ADS-B table + v19 extras. Table = mean of
finite leaf predictions over `leaf_members(run_dir)`. CPU; CatBoosts are 9–14k trees depth 8–10 (estimate 30–60 min for
all leaves on 344k rows, 12 threads).

**S8 Networks (144 seeds).** Load + predict: `prc2026_nn` (nnb/nnl), `prc2026_ctx.train.load/predict` + `Context.build`
over a `TokenStore` split (CPU seeds), `prc2026_ctx3.train.load/predict` (e32; predicts on CPU within the verifier's
0.5 s RMS / 2 s p99 tolerance of the GPU values). Tabular frame via `verify_adsb_nn.ranking_sample` logic (bundle ranking
features + cached ADS-B table + v19). No GPU needed. Estimate: ~1 h CPU for the 96 context nets, minutes for the MLPs.

**S9 Residual stage.** nb table: `export_adsb_v11.ranking_table(config, departures, stands_v11, v9)` with neighbour features
(`tools/rnn_ranking_table.py` saved `runs/adsb_rnn/ranking_nb.parquet`; its scope must equal v25's recorded scope).
**Neighbour columns are not bit-reproducible across rebuilds** (unstable sort moves a few dozen `nb_runway_prev/next`
values) — the rnet refits and verifier read the saved table for that reason. adj2: per-target-month boosters, **only 2026-01
and 2026-07 exist** (weights by circular month distance, `configs/adsb_v25_v1.json` `residual_variants.adj2.weights`); other
months need `export_adsb_v25.py --stage fit`-style refit on `scratch/v24` (321.8 s on 16 threads). cat: one "all" CatBoost
(reusable). rnet: 16 `refit12/model.pt` via `run_adsb_rnn.predict_rows`. Combine 0.05/0.05/0.9.

**S10 Recipe + export.** `export_adsb_ctx4.py` rebuilds v29 from the recorded detail (not from features) and refuses unless
integers equal the uploaded v29 → a new-file runner must compute every component itself and skip the reproduction guards;
floor −12 read from `scratch/v24/2025-12.parquet`; `prc2026_submission.contract.export_submission(values, template, output,
provenance)` — the template path is an argument (currently `data/raw/submitting.parquet` at every call site).
Verify: `verify_adsb_ctx4.py` (same v29 coupling). Audit: `tools/audit_export.py` (reads `data/raw/ranking.parquet`).

## 3. Blockers

- **B1 File format unknown.** No arrivals / no in-block times / a different column set would break v9 turnaround features,
  arrival tokens and timeline surface context. Check the file the day it appears.
- **B2 New dates need new external data**: adsb.lol extraction (GitHub release download per day; Azure recommended), IEM
  weather pull, v19 features, tokens split. Unknown availability lag of adsb.lol releases for recent days.
- **B3 adj2 is target-month specific** (2026-01/07). Other months → ~5 min refit (cheap, but a recipe decision: nearest
  month weights, recorded in the runner).
- **B4 Hard-coded ranking plumbing**: `RANKING_MONTHS` in ~20 tools, `"ranking.parquet"` keys in `build_tables`,
  `build_ctx_tokens.SOURCES`, `build_adsb_features.raw_path`, bundle `validate_config`, Phase-1 raw-hash validation,
  export guards against uploaded v9/v22/v29, `rnn_ranking_table` scope-equality guard.
- **B5 Non-bit-reproducible neighbour features** → the dry run cannot demand exact residuals; tolerance needed.
- **B6 Context set for timeline/traffic features** must be declared (Jan–Nov 2025 + ranking months + the new file; December
  was never in the frozen context).
- **B7 Workstation memory** (15.7 GB): the v7₁₂ prediction and a full-frame token/context build may need an Azure VM.

## 4. Smallest code changes (stage 2)

One new package + one runner, frozen files untouched (the December pattern):
1. `src/prc2026_final/features.py`: generalise `prc2026_december.features.build` to `(raw_path, months, context_sources)`;
   labels optional; Phase-1 names from `row_features`, context via `movement_context`; `--verify` mode compares with the
   frozen timeline `features/ranking.parquet`.
2. `src/prc2026_final/adsb.py`: frozen-stands matching for arbitrary days/raw file (from `build_december_matches.py`), both
   match dirs; v19 features; the 27-column learner ADS-B table; the residual nb table.
3. Tokens: `build_ctx_tokens.py --source NAME RAW MONTHS` (new split key) — tokens code imported, not edited.
4. `tools/final_predict.py --raw F --template T --root OUT [--weather-inventory J]`: v7 (pipeline), v9 (mixture + v7₁₂ nodes),
   29 leaves, 144 nets, residual (adj2 for the file's months, cat, rnet), recipe arithmetic imported from the frozen
   modules, `export_submission` with the given template; writes a detail parquet with the same columns as v30's.
5. A comparison tool for the dry run.

## 5. Runbook (new file, estimated wall time)

1. Inspect the file (columns, months, arrivals, IDs vs template) — 10 min.
2. ADS-B extraction for its days on Azure (`adsb_extract_day.py`, ~0.5–1 min/day/worker incl. download) and copy back — ~1 h per month.
3. IEM pull for the period — 15–30 min (throttled).
4. Phase-1 + six feature layers (S1–S2) — ~10 min local.
5. Matches ×2, v19 features, ADS-B caches, tokens — ~30 min.
6. v7 frozen + v9 mixture — minutes; v7₁₂ node prediction — ~10–20 min (Azure if memory-tight).
7. 29 leaves — 30–60 min; 144 nets — ~1–1.5 h (split across VMs if needed).
8. Residual: nb table, adj2 refit per new month (~5 min each), cat, rnet — ~20 min.
9. Recipe, export, independent verify (re-derive sampled rows), audit — ~30 min.
Total ≈ 5–7 h after the file appears, most of it inference; ~2 h less if no new dates (same months).

## 6. Proposed dry run (stage 3)

Copy `data/raw/ranking.parquet` to `runs/final_dryrun/input/final.parquet` and run the whole runner into a fresh root with
`--template data/raw/submitting.parquet`, treating it as unknown (its months are 2026-01/07, so adj2 uses the existing
boosters; ADS-B surface days already exist). Acceptance:
- S1–S2 features equal the frozen timeline `features/ranking.parquet` (exact except the 4 cyclic NM-clock columns ≤ 2e-16).
- Matches/caches/tokens: content hashes equal the recorded ones (`adsb_cache 1dcf5c7f…`, tokens `70ac0cc1…`) or row-level
  equality where timestamps enter manifests.
- v7, v9, committee, `ens_v22_lirf_q`: max |Δ| ≤ 1e-9 against the v30 detail columns.
- CPU nets ≤ 0.01 s max abs per seed; e32 nets ≤ 0.5 s RMS / 2 s p99 (as `verify_adsb_ctx4`); ensembles accordingly.
- Residual: equal on rows whose nb columns reproduce; overall ≤ a declared tolerance (e.g. 99.9% of rows within 1e-6 s).
- Final integers vs the uploaded v30: report the count of differing rows and max |Δ| (expect a handful, from e32 CPU
  reload and nb instability); v30 itself is never overwritten or re-uploaded.
Estimated: stage 2 ≈ 1.5–2.5 days of coding and unit tests; stage 3 ≈ 4–6 h compute (mostly leaves + nets) + review.

## 7. Open questions

- Which Azure workers still hold raw per-day tar parts or extraction scratch (not needed for the dry run — the workstation has
  every surface day — but relevant for new dates). Not checked: round E occupies arcw0–7.
- Whether any 2026 final file would be new months (then B2/B3 apply) or a re-cut of Jan/Jul 2026 (then only plumbing).
- Whether `pipeline_prediction` for the surface refit loads its v6/v5 fallback models from `runs/azure_arcw0_*` mirrors
  completely on the workstation (December scoring did, so likely yes).
