# Final-file runner (stage 2) — `tools/final_predict.py`, `tools/final_compare.py`, `src/prc2026_final/`

v30's recipe on **any movement file with the ranking file's 30-column schema**, from the saved models only. Built from the stage-1
inventory ([`inventory.md`](inventory.md)). Frozen packages, tools and configurations are untouched: every stage imports the frozen
function that first computed its column for the ranking file, and re-implements only plumbing that hard-codes `ranking.parquet`,
`RANKING_MONTHS` or a "must reproduce vN" guard (the December package is the template). No refit happens except the residual
stage's per-target-month adj2 booster, via an explicit subcommand. Nothing is downloaded, uploaded or written to `runs/submissions`.

## Usage

Every command takes `--root DIR`, a fresh directory per input file. Stages are resumable: each writes its outputs under the root and
its receipt `stages/<stage>.json` last (output hashes are re-verified before a stage is skipped; delete a receipt to rebuild it).
Leaves and networks also skip learners whose prediction file already exists, so an interrupted inference stage resumes per model.

```
python tools/final_predict.py inspect      --root R --raw FILE --template TEMPLATE   # S0: schema, months, arrivals, IDs vs template
python tools/final_predict.py external     --root R                                   # what new dates need (prints only)
python tools/final_predict.py extract-adsb --root R [--execute] [--workers 14]        # adsb.lol surface days (2-4 GB download each)
python tools/final_predict.py pull-weather --root R [--execute]                       # IEM station-months -> data/external/iem_final/<R>/
python tools/final_predict.py refit-adj2   --root R --month YYYY-MM [--threads 16]    # declared adj2 refit for a month without a booster
python tools/final_predict.py run          --root R [--stages ...] [--threads 12] [--workers 8]
                                           [--weather-inventory data/external/iem_final/<R>/inventory.json] [--version N] [--detail-ensembles]
python tools/final_compare.py --root R --output OUT.json [--version N] [--allow-subset]
```

External data and model inference are separate commands: `external` / `extract-adsb` / `pull-weather` touch the network (only with
`--execute`), `refit-adj2` is the only fit, and `run` is pure local inference on files already on disk.

## Stages (`run`)

| stage | what | frozen code applied | outputs |
|---|---|---|---|
| features | 335 timeline columns; context = the file's own departure months first, then the frozen 2025 training files filtered to Jan–Nov (the frozen source order) | phase-3 `row_features`/`traffic_features`, the five guarded layers via `prc2026_december.patched`, weather `observation_features` over the frozen 30 IEM objects + `--weather-inventory` | `features/timeline.parquet` |
| adsb | pass-2 matches per month for both match dirs (frozen stands), the learners' 27 ADS-B columns, the 11 v19 columns (month-level) | `match_departures`, `departure_features`, `neighbour_features`, `build_adsb_v19_features.pass_one/pass_two` | `adsb/{learners,residual}/matches2_<m>.parquet`, `adsb/learner_adsb.parquet`, `adsb/v19.parquet` |
| tokens | the context nets' token split `final` (K = 128) | `prc2026_ctx.tokens.build_tokens/encode_codes/neighbours/save_file` | `tokens/final.npz` |
| v9 | v7 (surface-refit pipeline incl. v6/v5 fallback), v9 frozen (Jan–Nov LIRF mixture, refitted from 2025 labels, byte-identical), v7 twelve-month nodes, v9 | `pipeline_prediction`, `run_lirf_mixture`, `run_v7_dec.predict` | `v9/v9.parquet` |
| leaves | 29 twelve-month leaves; committee `ens_v20_bag_q_all` and LIRF table `ens_v22_lirf_q` (mean over the frozen member lists) | `verify_december_refit.TwelveMonthVocabulary` (cached once as JSON), leaf route/offset rules | `leaves/leaf/*.parquet`, `leaves/tables.parquet` |
| networks | `ens_ctx_e24`, `ens_ctx_e24b`, `ens_ctx_e32` (GPU-trained, predicted on CPU), `ens_ctx_lirf` (+ `ens_ctx`, `ens_new`, `ens_nn`, `ens_nnl` with `--detail-ensembles`) | `prc2026_ctx`, `prc2026_ctx3`, `prc2026_nn` load/predict | `networks/<ens>/*.parquet`, `networks/ensembles.parquet` |
| residual | nb table (matches_v11), adj2 per target month, cat, 16 rnet seeds; 0.1 × mean(adj2, cat) + 0.9 × rnet on matched non-LIRF rows | `run_adsb_v25.predict_variant`, `run_adsb_rnn.predict_rows` | `residual/table.parquet`, `residual/residual.parquet` |
| recipe | v30 arithmetic: mix 0.25 e24 + 0.25 e24b + 0.5 e32 at share 0.75, outer 0.7, round-B LIRF blends, floor −12; scope column | `mixed`, `weighted`, `lirf_blends`, `spec_for`, `candidate_values` | `recipe/detail.parquet` (v30 detail's columns) |
| export | nearest-integer file in the template's order (`--version N` names it) | `export_submission` | `export/vigorous-whistle_v<N>.parquet` + receipt |

## A period other than January/July 2026

Nothing in `run` depends on `RANKING_MONTHS` or on ranking-only artifacts. What a new period needs, in order:
1. `inspect` (checks the file carries arrival rows with in-block and taxi-in — the v9 turnaround, arrival tokens and timeline surface
   context need them; blocker B1 if not).
2. `external` lists the surface days without an extracted file and the weather station-months no inventory covers; `extract-adsb
   --execute` (an Azure worker is recommended: 2–4 GB per day from `adsblol/globe_history_<year>`, ~0.5–1 min per day with 14
   processes; copy `surface_<day>.parquet` back) and `pull-weather --execute` (one IEM request at a time, the frozen pull code) fetch them.
   Without them the features stage refuses (`--allow-missing-weather` accepts the frozen missing-weather route) and the ADS-B stage
   leaves those departures unmatched (recorded as a warning).
3. `refit-adj2 --month M` for every month of the file other than 2026-01/07 (the frozen `fit_variant` on the twelve 2025 cross-fit
   months with circular month-distance weights; ~5 min on 16 threads, needs scratch/v24 and runs/adsb_v24/features).
4. `run --weather-inventory data/external/iem_final/<root>/inventory.json`.
Months that overlap the frozen Jan–Nov 2025 context are refused (a re-cut of fitting months needs its own context declaration).
Tested without downloads on a copy of 15 January 2026 shifted to 22 October 2026: `inspect`, `external`, `extract-adsb` and
`pull-weather` report the missing surface day and all ten station-months; the adj2 new-month wiring is unit-tested.

## Stage 3 — the dry run (not run yet)

```
mkdir runs/final_dryrun/input && copy data/raw/ranking.parquet runs/final_dryrun/input/final.parquet
python tools/final_predict.py inspect --root runs/final_dryrun/run --raw runs/final_dryrun/input/final.parquet --template data/raw/submitting.parquet
python tools/final_predict.py run --root runs/final_dryrun/run --threads 12 --workers 8 --version 900 --detail-ensembles
python tools/final_compare.py --root runs/final_dryrun/run --output docs/final_file/dryrun_compare.json --version 900
```
`--version 900` only names the file inside the run root (never copied, never uploaded). Run the long `run` with the Start-Process +
Wait-Process pattern; it can be split with `--stages` across sessions.

**Acceptance** (as proposed in the inventory, tolerances set from the smoke test below):
- features equal the frozen timeline bundle's ranking rows (exact; the 4 cyclic NM-clock float64 columns ≤ 2e-16);
- matches equal the frozen tables (≤ 1e-9: distance columns differ by ~1e-12 in float rounding); the learner ADS-B table equals the
  table every context seed used (`runs/adsb_ctx2/inputs/adsb_features_arcw0/adsb_ranking.parquet`, sha 1dcf5c7f…) except nb_* values
  moved by the unstable neighbour sort (reported); v19 equal (≤ 1e-9); the tokens' content hash equals the frozen ranking split's (70ac0cc1…);
- v7, v7 twelve-month, v9 frozen, v9, committee and ens_v22_lirf_q ≤ 1e-9; every leaf ≤ 1e-6 against its saved ranking predictions;
- CPU network seeds ≤ 0.01 s max abs; e32 seeds ≤ 0.5 s RMS and ≤ 2 s p99 (TF32 GPU vs float32 CPU);
- adj2/cat ≤ 1e-6 (except rows whose nb columns moved), rnet mean and residual ≤ 0.01 s; scope column equal; lirf_mix ≤ 0.01 s;
  committee_mix and v30 ≤ 0.5 s (they carry 0.75 × 0.5 of the e32 difference);
- exported integers vs the uploaded v30: differing rows and max |Δ| reported (expected: a handful of 1-s rounding flips from e32).

**Expected cost** (extrapolated from the one-day smoke; not measured on the full file): features ~10–15 min, adsb ~5–10 min, tokens
< 1 min, v9 ~3–5 min, leaves ~30–60 min (+~4 min for the one-time vocabulary cache), networks ~30–60 min for the 64 recipe nets
(~1.5 h with `--detail-ensembles`, 144 nets), residual ~5 min, recipe/export seconds: **~1.5–2.5 h** with 12 threads. Memory: the
features stage holds the 2025 context (~2.2 M movements) and the leaves/networks stages a 344,841 × ~373 frame; plan for ~4–6 GB free
(the workstation usually has 4–6 GB) or run it on an Azure worker. Peak memory was not measured on the full file.

## Smoke test (27 September, one day: 15 January 2026, 9,976 movements, 5,001 departures)

Every stage ran end to end (features 189 s, adsb 40 s, tokens 0.1 s, v9 20 s, leaves 304 s incl. the vocabulary cache, networks 75 s,
residual 13 s, recipe/export < 1 s) and `final_compare.py --allow-subset` ran on it. A one-day file truncates context, so rows near the
day's edges and context-dependent columns (headway "last 10 gaps", schedule-demand windows, last stand arrival, file-level neighbour
presence, month-level v19 queue points) legitimately differ; on the **2,627 departures whose every input is exact**: v7 and v9 exact
(0.0), all 29 leaves and both tree tables exact (0.0), CPU context nets ≤ 0.0012 s, e32 nets max 1.18 / RMS 0.053 / p99 0.18 s, adj2
and cat exact, residual ≤ 1.1e-4 s (Linux-saved float32 rnet reloaded on Windows), v30 ≤ 0.035 s; the scope column equals v30's on all
5,001 rows; matches equal the frozen tables within 1.8e-12. Tests: `tests/test_final_runner.py` (12 tests, incl. v30's arithmetic on
the five routes and v9 reproduced on 300 ranking departures), plus the existing ctx/ctx3/LIRF-mixture/submission tests: 44 pass.

## A code-run scenario (the organisers run our code on their file)

Additionally needed beyond this runner:
- **Environment**: Python 3.14 with the pinned research stack (numpy 2.4, pandas 3.0, pyarrow 23, LightGBM, CatBoost, PyTorch CPU,
  scikit-learn); CPU only (the e32 nets predict on CPU within tolerance). The frozen v9 mixture classifier refit is byte-identical on
  Windows but not on Linux (docs: v25 round) — for a package, ship the fitted classifier instead of refitting it.
- **Model files** (measured): the 29 leaves 4.59 GB (the CatBoosts dominate), the surface-refit run incl. v6/v5 fallback 235 MB, v7
  twelve-month nodes 101 MB, residual trees 221 MB, 16 rnet seeds 44 MB, the 64 recipe networks 644 MB (the 80 detail-only nets are
  not needed): **~5.8 GB**. Plus the twelve-month vocabulary JSON (0.5 MB, replaces the December build and the 2025 labels) and the
  frozen stands tables.
- **Data they must supply or we must ship**: the 2025 movement files as feature context (272 MB raw; the Phase-1 context parquet
  91 MB) and 2025 labels for the mixture classifier unless it is shipped fitted; the frozen IEM objects (18 MB) + OurAirports runways;
  adsb.lol surface files for their period (~12 MB per extracted day; extraction needs internet and 2–4 GB download per day).
  The challenge data's licence is unknown: do not put challenge rows or row-level derived tables in a public repo.
- **Run time**: ~1.5–2.5 h on a 12-thread CPU for a 345k-departure file after extraction (extraction ~0.5–1 min per day per 14-core VM).
