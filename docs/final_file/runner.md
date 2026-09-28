# Final-file runner (stage 2) — `tools/final_predict.py`, `tools/final_compare.py`, `src/prc2026_final/`

**The v35 incumbent** on **any movement file with the ranking file's 30-column schema**, from the saved models only: v30's chain
(stages features → recipe), then v31's EOBT_1 repair, the v33/v34 recipe weights and v35's stack (stages repair → final; see
[v31–v35 layers](#v31v35-layers-28-september) below). Built from the stage-1 inventory ([`inventory.md`](inventory.md)). Frozen
packages, tools and configurations are untouched: every stage imports the frozen function that first computed its column for the
ranking file, and re-implements only plumbing that hard-codes `ranking.parquet`, `RANKING_MONTHS`, the ranking's match tables or a
"must reproduce vN" guard (the December package is the template). No refit happens except the residual stage's per-target-month adj2
booster, via an explicit subcommand. Nothing is downloaded, uploaded or written to `runs/submissions`.

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
                                           [--vocab-cache runs/final_dryrun/run/leaves/vocabularies.json] [--repair-root DIR]
python tools/final_compare.py --root R --output OUT.json [--version N] [--allow-subset]
```

External data and model inference are separate commands: `external` / `extract-adsb` / `pull-weather` touch the network (only with
`--execute`), `refit-adj2` is the only fit, and `run` is pure local inference on files already on disk.

- **`extract-adsb --execute` downloads on the machine it runs on.** Each day is a 2–4 GB download from `adsblol/globe_history_<year>`,
  with `--workers` processes. The commands it prints use the Azure form (`.venv/bin/python`), but `--execute` runs them locally with
  the current interpreter. For more than a few days, run the printed commands on an Azure worker and copy
  `data/external/adsblol/surface/surface_<day>.parquet` back.
- **`--vocab-cache`** copies an earlier root's `leaves/vocabularies.json` into a new root before the leaves stage, if the new root
  has no cache yet. This skips the twelve-month vocabulary rebuild, which reads the December build and the 2025 labels. The cache
  depends only on frozen training data, so any completed run's copy is valid. Keys that a leaf needs but the copy lacks are still
  computed. The leaves receipt records the seed and the cache's SHA-256.

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
| repair | v31's input: label-free EOBT_1 detector per airport-month; for a flagged one, EOBT_1 := IOBT where EOBT_1 > IOBT and the chain features → recipe again in `<root>/repair/run` (or `--repair-root`, a completed run of exactly that content) | the chain's own stages | `repair/shares.parquet`, `repair/edited_ids.parquet` |
| rematch | the callsign fix's ADS-B rematch of the no-NM EDDF / EHAM departures | `tools/adsb_rematch_nonm.py` main, unchanged (its `REPO` pointed at this workspace) | `rematch/{deps,out}.parquet` |
| stack | v31 → v33 → v34 → v35 on the recipe detail (the exports' construction; deltas from the file's own detail) | `export_v33.recipe`, `export_fair_weights.recipe` + `V33_OUTER`/`V33_LIRF`, `export_disagreement_gate.V34_LIRF`/`gate_delta`, `run_callsign_fix.scope_table`/`offsets`, `run_queue_trust` segment rule/`offsets_all` | `stack/detail.parquet` (v30, repair effect, v31, v33, v34, v35, layer masks) |
| final | v35 as the nearest-integer file in the template's order (`--version N` names it) | `export_submission` | `final/vigorous-whistle_v<N>.parquet` + receipt |
| export | only when named (`--stages export`): v30's own file, a component check against the uploaded v30 | `export_submission` | `export/vigorous-whistle_v<N>.parquet` + receipt |

## v31–v35 layers (28 September)

The runner stopped at v30. The four stages after `recipe` bring it to the v35 incumbent (226.8989 official), in shipping order, each
through the frozen code that built the layer. Module: `src/prc2026_final/stack.py`.

1. **v31, the EOBT_1 repair (stage `repair`).**
   - **Detector (label-free).** For each departure airport-month of the file: the share of departures with EOBT_1 > IOBT (a missing
     clock counts as not later). It is flagged when the share exceeds the airport's **2025 maximum by more than the 2025 span**
     (maximum − minimum over its twelve 2025 months, read from `data/raw/training_2025-*.parquet`). Upward only: a share below the
     range is never an EOBT_1-announcement break.
   - **Edit.** In a flagged airport-month, EOBT_1 := IOBT on the departures with EOBT_1 > IOBT (the sensitivity run's edit). The chain
     features → recipe runs again on the edited file in `<root>/repair/run` (about 1.5 h; `--repair-root` reuses a completed run
     whose input has exactly that content). v31 = max(v30 + 0.5 × (v30 edited − v30), −12), as `tools/export_lfpg_eobt_blend.py`.
     No flag: v31 = v30 and nothing reruns.
   - **On the ranking file it reduces to v31's rows exactly.** Only LFPG 2026-07 is flagged: share 34.5% against the threshold 10.2%
     (2025: 3.8–7.0%). The closest unflagged airport-month is LEMD 2026-01, 21.9% against 23.7% (margin −1.8 points). A plain
     "outside the 2025 range" rule would also flag LEMD January (above its 20.6% maximum) and LIRF January (below its minimum), hence
     the one-span margin and the upward direction. The edit is the same 7,510 rows, with content identical to
     `runs/final_lfpg_eobt/input/final.parquet` (input SHA-256 `cc474432…`). The effect is non-zero on 21,575 rows, all LFPG July.
2. **v33 / v34, the recipe weights (stage `stack`).**
   - v33 = max(v31 + `export_v33.recipe`(detail, 1.0, 0.6, 0.7) − `recipe`(detail, 0.7, 0.7, 0.7), −12).
   - v34 = max(v33 + `export_fair_weights.recipe`(detail, `V33_OUTER`, `V34_LIRF`) − `recipe`(detail, `V33_OUTER`, `V33_LIRF`), −12).
   - That gives outer weights LTFM 1.0 / matched 0.6 / unmatched 0.7; LIRF matched v9 .15 / net .40 / tree .45; LIRF unmatched v9 0 /
     net .35 / tree .25.
   - Guards: the detail must be v30's recipe under both frozen rebuilds (`export_v33` at 0.7 and `export_fair_weights` at v30's
     weights, ≤ 1e-9).
3. **v35, the stack (stages `rematch`, `stack`), following `tools/export_stack.layers`.**
   - **(a) u300.** `export_disagreement_gate.gate_delta` with `configs/disagreement_gate_v1.json`'s u300: on unmatched non-LIRF
     non-LTFM rows where the committee mix exceeds v9 + residual by more than 300 s, the value is the mix.
   - **(b) Callsign fix, 50%.** `tools/adsb_rematch_nonm.py` (unchanged; its `REPO` pointed at this workspace) on the departures
     without an NM flight record at EDDF / EHAM. `run_callsign_fix.scope_table` then keeps rows where the key found nothing and the
     IATA → ICAO callsign gives a gate segment. value = max(0.5 × (takeoff − motion_start + offset) + 0.5 × value, −12). The offsets
     are `run_callsign_fix.offsets()`, constants from 2025 (EDDF 73.72 s, EHAM 6.49 s).
   - **(c) Queue trust, 0.75, regular matches.** Gate-covered segments from the run's own residual match tables (the `matches_v11`
     stands) plus the callsign rows. The scope is `run_queue_trust`'s rule: ≥ 5 stops, ≥ 5 km, stationary ≤ 75%, taxi > 3,000 s.
     Only the regular matches are applied. value = max(0.75 × (taxi + `offsets_all()`[airport]) + 0.25 × value, −12).
   - `export_stack.layers` reads the frozen ranking tables, so the runner re-plumbs `run_queue_trust.segments` onto the run's
     tables: same columns, rule and duplicate guard.
   - Weights are read from the declarations: gate u300, callsign `u300_cs50` 0.5, queue `qt75` 0.75 on regular matches, as
     `docs/disagreement/export_v35.json` shipped them.
4. **`final`** writes v35's integer file into the run root. `export` (v30's own file) now runs only when named.

**Dry run (28 September): passes.** The new stages ran on `runs/final_dryrun/run`, reusing its completed chain, with
`--repair-root runs/final_lfpg_eobt/run --version 935`: repair 11 s, rematch 78 s, stack 2 s, final < 1 s.
`tools/final_compare_v35.py` → [`dryrun_v35_compare.json`](dryrun_v35_compare.json):
- **Repair.** It flags exactly LFPG 2026-07 with 7,510 edited rows (input hash equal to the recorded one). The effect equals the
  export's LFPG effect exactly (max |Δ| 0.0).
- **Rematch.** 1,627 departures, 0 surface days missing. It equals the Azure ranking rematch (`runs/adsb_rematch/out_arcw1.parquet`)
  on every row and column within 1e-6. Five `iata_mapped_taxi_path_m` values differ in float rounding.
- **Layers.** u300 **758**, callsign **163**, queue **161**, equal to `export_v35.json` layer_rows. Every mask equals
  `export_stack.layers`' mask on the frozen tables, row for row. `export_stack.layers` applied to the runner's own v34 on the frozen
  tables returns the runner's v35 **exactly** (max |Δ| 0.0).
- **Floats.** Runner v35 − shipped v35 has RMS 0.031 s and maximum 6.3 s. On all 344,839 rows off the floor this equals the dry run's
  v30 noise (runner v30 − recorded v30, the e32 GPU → CPU difference) carried through the layer weights (×1, ×0.5 on callsign rows,
  ×0.25 on queue rows), to 2.7e-12.
- **Integers.** Against the uploaded v35 (`75f4a4f4…`), **1,140 rows differ, maximum 6 s**: 1,118 by ±1 s and 22 by 2–6 s. This
  matches the v30 dry run's 1,141 rows / 7 s. The rows differ from the v30 dry run's rows because the rounding boundaries moved; the
  noise is the same. Within the layers, 14 u300 and 3 queue rows differ, all noise. Output:
  `runs/final_dryrun/run/final/vigorous-whistle_v935.parquet` (SHA-256 `ee04cbd3…`), never copied or uploaded.
- **Tests.** `tests/test_final_stack.py` (5 tests) covers the detector (one-span margin, upward only, airports without a 2025
  reference), the reset edit, the stack arithmetic hand-computed on eight routes, the v30 guard and the rematch scope. With
  `tests/test_final_runner.py`, 18 pass.

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
4. `run --weather-inventory data/external/iem_final/<root>/inventory.json`. If the detector flags an airport-month, `repair` runs
   the chain a second time on the edited input (about 1.5 h more; the sub-root reuses the main root's vocabulary cache). The rematch
   reads the same surface days as the ADS-B stage; days missing there leave their no-NM EDDF / EHAM departures without a callsign fix.
   The detector's 2025 reference and both offset tables are constants from the 2025 files, so they apply to any later period.
Months that overlap the frozen Jan–Nov 2025 context are refused (a re-cut of fitting months needs its own context declaration).
Tested without downloads on a copy of 15 January 2026 shifted to 22 October 2026: `inspect`, `external`, `extract-adsb` and
`pull-weather` report the missing surface day and all ten station-months; the adj2 new-month wiring is unit-tested.

## Stage 3 — the dry run (done 27 September: v30 to 0.031 s RMS, [`dryrun.md`](dryrun.md); the plan as written before it ran)

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
- **v31–v35 layers** add no models. The detector reads the 2025 movement files, which are already shipped as context. The rematch
  reads the `matches_v11` stands table and the period's surface files. The callsign / queue offsets come from the 2025 labels and
  the `matches_v11` tables of eight 2025 months; ship the two small offset dictionaries instead. A repair flag doubles the run time.
- **Data they must supply or we must ship**: the 2025 movement files as feature context (272 MB raw; the Phase-1 context parquet
  91 MB) and 2025 labels for the mixture classifier unless it is shipped fitted; the frozen IEM objects (18 MB) + OurAirports runways;
  adsb.lol surface files for their period (~12 MB per extracted day; extraction needs internet and 2–4 GB download per day).
  The challenge data's licence is unknown: do not put challenge rows or row-level derived tables in a public repo.
- **Run time**: ~1.5–2.5 h on a 12-thread CPU for a 345k-departure file after extraction (extraction ~0.5–1 min per day per 14-core VM).
