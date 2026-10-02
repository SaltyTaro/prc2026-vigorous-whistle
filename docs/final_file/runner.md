# Final-file runner (stage 2) — `tools/final_predict.py`, `tools/final_compare.py`, `src/prc2026_final/`

**The v38 incumbent (or v35 / v36 / v37 with `--layers`)** on **any movement file with the ranking file's 30-column schema**, from
the saved models only: v30's chain (stages features → recipe), then v31's EOBT_1 repair, the v33/v34 recipe weights and the stack
(stages repair → final; see [v31–v38 layers](#v31v38-layers-28-september) below). Built from the stage-1 inventory ([`inventory.md`](inventory.md)). Frozen
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
                                           [--layers v35|v36|v37|v38|v39|v40|v41|v45|v46|v51m|v52m|v53m|v54m|v55m]
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
| members | the 48 twelve-month member networks of v51 / v52 (`combo_dl32`, `combo_m25dl32`, `mask_m25_32`, 16 seeds each, GPU-trained, predicted on CPU), seed means, routed by the file's ADS-B match flag into `combo_all` and `mask_unmatched` (`src/prc2026_final/members.py`) | `prc2026_ctx3` load/predict via `networks.predict_learner` | `members/<ensemble>/*.parquet`, `members/ensembles.parquet`, `members/members.parquet` |
| stack | v31 → v33 → v34, then v35 / v36 / v37 / v38 on the recipe detail (the exports' construction; deltas from the file's own detail), then v39 = v38 + the fill pins of the file's own LIRF long-interval departures (`configs/fill_pin_v1.json`, `src/prc2026_final/fill_pin.py`) | `export_v33.recipe`, `export_fair_weights.recipe` + `V33_OUTER`/`V33_LIRF`, `export_disagreement_gate.V34_LIRF`; `tools/export_stack.layers` itself, its readers answering from the run's tables (`run_callsign_fix.scope_table`, `run_queue_trust` segment rule, `peer_congestion.peer_table`) | `stack/detail.parquet` (v30, repair effect, v31, v33, v34, v35–v38, per-set layer masks), `stack/peers.parquet` |
| final | the `--layers` set (default v38) as the nearest-integer file in the template's order (`--version N` names it); receipt `stages/final-<layers>.json` | `export_submission` | `final/vigorous-whistle_v<N>.parquet` + receipt |
| export | only when named (`--stages export`): v30's own file, a component check against the uploaded v30 | `export_submission` | `export/vigorous-whistle_v<N>.parquet` + receipt |

## v31–v38 layers (28 September)

The runner stopped at v30. The four stages after `recipe` take it to the v35 incumbent (226.8989 official) and to the exported,
not-uploaded v36, v37 and v38, in shipping order, each through the frozen code that built the layer. Module:
`src/prc2026_final/stack.py`.

**`--layers v35|v36|v37|v38|v39|v40|v41|v45|v46|v51m|v52m|v53m|v54m|v55m`** picks the final file's layer set. The default is **v54m**
(2 October: v52m with the masked LIRF member and the dark-month re-read, the model part of v54 before the daily-totals reconciliation;
see [v53m / v54m / v55m](#v53m--v54m--v55m-2-october) at the end; v55m is built but never uploaded). Before that the default was
**v52m** (30 September: v46 with the two member networks in the network table, the model part of v52 before the daily-totals
reconciliation; see [v51m / v52m](#v51m--v52m-30-september)). Before that the default was **v46** (29 September: v45 with the
committee mix of `configs/thinned_regime_round_v1.json` `n85_nd1200`, officially 223.9339, the clean model line's best;
`docs/thinned_regime/README.md`; masks `v46_share`, `v46_share_gate`). Before that it was **v45** (v40 plus the LIRF
disagreement gate of `configs/lirf_gate_v1.json`, officially 224.2398; `docs/lirf_gate/README.md`; mask
`v45_lirf_gate`), and before that **v40**, the official best from its upload
(224.4588 on 28 September; v38 226.8053, v39 226.8802, v35 226.8989). **v39** (`docs/fill_pin/README.md`) is v38 with every fill pin:
the file's own LIRF departures with a schedule interval above 43,200 s that are the v9 population (no NM record, no aircraft type) or
carry a one-day-early NM clock take their schedule interval (`stack/fill_pins.parquet` lists them). **v40** is v38 with
`configs/fill_pin_v2.json`'s `evidenced_nine`: population rows only, with intervals in [43,200, 58,000] or at or above 79,200 s.
The stack stage computes all six columns every time. The final stage exports the chosen one and keeps one receipt
per layer set (`stages/final-<layers>.json`, which records the layer set and version), so several files can live in one root.

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
2. **v33 / v34, the recipe weights (stage `stack`).** This is `export_disagreement_gate.rebuild_v34`'s arithmetic with the file's
   own v30:
   - v33 = max(v31 + `export_v33.recipe`(detail, 1.0, 0.6, 0.7) − `recipe`(detail, 0.7, 0.7, 0.7), −12).
   - v34 = max(v33 + `export_fair_weights.recipe`(detail, `V33_OUTER`, `V34_LIRF`) − `recipe`(detail, `V33_OUTER`, `V33_LIRF`), −12).
   - That gives outer weights LTFM 1.0 / matched 0.6 / unmatched 0.7; LIRF matched v9 .15 / net .40 / tree .45; LIRF unmatched v9 0 /
     net .35 / tree .25.
   - Guards: the detail must be v30's recipe under both frozen rebuilds (`export_v33` at 0.7 and `export_fair_weights` at v30's
     weights, ≤ 1e-9).
3. **v35 – v38 (stages `rematch`, `stack`): `tools/export_stack.layers` itself.** The runner calls the frozen function for each layer
   set. During the call (`stack.run_tables`), its three ranking-bound readers answer from the run's own tables:
   - `scope_table` gets the file's rematch through `run_callsign_fix.scope_table`;
   - `segments` gets `run_queue_trust`'s segment rule on the run's residual match tables (the `matches_v11` stands) plus the
     callsign rows;
   - `ranking_peers` gets `tools/peer_congestion.peer_table` on the movement file itself.

   Everything else is `export_stack`'s own code: `gated_detail`, `gate_delta`, the callsign and queue blends, `peer_lifts` and
   `tail_stretch`. The rematch is `tools/adsb_rematch_nonm.py`, unchanged, with its `REPO` pointed at this workspace, on the
   departures without an NM flight record at EDDF / EHAM.
   - **v35** = `layers(v34, …, 0.75, "regular")`:
     - u300: on unmatched non-LIRF non-LTFM rows where the mix exceeds v9 + residual by more than 300 s, the value is the mix;
     - callsign fix 50%: offsets EDDF 73.72 s / EHAM 6.49 s from `run_callsign_fix.offsets()`;
     - queue trust 0.75 on regular gate matches: ≥ 5 stops, ≥ 5 km, stationary ≤ 75%, taxi > 3,000 s, `offsets_all()`.
   - **v36** = the same with `mix_up=600, peer=True`, from `configs/higher_bundle_v2.json` `mu_pc_pcc`:
     - the mu600 mix gate: network share 1.0 where nets − committee > 600 s, with the recipe and u300 recomputed on the gated mix;
     - then the peer-congestion lifts: +100 s on non-LIRF rows without an NM clock and +50 s on untracked non-LIRF rows with one, in
       airport-hours with ≥ 5 NM proxies whose mean runs more than 600 s above the airport-month's median.
   - **v37** = the same with `stretch=1500`, from `configs/tail_stretch_v1.json` `v35_t1500`. On non-LIRF rows outside the callsign
     and queue layers, value + k × (value − 1,500) above 1,500 s. k is the four-month fit, untracked 0.1258 / tracked 0.0848.
   - **v38** = the same with `stretch=1500, stretch_lower=900`, from `configs/tail_stretch_v2.json` `two_sided_900`. On every
     unanchored non-LIRF row: u = value + k1 × max(value − 1,500, 0), then max(u + k2 × max(900 − u, 0), −12). k1 is v37's;
     k2 is untracked −0.0436 / tracked −0.0362.
   - **v39** = v38, then `fill_pin.apply_pins` with `fill_pin.fill_pins` of the file's own rows (`configs/fill_pin_v1.json`, 28 September):
     LIRF departures with a schedule interval above 43,200 s that have no NM record and no aircraft type (the v9 population), or whose
     NM off-block clock is one day early (takeoff − AOBT_3 within 7,200 s of 86,400 s), take their schedule interval. No coefficient;
     the ranking file has 12 such rows (`docs/fill_pin/export_v39.json`). Saved as `stack/fill_pins.parquet`, mask `v39_fill_pin`.
     Dry run (28 September, stack 7 s, final 0.3 s, `--layers v39 --version 939`): [`dryrun_v39_compare.json`](dryrun_v39_compare.json),
     the 12 pinned rows equal the export exactly, v939 − v938 is non-zero on exactly those rows, and the other 1,157 differing rows
     are the same float-noise rows as v938 vs v38 (≤ 7 s).
   - **v40** = v38, then `fill_pin.candidate_pins(fill_pins, evidenced_nine)` from `configs/fill_pin_v2.json`: population scope only,
     interval bands [43,200, 58,000] and [79,200, ∞) s. On the ranking file: 9 rows (`docs/fill_pin/export_v40.json`); mask
     `v40_fill_pin`. Dry run: [`dryrun_v40_compare.json`](dryrun_v40_compare.json).
   - **v41** = v40, then `configs/fill_pin_v3.json` `nosos431_one_day`: population rows with intervals of 100,000 s or more take
     86,400 s + the 2025 one-day taxi mean (87,505 s). `--layers v41`.
4. **`final`** writes the chosen layer set's integer file into the run root. `export` (v30's own file) runs only when named.

**Dry run (28 September): all four pass.** The new stages ran on `runs/final_dryrun/run`, reusing its completed chain, with
`--repair-root runs/final_lfpg_eobt/run` and `--layers v3N --version 93N` (repair 11 s, rematch 78 s, stack 6 s, final < 1 s each).
`tools/final_compare_stack.py --layers v3N` → [`dryrun_v35_compare.json`](dryrun_v35_compare.json),
[`v36`](dryrun_v36_compare.json), [`v37`](dryrun_v37_compare.json), [`v38`](dryrun_v38_compare.json):
- **Repair.** It flags exactly LFPG 2026-07 with 7,510 edited rows (input hash equal to the recorded one). The effect equals the
  export's LFPG effect exactly (max |Δ| 0.0).
- **Rematch and peers.** The rematch covers 1,627 departures, with 0 surface days missing. It equals the Azure ranking rematch
  (`runs/adsb_rematch/out_arcw1.parquet`) on every row and column within 1e-6; five `iata_mapped_taxi_path_m` values differ in float
  rounding. The file's peer table equals `ranking_peers()` exactly.
- **Layers.** Every per-layer count equals the shipped export's `layer_rows`, and every mask equals `export_stack.layers`' mask on
  the frozen tables and on the shipped construction, row for row. `export_stack.layers` applied to the runner's v34 on the frozen
  tables returns the runner's values **exactly** (max |Δ| 0.0) for all four.
- **Floats.** Runner − shipped has RMS 0.031–0.032 s and maximum 6.3–6.9 s. It equals the dry run's v30 noise (runner v30 − recorded
  v30, the e32 GPU → CPU difference) carried through the layer slopes: ×1, ×0.5 callsign, ×0.25 queue, and the stretch's slope
  (1 + k1 above 1,500 s, 1 − k2 below 900 s). This holds to ≤ 3.5e-12 on every row off the floor; for v38, 2 rows within float noise
  of a stretch threshold are excluded.

| layer set | layer rows (= export record) | shipped file | integers differing (max) | ±1 s | runner file (never uploaded) |
|---|---|---|---:|---:|---|
| v35 | u300 758, callsign 163, queue 161 | `75f4a4f4…` | 1,140 (6 s) | 1,118 | `v935` `ee04cbd3…` |
| v36 | mix_gate 439, u300 771, callsign 163, queue 161, peer_noclock 609, peer_clock_untracked 1,476 | `ed4bcd36…` | 1,139 (6 s) | 1,117 | `v936` `43842dd6…` |
| v37 | v35's + stretch_untracked 9,982, stretch_tracked 21,299 | `edb11dce…` | 1,174 (7 s) | 1,152 | `v937` `b7dc3f18…` |
| v38 | v35's + stretch_untracked 101,344, stretch_tracked 216,274 | `61264cdb…` | 1,157 (7 s) | 1,135 | `v938` `de40ef9d…` |

The v30 dry run had 1,141 differing rows (max 7 s). The flipped rows move because the rounding boundaries moved; the noise is the
same. The v35 file is byte-identical to the one first built at `0e623b3` (before the switch to `export_stack.layers` itself). Those
v35-only outputs are kept under `runs/final_dryrun/run/superseded_0e623b3/`. Files are under `runs/final_dryrun/run/final/`.

**Tests.** `tests/test_final_stack.py` (5 tests) covers:
- the detector: one-span margin, upward only, airports without a 2025 reference;
- the reset edit;
- the stack arithmetic, hand-computed for all four layer sets on twelve routes, including the patched readers being restored;
- the v30 guard;
- the rematch scope.

With `tests/test_final_runner.py`, 18 pass.

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
  not needed): **~5.8 GB**, plus the 48 member networks of v51m / v52m (702 MB) and the 16 masked LIRF networks of v53m–v55m (159 MB). Plus the twelve-month vocabulary JSON (0.5 MB, replaces the December build and the 2025 labels) and the
  frozen stands tables.
- **v31–v38 layers** add no models. The detector reads the 2025 movement files, which are already shipped as context. The rematch
  reads the `matches_v11` stands table and the period's surface files. The callsign / queue offsets come from the 2025 labels and
  the `matches_v11` tables of eight 2025 months; ship the two small offset dictionaries instead. The peer table reads only the file
  itself. The stretch k values come from `docs/disagreement/tail_stretch_v1/v2_results.json`. A repair flag doubles the run time.
- **Data they must supply or we must ship**: the 2025 movement files as feature context (272 MB raw; the Phase-1 context parquet
  91 MB) and 2025 labels for the mixture classifier unless it is shipped fitted; the frozen IEM objects (18 MB) + OurAirports runways;
  adsb.lol surface files for their period (~12 MB per extracted day; extraction needs internet and 2–4 GB download per day).
  The challenge data's licence is unknown: do not put challenge rows or row-level derived tables in a public repo.
- **Run time**: ~1.5–2.5 h on a 12-thread CPU for a 345k-departure file after extraction (extraction ~0.5–1 min per day per 14-core VM).
   - **v45** = v40, then `configs/lirf_gate_v1.json` `u600_m600` (`src/prc2026_final/lirf_gate.py`): LIRF non-population rows whose LIRF
     tree (`ens_v22_lirf_q`) sits more than 600 s above the LIRF network (`ens_ctx_lirf`) take the leaning blend (unmatched 0 / 0.2 / 0.8,
     matched 0.15 / 0.2 / 0.65 over v9 / network / tree), added as a delta against v34's blend and floored. On the ranking file: 100 rows
     (26 January, 74 July; `docs/lirf_gate/export_v45.json`); mask `v45_lirf_gate`. `--layers v45` (the default since 29 September).
     Dry run (29 September, stack 7.7 s, final 0.4 s, `--layers v45 --version 945`): [`dryrun_v45_compare.json`](dryrun_v45_compare.json),
     1,182 integers differ from the uploaded v45 by at most 7 s (the chain's float noise), the 100 gated rows match.
     **Scored 224.2398 on 29 September (−0.219 s vs v40).**
   - **v46** = v45 with the committee mix of `configs/thinned_regime_round_v1.json` `n85_nd1200` (`src/prc2026_final/share_gate.py`):
     network share 0.85 instead of 0.75, and 0.5 on rows whose committee sits more than 1,200 s above the networks. v34's recipe and
     v38's layers (u300 gate, callsign fix, queue trust, two-sided stretch) are recomputed on that mix through `export_stack.layers`;
     every row whose value changes takes the recomputed value, LIRF rows keep v45. On the ranking file: 263,907 integers change, 55 gate
     rows (40 January, 15 July; `docs/thinned_regime/export_v46.json`); masks `v46_share`, `v46_share_gate`. The declared rule selected
     nothing; the layer is the user's override. `--layers v46` (the default since 29 September).
     Dry run (29 September, stack 8.9 s, final 0.4 s, `--layers v46 --version 946`): [`dryrun_v46_compare.json`](dryrun_v46_compare.json),
     1,196 integers differ from the uploaded v46 by at most 7 s (the chain's float noise), the 55 gate rows match.
     **Scored 223.9339 on 29 September (−0.306 s vs v45): the clean model line's best.**

## v51m / v52m (30 September)

The model parts of v51 and v52, from the saved models only. The EUROCONTROL daily-totals reconciliation that v51 / v52 apply last
is **not** part of the runner: it reads published aggregates of the scored months, so it stays in
`tools/export_daily_totals.py --base FILE` (`configs/daily_totals_v1.json`) and applies to the ranking file only. The runner's
v52m file is therefore the open-data model file (adsb.lol ODbL traces, IEM, OurAirports, OSM, the challenge data).

- **Stage `members`** (new, between `rematch` and `stack`; `src/prc2026_final/members.py`). It predicts the 48 twelve-month
  networks named by the two declarations: `configs/adsb_ctx3_combo_dev_v1.json` (dl32 and m25dl32, seeds 601–616) and
  `configs/adsb_ctx3_mask_dev_v1.json` (m25_32, seeds 701–716), with `networks.predict_learner` on CPU. It writes one file per
  learner, the three seed means (`members/ensembles.parquet`) and the routed members (`members/members.parquet`, match flag from
  `adsb/learner_adsb.parquet`): `combo_all` = dl32 on ADS-B-matched rows, m25dl32 on the others; `mask_unmatched` = m25_32 on
  unmatched rows, empty on matched ones. The receipt records every model's hash. On the ranking file: 2,320.6 s on 12 threads
  (about 47 s per model), 225,278 matched rows.
- **Stack.** With a members receipt present, the stack stage adds two layer sets, built inside the same table block as v46 and in
  `tools/export_network_member.through_stack`'s order. The network table (0.25 e24 + 0.25 e24b + 0.5 e32) takes `combo_all` at
  0.40 where it has a value (v51m), then `mask_unmatched` at 0.40 (v52m); LIRF rows keep the shipped table. The v46 mix
  (network share 0.85, 0.5 where the committee sits more than 1,200 s above the table), v34's recipe and v38's layers are
  recomputed through `export_stack.layers`. Every row whose value moves against the previous set takes the new value (v51m on
  v46, v52m on v51m). Columns `v51m`, `v52m`, masks `*_member`, `*_share_gate`. Without the members stage the stack is
  unchanged and `--layers v51m|v52m` fails.
- **Dry run** (`runs/final_dryrun/run`, members 2,320.6 s, stack 15.3 s, final 0.4 s; `--layers v51m --version 951`,
  `--layers v52m --version 952`). `tools/final_compare_members.py` →
  [`dryrun_members_compare.json`](dryrun_members_compare.json):
  - Every earlier layer column (v30 … v46) equals the stack detail from before the members stage exactly (max |Δ| 0.0).
  - Seed means, runner CPU against the shipped GPU per-seed predictions: RMS 0.065 / 0.053 / 0.057 s (dl32 / m25dl32 / m25_32),
    p99 0.08 s, single-row maxima 23 / 11 / 11 s. The routed members against the exported member tables: RMS 0.064 / 0.029 s,
    with exactly the same rows without a value.
  - Layer rows equal the exports' records: v51m 317,942 member rows, 317,935 moved, 65 share-gate rows
    (`docs/adsb_ctx3_combo/member_export_dry_v51.json`); v52m 101,507 member rows, 101,500 moved, 64 share-gate rows
    (`docs/adsb_ctx3_mask/member_export_dry_v52.json`).

| layer set | shipped member file | integers differing (max) | ±1 s | runner file (never uploaded) |
|---|---|---:|---:|---|
| v51m | `runs/adsb_ctx3_combo/export_dry/vigorous-whistle_v51.parquet` `bd8f73aa…` | 1,669 (10 s) | 1,639 | `v951` `531251ba…` |
| v52m | `runs/adsb_ctx3_mask/export_dry/vigorous-whistle_v52.parquet` `16e9f33a…` | 1,744 (101 s) | 1,713 | `v952` `42b4692d…` |

The runner's v46 column differs from the uploaded v46 on 1,196 integers (max 7 s), the chain's known e32 GPU → CPU noise. The
  member networks add about 500 rows on top of it, nearly all ±1 s. Its committee mix sits 300.041 s above v9 in the shipped construction and 299.988 s in the runner's, so the u300
  disagreement gate (`configs/disagreement_gate_v1.json` `unm_up` 300) fires in the shipped v52 (1,769) and not in the runner
  (1,668). The 0.053 s gap is the members' CPU / GPU seed-mean difference on that row (m25_32 1,800.45 against 1,800.59;
  combination 1,755.02 against 1,755.06). It is a hard threshold meeting float noise, not a construction difference. The same
  can happen on any file; earlier dry runs had no such row by chance.

## v53m / v54m / v55m (2 October)

The LIRF layers of v53, v54 and the not-uploaded v55, from the saved models only (`src/prc2026_final/lirf_members.py`). As for
v51m / v52m, the daily-totals reconciliation that v53 / v54 apply last stays outside the runner: **v54m (the new default) is v54's
open-data model part.**

- **Stage `members`** gains a fourth ensemble, `lirf_lm25_32`: the 16 twelve-month refits (seeds 801–816) of the masked LIRF context
  network of `configs/adsb_ctx3_lirf_mask_dev_v1.json`, GPU-trained, predicted on the CPU with `networks.predict_learner` (ctx3 path, the
  same token store). Its seed mean enters `members/members.parquet` as `lirf_mask` on every row, unrouted. The receipt adds the
  declaration and the 16 model hashes. On the ranking file: 743.6 s for the stage at 8 threads (45–52 s per network; the 48 earlier
  member predictions were reused from their per-learner files). A members table without `lirf_mask` builds no LIRF layer set.
- **Stack.** After v52m, each LIRF layer adds a delta against the shipped LIRF arithmetic on its own route of the recipe detail
  (v9, `ens_ctx_lirf`, `ens_v22_lirf_q`), through the frozen scorers' functions, and floors at −12 s:
  - **v53m** (w55, the declared rule's selection): `lirf_nonpop_unmatched` rows, v34's unmatched blend with v45's gate u600 on
    network' = 0.45 `ens_ctx_lirf` + 0.55 M, minus the same on `ens_ctx_lirf` (`tools/export_lirf_member.py`'s arithmetic).
  - **v54m** (w70_s750): the same rows in **dark LIRF months only**, read from the file's own scopes (LIRF non-population ADS-B match
    share below 5%); `tools/run_lirf_dark.values` at (0.70, 0.75) minus v53's (0.55, 0.35/0.60), gate u600. On the ranking file the
    dark month is July 2026 (34 matched of 15,549); January 2026 (79.9% matched) is covered.
  - **v55m** (mw40, `configs/lirf_combo_reread_v1.json`, never uploaded): `lirf_nonpop_matched` rows, v34's matched blend with the
    m600 gate on network' = 0.60 `ens_ctx_lirf` + 0.40 M minus the same on `ens_ctx_lirf` (`tools/run_lirf_combo.values`).
  Guards: a route row that is not LIRF, or that is pinned, or a missing member value on a route row stops the stack; no row outside the
  route may move. The selected candidates are read from each declaration's report and must be the shipped ones. Columns `v53m`,
  `v54m`, `v55m`, masks `*_lirf`; the stack record keeps each layer's route rows, gate rows (control / candidate), the dark-month
  shares and the base-minus-control check. Stack stage 16.9 s; each final file 0.5–0.6 s.
- **Dry run** (`runs/final_dryrun/run`, versions 953 / 954 / 955). `tools/final_compare_lirf.py` →
  [`dryrun_lirf_compare.json`](dryrun_lirf_compare.json):
  - Every earlier column (v30 … v46, v51m, v52m) equals the stack detail from before the LIRF layers exactly (max |Δ| 0.0;
    `runs/final_dryrun/before_lirf/stack_detail.parquet`).
  - The seed mean, runner CPU against the shipped GPU refit12 predictions, on the 26,499 LIRF non-population rows: RMS 0.106 s,
    p99 0.14 s, one row at 9.6 s.
  - Route and gate rows equal the exports' records: v53m 17,718 rows, gate 79 → 78; v54m 15,515 rows (July 2026), gate 72; v55m
    8,781 rows (January 2026 8,747, July 2026 34), gate 21 → 18. v52m holds the control value on every route row (max 1.8e-12 s).
  - No step moves a non-LIRF row, in the runner or in the shipped files. Each layer's step (runner vN − runner vN−1 against shipped
    vN − shipped vN−1) differs by at most 1 s (v53m, v54m) and 2 s (v55m).

| layer set | shipped model part | integers differing (max) | ±1 s | on LIRF non-population rows | runner file (never uploaded) |
|---|---|---:|---:|---:|---|
| v53m | `runs/adsb_ctx3_lirf_mask/export_dry/vigorous-whistle_v53.parquet` `1cb22c38…` | 6,305 (101 s) | 6,274 | 4,591 (max 4 s) | `v953` `729d48aa…` |
| v54m | `runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet` `b0fbbd19…` | 7,560 (101 s) | 7,529 | 5,846 (max 4 s) | `v954` `f3b1ae1a…` |
| v55m | `runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet` `266b8c93…` | 9,777 (101 s) | 9,745 | 8,063 (max 5 s) | `v955` `0d3dc265…` |

  Outside the LIRF routes the differences are v52m's (1,714 rows, including the 101 s LEMD u300 flip above). The new LIRF rows are
  rounding: each shipped export added a float delta to the previous step's **integer** file and rounded again, while the runner chains
  the floats and rounds once, so a route row can land one second apart per step (the largest new differences are 1 s; the member's
  CPU / GPU noise is about 0.03 s after its weights). The 4–5 s rows are the 30 LIRF rows on which v52m already differed.
  Runner SHA-256: v953 `729d48aacb60c0c92b941dc873d4f4c7ee3e09784bc3853a022157c17310c675`, v954
  `f3b1ae1a98c26d88b271fdb2c22367d0b0a4e96e35ee596cfbcc69ffe6667826`, v955 `0d3dc26597a59d95ab7693d972590dbead0497e26bb6ac530b0d5e93e27250e9`.
- Tests: `tests/test_final_lirf_members.py` (hand-computed v53m / v54m / v55m on the stack fixture's LIRF rows, a dark and a covered
  month, the 5% rule, no LIRF layer without the member, route guards) and the updated `tests/test_final_members.py`.
