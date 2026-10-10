# Lane V: the final-file pipeline on an Azure VM (8 October 2026)

Goal (`final_round_plan.md`): show that the final file can be produced on an Azure worker instead of this workstation, and time it.
First on the December 2025 stand-in (`runs/final_standin/dec2025/`); then, on the coordinator's word once the final dataset arrived
(`final_dataset.md`), **the real file `data/raw/final_ranking.parquet` through the runner's pre-stack stages on arcw0** (stack / final wait
for the user's fill-pin decision). Nothing was uploaded; nothing is under `runs/submissions`.

## What arcw0 needed and how it was synced

arcw0 (Standard_D16ds_v7, 16 vCPU, 62 GB RAM, 61 GB disk with ~29 GB free at the start) held the frozen phase bundles but only 7,070 of
this workstation's 37,294 files under `runs/ data/ src/ tools/ configs/ docs/`. A full copy (`runs/` 74 GB, `data/` 11 GB) does not fit,
so the read set was assembled:

| batch | what | files | GB |
|---|---|---:|---:|
| code + docs | every file of `src/ tools/ configs/ docs/` missing or different on arcw0 (working-tree bytes, CRLF kept) | 2,296 | 0.08 |
| models | the paths the runner's own dry-run receipts name (`runs/final_dryrun/run/stages/*.json`, `inventory.json`), narrowed to the twelve-month `refit` / `refit12` subtrees | 3,484 | 6.72 |
| nets / tokens / externals | the 32 own-points nets (`refit12`), `data/ctx/tokens_v1/manifest.json`, December 2025 surface days (+29-30 Nov, 1 Jan), IEM / OurAirports / EUROCONTROL / OPDI / match tables, the stand-in | 429 | 2.30 |
| leaves | the 36 twelve-month leaf refits under `runs/adsb_v23` (`configs/adsb_v23_v1.json`), the v5 weather fallback, `data/phase1/ceacb8a6...`, the vocabulary cache | 285 | 4.67 |
| chain tables | the early chain's development tables, as far as the chain tools' path literals name them inside Lane E's directories | 2,282 | 4.08 |
| final file | `final_ranking.parquet`, `final_submitting.parquet`, the 2026-01/02/06/07 surface days (238 files), February / June IEM inventory, the February / June adj2 boosters | 387 | 1.65 |

Manifest with every file's SHA-256: [`lane_V_synced.tsv`](lane_V_synced.tsv) (9,168 files, 19.5 GB). Not needed by the runner and not
sent: the Jan-Nov (eleven-month) refits of `runs/adsb_v15`-`adsb_v22` (4.8 GB; the leaves stage loads only the `adsb_v23` refits), the
3 GB `data/ctx/tokens_w256`, ranking-file runner roots. One input no receipt names was found by the run itself:
`runs/adsb_ctx/dev5/selection.json` (recipe stage).

**Check.** After the stand-in, every file present on both machines (12,283, outputs excluded) was hashed on both sides: 38 differ, and
none is read by the runner: logs; `data/phase1/58d621b245de3f5e12dd/manifest.json` (the known host-metadata difference of the Azure
original); docs / tools committed after the first sync (re-synced before any chain step); `runs/adsb_nn/features` (read only with
`--detail-ensembles`); `data/ctx/tokens_v1/*.npz` (zip metadata only: all 13 files' array contents equal the manifest's `content_sha256`,
and the runner reads only the manifest).

## Problems found and fixed

- **Overwritten without backup (my error).** The first code sync meant to back up arcw0's 18 differing files first; the file list was
  written with CRLF line ends on Windows, `tar` found no file, and the sync ran anyway (`;` instead of `&&`). All 18 previous versions were
  recovered exactly (12 from git history, 6 READMEs from campaign archives on arcw0) into `~/backup_laneV_code_20261008.tgz` on arcw0
  (SHA-256 `354b9e7d...`). Later lists are written with `newline="\n"` and each step chained with `&&`; the second sync's backup is
  `~/backup_laneV_code2_20261008.tgz`.
- **Tracing cost.** A first attempt traced file reads with `strace -f`; it stops every thread at every system call (LightGBM's OpenMP
  futexes), so the adj2 refit crawled (5.9 million context switches in 14 minutes); `--seccomp-bpf` still cost ~6,500 switches/s.
  Tracing was dropped for the hash check above.
- **Stand-in template.** My December template stored `TAXITIME_SEC_mvt` as float64; the official templates use an all-null int32, so
  `export_submission` refused it in the final stage. Fixed in `runs/final_standin/dec2025/template.parquet` (the old file kept as
  `template_float64_old.parquet`); the real `final_submitting.parquet` is int32.
- **arcw6's slow adj2 is not explained** by this lane: arcw0 shows the same CPU (Intel Xeon 6973P-C, 8 cores
  x 2 threads) and refits in ~8.6 min without tracing (concurrent with other work); arcw5 / arcw7 took 4-5 min.

## December 2025 stand-in (runner on arcw0)

| stage | s | | stage | s |
|---|---:|---|---|---:|
| inspect | 3 | | residual | 90 |
| refit-adj2 2025-12 | 516 | | recipe | 3 |
| features | 269 | | repair (nothing flagged) | 9 |
| adsb | 212 | | rematch | 27 |
| tokens | 4 | | members | 976 |
| v9 | 28 | | stack (`--no-fill-pins`) | 9 |
| leaves | 127 | | final | failed: template dtype (above) |
| networks | 892 | | early chain step 0 (key map) | 19 |

Sanity score of the stack detail against the held-back December labels (in-sample: December is inside every twelve-month refit, so this
checks plumbing, not accuracy): v30 130.74 / v35 128.86 / v46 127.63 / v52m 127.00 / **v55m 126.25 s**, every value finite, minimum -12.
The stand-in root was then deleted on arcw0 to make room (records mirrored: `runs/final_standin/dec2025/vm_records/`).

## The real final file on arcw0 (pre-stack stages)

Root `runs/final_blind/run_final4m` (arcw0), input `data/raw/final_ranking.parquet` (4b0692a7...) with `final_submitting.parquet`
(b3445d80...); `inspect`: 1,341,197 movements, 670,790 departures (2026-01 / 02 / 06 / 07), template order exact, no label. Weather:
the frozen objects (January / July) + `data/external/iem_final/prestage_2026/inventory.json` (February / June). adj2: the saved January /
July boosters + the pre-staged February / June refits copied into `residual/adj2/`. **The EOBT_1 detector flags LFPG 2026-06 (share 26.7%)
and LFPG 2026-07 (34.5%)** against a 10.2% threshold, so `repair` reruns the chain features -> recipe on the edited file (as declared).

**Status: PAUSED at the user's request (8 October, ~14:00 UTC).** The runner stages features -> recipe are complete; `repair` stopped in
its sub-run (below); `rematch`, `members` (arcw0), `stack` and `final` were not started. Every output is mirrored to this workstation
with matching SHA-256 (`runs/final_blind/mirror_main.sha`, 125 files; `runs/final_blind/mirror_repair.sha`, 139 files; root 3.1 GB at
`runs/final_blind/run_final4m/`). arcw0 and arcw3 are deallocated; their disks keep the same files.

| stage (arcw0, 16 threads) | s | note |
|---|---:|---|
| features | 202 | |
| adsb | 1,295 | matches for the four months by 11:27, then ~15 min single-threaded (learner / v19 columns) |
| tokens | 10 | |
| v9 | 92 | |
| leaves | 438 | |
| networks | 3,730 | 64 nets, 58 s each (670,790 departures) |
| residual | 818 | adj2: saved January / July, pre-staged February / June |
| recipe | 6 | `recipe/detail.parquet` (v30 detail of the final file) |
| repair: detector + edit | — | LFPG 2026-06 / 2026-07 flagged; 13,034 departures edited (`repair/edited_ids.parquet`); edited input `repair/input/final.parquet` |
| repair sub-run features / adsb / tokens / v9 / leaves / networks | 199 / 1,385 / 6 / 86 / 429 / 2,771 | complete, receipts in `repair/run/stages/` |
| repair sub-run residual | failed | **"No adj2 booster for 2026-02"**: the sub-run looks in `repair/run/residual/adj2/`; the February / June boosters were only in the root's `residual/adj2/` |
| members (arcw3, in parallel) | 3,624 | all 64 member nets predicted on arcw3 (same CPU type, same torch / numpy / pandas) from arcw0's hash-checked inputs; mirrored to `runs/final_blind/members_arcw3/` (per-net files, `members.parquet`, `ensembles.parquet`, its receipt); **not** placed into arcw0's root |

**Finding for the procedure (runner.md step 3):** when the detector flags an airport-month in a file whose months need refitted adj2
boosters, the boosters must also be copied into `<root>/repair/run/residual/adj2/` (the sub-run is a separate root). Neither the
stand-in (nothing flagged) nor the ranking file (saved January / July boosters) could show this.

### Resume point (exact)

    R=runs/final_blind/run_final4m          # on arcw0 (start it) or on this workstation (the mirror is complete)
    mkdir -p $R/repair/run/residual/adj2 && cp runs/final_prestage/adj2/residual_adj2_2026-0[26].* $R/repair/run/residual/adj2/
    OPTS="--threads 16 --workers 8 --weather-inventory data/external/iem_final/prestage_2026/inventory.json --vocab-cache runs/final_dryrun/run/leaves/vocabularies.json --layers v55m"
    .venv/bin/python -X utf8 -B tools/final_predict.py run --root $R --stages repair  $OPTS   # sub-run: residual + recipe only (~14 min), then v31's edit
    .venv/bin/python -X utf8 -B tools/final_predict.py run --root $R --stages rematch $OPTS   # ~1 min
    # members (~60 min on a D16), or reuse arcw3's predictions: copy runs/final_blind/members_arcw3/members/<ensemble>/*.parquet into
    # $R/members/<ensemble>/ (the stage reads existing per-net files and only averages them; recompute one net first as a cross-VM check)
    .venv/bin/python -X utf8 -B tools/final_predict.py run --root $R --stages members $OPTS
    # after the user's fill-pin decision:
    .venv/bin/python -X utf8 -B tools/final_predict.py run --root $R --stages stack final --no-fill-pins --version N $OPTS
    # then lane_E.md 'For the final file' (key map, own member, early chain) and lane_L.md (late chain) on $R's outputs

Expected remaining wall time on a D16: repair ~15 min, rematch ~1 min, members ~60 min (or seconds with arcw3's files), stack / final
~1 min, then the own member (~35 min for 670,790 departures) and the early / late chains.

## Command sequence for the real final file on a VM

    # 0. inputs on the VM (this lane's manifest), the final files, the months' surface days, February / June weather, adj2 boosters
    R=runs/final_blind/run_final4m
    .venv/bin/python -X utf8 -B tools/final_predict.py inspect --root $R --raw data/raw/final_ranking.parquet --template data/raw/final_submitting.parquet
    .venv/bin/python -X utf8 -B tools/final_predict.py external --root $R          # must list nothing missing
    mkdir -p $R/residual/adj2 $R/repair/run/residual/adj2
    cp runs/final_prestage/adj2/residual_adj2_2026-0[26].* $R/residual/adj2/
    cp runs/final_prestage/adj2/residual_adj2_2026-0[26].* $R/repair/run/residual/adj2/   # the repair sub-run is its own root
    for s in features adsb tokens v9 leaves networks residual recipe repair rematch members; do
      .venv/bin/python -X utf8 -B tools/final_predict.py run --root $R --stages $s --threads 16 --workers 8 \
        --weather-inventory data/external/iem_final/prestage_2026/inventory.json \
        --vocab-cache runs/final_dryrun/run/leaves/vocabularies.json --layers v55m || break
    done
    # stack + final once the fill-pin decision is made (add --no-fill-pins for decision 2 of final_round_plan.md):
    .venv/bin/python -X utf8 -B tools/final_predict.py run --root $R --stages stack final --layers v55m --no-fill-pins --version N --threads 16
    # then lane_E.md 'For the final file' (key map, own member, early chain) and lane_L.md (late chain) on $R's outputs

## Resumed on 10 October

The final data were confirmed definitive, so the run resumed on arcw0
(started 05:44 UTC; the root disk had 3.0 GB free).

| step | where | s | result |
|---|---|---:|---|
| repair (boosters copied into `repair/run/residual/adj2` first) | arcw0 | 817 | LFPG 2026-06 / 2026-07 flagged as on 8 October, 13,034 departures edited |
| rematch | arcw0 | 111 | |
| members | arcw0 | 73 | 63 of 64 per-net files uploaded from arcw3's mirror (hash-checked); `dl32_s601` recomputed on arcw0 as the cross-VM check: **byte-identical** to arcw3's (`d612007d...`), and `ensembles.parquet` / `members.parquet` equal arcw3's (`927c8736...` / `e591ffaf...`) |
| stack + final (`--layers v55m --no-fill-pins --version 9001`) | arcw0 | 18 | `R/final/vigorous-whistle_v9001.parquet` `cdb81e08...`: 670,790 rows, min -12, median 946 s, no null |
| key map (`run_rematch_nonm.py --matches`) | arcw0 | | 3,707 no-NM rows at EDDF / EHAM / LSZH, 455 qualifying |
| own member features / predict (32 nets, ~60 s each) / export-v19own | arcw0 | | 670,790 rows, 487,175 with an inbound |

Inputs added for the chains: the five OPDI windows of February / June 2026 (`docs/opdi_stack/sources.json`), and the early / late
chains' 2025 development artefacts that arcw0 lacked (595 + 67 files, 2.66 + 0.74 GB). Those artefacts sit on the VM's local NVMe
(`/mnt/scratch/inputs`, formatted for this run, wiped at deallocation) and are symlinked at their repository paths; every file is
hash-checked against this workstation. Work directories of the chains are on `/mnt/scratch` too, and are mirrored here before
deallocation. One mistake on the way: a first upload went to the root disk and filled it (the size was under-counted: `xargs` split
`du` into batches and only the last total was read); it was stopped at 178 MB free and exactly the files it had added were removed
(557, all absent before), before the own-member job wrote its remaining files. The own-member job was not affected.

**export-v19own** first refused the file: the runner's `adsb/v19.parquet` holds the final file's 670,790 departures in another order
(22.8% of positions equal; the ranking file's happened to match the template). The runner joins v19 by ID (`runner.py:110`), so the
model part is unaffected; the export now reorders after checking the ID sets are identical and unique (26906b5; a no-op on the
ranking file). Monthly files: January 152,719 / February 144,158 / June 181,791 / July 192,122 rows.

**Early chain on arcw0 stopped at step 2**: `run_rematch_nonm.py --ranking-base` reads `runs/rematch_nonm/matches_2025.parquet`, which
the Lane V sync did not include. Step 1 had completed (model part `aad72fb7...`). The full set of directories the early-chain tools name
holds 2,220 files (12.7 GB) absent on arcw0, most of them not read by the chain; rather than guessing the read set one failure at a time,
**the early chain runs on this workstation**, where Lane E proved it end to end (the compute rule's exception: it cannot run remotely
without that read set). arcw0's new runner outputs were mirrored here first (`runs/final_blind/mirror_20261010.sha`, 265 files, all
hashes equal), its scratch outputs and logs copied to `runs/final_blind/arcw0_scratch_20261010/` (hash-checked), its 662 scratch
symlinks removed (no dangling link left), and arcw0 deallocated (all eight workers deallocated at 07:03 UTC). Step 0 locally gives
the same key map and no-NM matches as arcw0, byte for byte (`835e98d0...`, `80d4d713...`).

**Sanity check of the model part against the shipped one** (`tools/final_vs_shipped.py`, label-free; final `v9001` against the shipped
`runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet` on the 344,841 January / July rows the two files share): median |difference|
5 s, p90 15 s, p99 41 s; outside LIRF the median is 4.3 s, the size of the +-12 s take-off jitter. 49 rows move by more than 300 s at
the recipe (v30) already: 32 LIRF `fixed`-scope rows (the long-interval population, whose mixture bands are anchored on the stand's
previous arrival, itself jittered) and 17 rows elsewhere (ADS-B match / scope flips: 108 rows change scope); every recipe component
moves together on them, an input flip rather than a component fault. v40's nine fill pins are off in the final file as decided
(pinned rows: 9 in the ranking dry run, 0 here).
