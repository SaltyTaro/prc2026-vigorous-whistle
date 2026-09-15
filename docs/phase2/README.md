# Phase 2: CPU baselines and first official submission

Completed **13 September 2026**. The first submission,
`vigorous-whistle_v1.parquet`, scored **436.8976 seconds RMSE**, using all
**344,841 departures**. The organizer's result and persistence receipts both
report success.

The leaderboard snapshot retrieved at **14:52 UTC on 13 September** places
the team **97th of 131 scored teams**. Third place is **261.9189 seconds**,
requiring about a 40.1% RMSE reduction from this baseline. These are dated
standings, not a forecast of the final qualifying score.

Evidence: [official API result](official_result.json),
[organizer result](vigorous-whistle_v1.parquet_result.json),
[persistence receipt](vigorous-whistle_v1.parquet_persist.json), and
[submission/model provenance](submission_v1.json).

## Fixed comparison

Both libraries use the same 25 predictors and phase-1 bundle
`58d621b245de3f5e12dd`. LightGBM categorical vocabularies are fitted only on
fitting rows; unseen values become missing. CatBoost uses native categorical
processing. Movement and flight IDs remain outside the predictors.

October is used for early stopping in both development folds. Scoring
predictions are made only after choosing the iteration using October.
Every scoring label is retained, including negative and extreme taxi times.
December is excluded from development, selection, and the submission refit.

[baselines_v1.json](../../configs/baselines_v1.json) was written before
examining the model results. Selection requires improvement over airport means
on both development folds, then minimizes the equally weighted mean of the
two RMSE ratios to that reference.

| Model | January/July 2025 RMSE | November 2025 RMSE |
| --- | ---: | ---: |
| Airport mean, phase-1 reference | 660.0889 | 504.6090 |
| **LightGBM 4.7.0** | **490.6022** | **341.2574** |
| CatBoost 1.2.10 | 494.4830 | 349.1857 |

The ranking-month-weighted seasonal RMSE is 490.9448 for LightGBM and 494.7988
for CatBoost. The actual 2026 official result falls between the two ordinary
development-period results; local scores are not official scores.

LightGBM wins the observed comparison on both folds. The difference between
the libraries is uncertain: paired day resampling gives 95% intervals for
LightGBM minus CatBoost of **[-20.80, 12.99] seconds** seasonally and
**[-50.34, 37.69] seconds** in November. These are descriptive intervals on
reused development data, not an independent final test.

Complete metrics, source identities, and the 2,000-replicate day-resampling
results are in [development_results.json](development_results.json). Keep both
libraries available for later feature experiments.

## Selected recipe and submission

LightGBM selected 348 trees on the seasonal fold and 209 on the forward fold.
The predeclared refit rule rounds their median to **278 trees**. This refit
uses **1,919,370 departures from January–November 2025**, without early
stopping. December's 165,677 rows remain reserved.

The first recipe uses raw squared-error training, 63 leaves, learning rate
0.04, at least 100 observations per leaf, L2 regularization 10, feature
sampling 0.9, row sampling 0.8, and seed 20260913. Exact parameters and
categorical settings are recorded in the configuration and run artifacts.

The export preserves `MVT_ID_mvt: double` and `TAXITIME_SEC_mvt: int32`,
rounding to the nearest second with ties to even and no clipping. Its largest
rounding change is below 0.5 seconds. It contains 32 negative predictions,
with a minimum of -116 seconds and maximum of 37,456; all rows were scored.
These values are retained for an exact account of this baseline.

- Filename: `vigorous-whistle_v1.parquet`.
- File size: **1,336,139 bytes**.
- SHA-256: `ba074016e32978fd8817aac138a6afaf4b3fe4d463a87b1a00f330872fef7e6e`.
- Bucket: `prc-2026-vigorous-whistle` at the organizer's OpenSky endpoint.
- Uploaded: **14:49:05 UTC**, 13 September.
- Official persistence time: **14:49:41 UTC**, 13 September.
- Official submission ID: `41f5d444-cf7e-48d7-8855-97de8fbc4318`.

Only one submission was made. Before upload, the tool rechecked the live
ranking/template versions, validated the export hash/schema/IDs, checked
capacity and recent submissions, and used a conditional write that cannot
overwrite a version. Upload and scoring receipts are separate.

## Compute and checks

Full-data training ran only on Azure `arcw0`, a `Standard_D16ds_v7` with
16 vCPUs and about 63 GiB usable RAM. The other seven VMs remain deallocated.

| Run | Wall time including loading/scoring | Sampled peak process RSS |
| --- | ---: | ---: |
| LightGBM seasonal | 16.92 s | 1.19 GiB |
| LightGBM forward | 14.16 s | 1.36 GiB |
| CatBoost seasonal | 93.39 s | 3.56 GiB |
| CatBoost forward | 116.95 s | 3.95 GiB |
| LightGBM refit and ranking predictions | 20.82 s | 1.56 GiB |

These are single-run measurements with 50 ms RSS sampling, excluding package
installation and transfers. CatBoost selected 1,596 and 1,599 trees, close to
its 1,600-tree cap; this is a bounded baseline, not a conclusion about its best
possible performance. Larger feature sets can change resource requirements.

**48 tests pass on Windows and Azure.** They cover the original validation
contract plus native training, saved-model prediction reproduction, fit-only
vocabularies, target alignment, audit exclusion, exact submission IDs/types,
invalid predictions, rounding overflow, immutable versions, and upload limits.
One pandas deprecation warning concerns unseen categories in a future release;
the tested environment pins pandas 3.0.0.

Full-data models were reloaded and verified to reproduce a prediction sample
within 1e-9 seconds. All saved development predictions were rescored locally,
reproducing Azure RMSE within floating-point tolerance. Training itself was
not rerun on Windows.

## Reproduce

The original phase-1 Python package remains unchanged. Model code is layered
in `src/prc2026_models/` and submission validation in `src/prc2026_submission/`,
preserving the feature bundle's identity.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase2.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe tools/package_phase2.py
scp infra/azure/.local/phase2-code.tar arcw0:/home/azureuser/prc-flight-2026/incoming/
ssh arcw0 'cd /home/azureuser/prc-flight-2026 && sha256sum incoming/phase2-code.tar'
```

Compare that hash to `infra/azure/.local/phase2-code.json`, then run:

```powershell
ssh arcw0 'cd /home/azureuser/prc-flight-2026 && tar -xf incoming/phase2-code.tar && bash infra/azure/phase2.sh'
```

The runner installs pinned libraries, runs tests, and trains four jobs
sequentially. It does not start extra VMs or upload submissions. Collect into
a fresh local directory to avoid `scp` nesting:

```powershell
scp -r arcw0:/home/azureuser/prc-flight-2026/runs/phase2 runs/azure_arcw0_phase2_reproduction
.\.venv\Scripts\python.exe tools/select_phase2.py --runs runs/azure_arcw0_phase2_reproduction
scp runs/phase2/selection.json arcw0:/home/azureuser/prc-flight-2026/runs/phase2/selection.json
ssh arcw0 'cd /home/azureuser/prc-flight-2026 && OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 .venv/bin/python -u tools/run_phase2.py refit --selection runs/phase2/selection.json'
```

Collect the new refit directory printed by the command. Use
`tools/build_submission.py --refit-run PATH --version N` to build a local
version. Existing versions cannot be overwritten. Submission is separate:

```powershell
.\.venv\Scripts\python.exe tools/prc_submit.py upload --file PATH --expected-sha256 SHA256
.\.venv\Scripts\python.exe tools/prc_submit.py status --filename vigorous-whistle_vN.parquet
```

Credentials stay in the local `.env`; none were copied to Azure. Do not
re-upload v1 to reproduce training. Its accepted file and receipts are preserved.

Development runs save models, encoders, iteration histories, feature importance,
full scoring predictions, metrics, configuration, and hashes/resource records.
The refit also saves the selected recipe and ranking predictions. Originals
are on arcw0 under `runs/phase2/`; local copies are in `runs/azure_arcw0_phase2/`.

## Phase 3 priorities

- **Airport and traffic context:** test airport-scoped runway/stand combinations,
  schedule reliability, cyclic time features, and arrival/departure traffic
  windows using the preserved target-masked context.
- **Concentrated errors:** LIRF RMSE is 1,360.25 seconds seasonally and 906.83
  in November; LFPG is 602.37 and 364.68. Labels above three hours contribute
  56.1% of LightGBM seasonal squared error and 41.4% in November. Study their
  patterns while retaining them in scoring.
- **NM timing and missingness:** scheduled/EOBT deltas are influential.
  `AOBT_3_flt`, `LOBT_flt`, and `ARVT_3_flt` remain excluded pending the
  original organizer guidance discussed in phase 1. Other feature work can
  proceed while that interpretation is resolved.

The [airport errors](airport_errors.csv) and saved predictions support paired
comparisons. Keep December reserved and check every change on both development
folds. Improve features before expanding the parameter search or VM count.
