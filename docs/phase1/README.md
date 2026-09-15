# Phase 1: validation and experiment foundation

Completed **13 September 2026** for `vigorous-whistle`, ahead of the proposed
14–15 September window. The same full-data pipeline ran locally and on Azure
`arcw0`. All **43 materialized feature, label, context, catalog, and split files
have identical SHA-256 hashes across the two machines**. Both development
prediction sets match exactly.

The user allocated `arcw0`–`arcw7` and authorized starting **only `arcw0`**.
The pool consists of `Standard_D16ds_v7` machines in `eastus`. The pilot observed
16 logical CPUs and 62.79 GiB of usable RAM. Its isolated project directory is
`/home/azureuser/prc-flight-2026`. See the [allocation](../../infra/azure/allocation.json)
and [Azure run instructions](../../infra/azure/README.md).

## Frozen validation

The executable specification is [validation_v1.json](../../configs/validation_v1.json).
Partitions use each departure's **actual UTC movement month**, not its filename.
Six movements cross a filename boundary: two departures and four arrivals.
The departure exceptions fall in February and August. Their full source list is
in the [validation report](validation_report.json).

| Fold | Fitting months in 2025 | Early stopping | Scoring | Fit rows | Score rows |
| --- | --- | --- | --- | ---: | ---: |
| Seasonal development | Feb–Jun, Aug–Sep | October | January and July | 1,226,945 | 344,419 |
| Forward development | January–September | October | November | 1,571,364 | 162,332 |
| Final audit, sealed | January–October | November | December | 1,757,038 | 165,677 |

October has 185,674 departure rows; November has 162,332. November is unused
in the seasonal fold. December is reserved from both development folds.
The seasonal fold measures transfer to the ranking file's seasons; fitting on
later months means it is **not** a forward-time evaluation.

Nonmissing `FLIGHT_ID_mvt` values are kept separate across fit, early-stopping,
and scoring partitions. Scoring rows take priority: any conflicting training or
early-stopping row is purged. Null flight IDs are independent movements. The
actual data needs no purges and has zero flight-group overlap between any
partitions, including the reserved audit. Synthetic tests exercise the conflict
cases so the behavior is defined if a future dataset changes.

All scoring populations have saved, order-independent ID hashes. Every label
is retained in evaluation, including negative, zero, and extreme values.
The final-audit CLI path rejects scoring in this phase. December's labels were
included in the earlier descriptive audit and in integrity checks; they have
not been used to select or score a model.

## Feature and data contract

All 14 raw files are verified against the original download hashes before
preparation. Original files remain unchanged. The feature bundle is:

`data/phase1/58d621b245de3f5e12dd`

Its full identity, input hashes, code hashes, fold assignments, and artifact
receipts are in [validation_report.json](validation_report.json). The 43
materialized artifacts occupy **190,879,826 bytes**.

The transform first masks **every departure's** target and airport off-block
time, including training departures. It then applies the same functions to
training and ranking data. Labels are saved separately. Arrival movements and
their supplied taxi-in/block-time context are preserved in separate context
files; no departure target is present in that context.

The initial contract contains **25 predictors**:

- 13 categorical fields: airport, destination, runway, stand, aircraft type,
  movement rule, NM origin/destination/aircraft type, market segment, flight
  type, wake category, and operator.
- 12 numeric fields: UTC hour/weekday/month/minute of day, airport-local
  hour/weekday, movement minus scheduled/EOBT/IOBT time, planned flight
  duration, NM-record missingness, and NM-origin agreement.

UTC timestamps are retained; the ten airports have explicit IANA timezone
mappings, with daylight-saving and midnight behavior tested. Missing categories
use an explicit token; missing numeric values remain missing.

Movement and flight IDs are alignment/grouping keys. They are not predictors.
The initial contract also excludes `AOBT_3_flt`, `LOBT_flt`, and `ARVT_3_flt`.
This is a scoped implementation choice while checking the original organizer
announcements, **not a claim that the public rules ban those supplied fields**.
The official site was refreshed again on 13 September: no field-specific
restriction was found in the retrieved public pages.

Ranking features preserve all **344,841 template IDs in exact template order**.
Its departure mix is 152,719 January and 192,122 July rows. Seasonal reports
include an additional RMSE weighted to that observed mix; the ordinary complete
validation RMSE remains the main reported local metric.

## Pilot measurements

The arithmetic airport mean is fitted only on the fitting partition, with a
fitting-set global mean as fallback. It exercises feature loading, alignment,
model/prediction persistence, and scoring. It does not use early stopping.
These are **local plumbing benchmarks**, not official results or competitive
LightGBM/CatBoost baselines.

| Work | Local Windows time | Azure arcw0 time | Azure sampled peak RSS |
| --- | ---: | ---: | ---: |
| Hash checks, feature/context preparation, frozen splits | 119.28 s | 67.74 s | 1.19 GiB |
| Seasonal airport-mean fit and complete scoring | 7.19 s | 3.88 s | 1.27 GiB |
| Forward airport-mean fit and complete scoring | 5.52 s | 2.96 s | 1.41 GiB |

RSS is sampled every 50 ms and can miss shorter allocation spikes. Timings
exclude initial package installation and file transfer. These measurements
cover the 25-feature pipeline and arithmetic means; they do not establish
LightGBM/CatBoost training time, future traffic-feature memory, or GPU speed.
Azure stages ran sequentially in one Python process; local evaluation commands
used separate processes.

| Evaluation | Rows | RMSE in seconds |
| --- | ---: | ---: |
| January/July seasonal | 344,419 | 660.0889 |
| Seasonal, weighted to ranking month mix | 344,419 | 660.5243 |
| November forward | 162,332 | 504.6090 |
| December final audit | 165,677 | **Not scored** |

The two machines produced identical predictions. RMSE agrees to at least
1e-12 relative tolerance; minor SSE accumulation differences reflect floating
point reductions.

Extreme labels matter to the real objective: 60 labels over three hours
account for 57.5% of the seasonal squared error; six account for 41.7% of the
forward squared error. They remain included. Model comparisons should inspect
these contributions alongside full-population RMSE.

The [benchmark evidence](benchmark_summary.json) contains exact timings,
sampled memory, package versions, code/input identities, per-airport/month/day
and target-band metrics, missing-NM slices, and cross-platform checks.
Full predictions and model files are kept under `runs/phase1/` locally,
`runs/azure_arcw0_phase1/` for the collected Azure runs, and the isolated
project's `runs/phase1/` on `arcw0`.

## Reproduce locally

Use Python 3.12 or newer. The Azure run used 3.12.3; the existing local
environment used 3.14.2. Model-library compatibility must be checked separately
in phase 2.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
.\.venv\Scripts\python.exe -m pytest tests/test_artifacts.py tests/test_features.py tests/test_validation.py
.\.venv\Scripts\python.exe tools/run_phase1.py prepare --label local-windows
```

Use the bundle path printed by `prepare`. Code/configuration, input, or transform
dependency changes produce a new bundle identity. A complete existing bundle is
hash-verified before reuse; incomplete builds are not accepted.

```powershell
.\.venv\Scripts\python.exe tools/run_phase1.py benchmark --bundle data/phase1/58d621b245de3f5e12dd --fold seasonal
.\.venv\Scripts\python.exe tools/run_phase1.py benchmark --bundle data/phase1/58d621b245de3f5e12dd --fold forward
.\.venv\Scripts\python.exe tools/run_phase1.py verify --bundle data/phase1/58d621b245de3f5e12dd
```

**26 tests passed on both platforms.** They cover target masking and feature
invariance, arrival context, timezone changes, invalid IDs/airports, date/group
splits, audit sealing, complete-set metric alignment, nonfinite values, outlier
retention, corruption detection, and Windows/Linux resource logging.

Each benchmark saves `model.json`, `predictions.parquet`, `metrics.json`, and
`run.json`. The latter records parameters, exact code identity, bundle identity,
environment, resource measurements, and output hashes.

After collecting Azure results, regenerate the portable reports with:

```powershell
.\.venv\Scripts\python.exe tools/summarize_phase1.py --local-bundle data/phase1/58d621b245de3f5e12dd
```

## Next phase

Phase 2 builds LightGBM and CatBoost baselines against these frozen development
folds, checks feature availability and complete submission formatting, and
establishes the first official score. No submission was generated or uploaded
in phase 1.

One CPU VM is ample for the measured foundation pipeline. Before a large
experiment campaign, allocate artifact storage: the pilot root filesystem
has about 10.26 GiB free after preparation. Two 440 GiB NVMe devices were visible
but unmounted and were left untouched. The remaining seven VMs stay deallocated
under the current instruction.
