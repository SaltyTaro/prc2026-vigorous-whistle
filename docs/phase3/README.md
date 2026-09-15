# Phase 3: airport, timing, and traffic features

Feature experiments completed 13 September 2026 on the existing Azure `arcw0`
worker. **Timing plus recent traffic** is the selected combination: **475.1669
seconds seasonal RMSE** and **311.5701 seconds forward RMSE**. These improve
Phase 2 LightGBM by **3.15%** and **8.70%**, respectively. December remains
reserved. **v2 scored 421.3047 seconds RMSE on all 344,841 ranking rows**, a
**3.57% improvement** over v1's official 436.8976.

## Official submission

`vigorous-whistle_v2.parquet` was uploaded at **16:40:23 UTC on 13 September**
and persisted by the organizer at **16:40:59 UTC**. The official API confirms
**421.3047 RMSE** and **344,841 used pairs**. The version is immutable and must
not be re-uploaded or overwritten.

The full leaderboard snapshot retrieved at **16:44:50-16:45:08 UTC** places
the team **96th of 131 scored teams**, across 1,190 successful submissions.
Third place is **261.9189**, leaving a further **37.83% RMSE reduction** to
that dated cutoff. Phase 3 improves the model; the top-three objective remains
substantial work for later phases.

- File size: **1,338,296 bytes**.
- SHA-256: `98efb0ff6fab73afa04763db7b7e19d81cbdfa5ed808e1cd4257d409c1fe9724`.
- Submission ID: `22ce5a81-e043-4e77-b27d-93667f69d7e5`.
- Export: exact template ID order, `MVT_ID_mvt: double`, `TAXITIME_SEC_mvt: int32`;
  nearest-second rounding with ties to even and no clipping.
- Prediction range: -834 to 42,903 seconds; 43 negative predictions were retained
  under the declared no-clipping policy. All rows were accepted and scored.
- Daily use: two submissions on 13 September, including v1. Recheck the live
  bucket before a later version; no additional upload was made for testing.

Evidence: [official result](official_result.json),
[submission/model/verification receipt](submission_v2.json), and
[full leaderboard snapshot summary](leaderboard_snapshot.json).

## Results

| Added families | Predictors | January/July RMSE | November RMSE | Improves both vs v1 model |
| --- | ---: | ---: | ---: | --- |
| Phase-2 reference | 25 | 490.6022 | 341.2574 | Reference |
| Layout | 44 | 499.8328 | 345.6371 | No |
| Timing | 54 | 483.3747 | 318.9268 | Yes |
| Traffic | 57 | 486.5024 | 326.9144 | Yes |
| Layout + timing | 73 | 494.9221 | 326.7707 | No |
| Layout + timing + traffic | 105 | 489.5724 | 316.9898 | Yes |
| All four families | 116 | 490.9247 | 317.7757 | No |
| **Timing + traffic** | **86** | **475.1669** | **311.5701** | **Selected** |
| Timing + traffic + completed arrivals | 97 | 481.4396 | 313.1586 | Yes |

The selected model improves RMSE at **all ten airports on both folds**.
Its ranking-month-weighted seasonal RMSE is **475.5184**. Squared error is
lower on 48 of 62 seasonal days and all 30 November days. The paired
2,000-replicate day-bootstrap intervals for candidate minus v1 RMSE are
**[-25.04, -6.13] seconds** seasonally and **[-41.17, -17.23] seconds** in
November. These describe reused development data and are not final-audit
confidence claims.

Full results, including every candidate and the adaptive batch declaration,
are in [development_results.json](development_results.json). Airport-level
metrics are also available in [airport_errors.csv](airport_errors.csv).

## Selected refit

The two October early-stopping runs select 593 and 331 trees. The declared
median rule selects **462 trees** for refitting on **1,919,370 January-November
departures**. The refit produces all **344,841 ranking predictions**, without
target filtering or prediction clipping. Model parameters remain those of
the Phase 2 LightGBM baseline.

- Recipe: `timing_traffic` in `configs/phase3_followup_v1.json`.
- Feature bundle: `410850c6cdc70bd5b987b93770a40b09b0e30c6acb4f4f28cbc4f86b05f12ae8`.
- Refit: `20260913T161326150055Z_refit_timing_traffic_lightgbm`.
- Local artifacts: `runs/azure_arcw0_phase3/`; original Azure artifacts:
  `/home/azureuser/prc-flight-2026/runs/phase3/`.

## What remains difficult

LIRF still has RMSE **1,323.14** seasonally and **829.75** in November; LFPG
has **593.12** and **316.75**. Labels above three hours account for **56.45%**
and **39.79%** of the selected model's squared error, despite comprising only
60 and 6 scoring rows. Missing NM records have RMSE **2,997.90** and
**2,292.76** seconds. Phase 4 should address these records while preserving
complete-set scoring, then test airport-specific models. Increasing the VM
count is not the limiting factor for this feature batch.

## Experiment design

The configuration in [phase3_v1.json](../../configs/phase3_v1.json) declares
six combinations before their results are inspected. They keep the Phase 2
LightGBM parameters, seed, fitting IDs, October early-stopping IDs, and scoring
IDs. All scoring labels are retained, including negative and extreme values.
There is no prediction clipping or label transformation.

The four additional feature families are:

- **Layout (19 predictors):** airport-scoped runway, stand, stand/runway,
  stand-prefix, operator, aircraft, destination, and wake-category combinations;
  parsed runway number/side and stand number/suffix; flight designator prefix.
  Stand areas are code prefixes or numeric hundreds, not verified terminal
  geometry. Full flight/movement identifiers are not predictors.
- **Timing (29 predictors):** local clock and weekday cycles, annual cycle,
  UTC offset with DST, scheduled/initial flight-plan clocks, EOBT/IOBT/schedule
  differences, planned duration, missingness, and field-agreement flags.
- **Traffic (32 predictors):** preceding airport arrivals/departures over
  5/15/30/60 minutes, runway traffic over 15/60 minutes, previous-arrival stand
  context, headways, and preceding movement schedule-delay means/deviations.
- **Completed arrivals (11 predictors):** preceding airport/runway arrival
  taxi-in counts, means, deviations, and gaps, available only after in-block.

These extend the original 25 predictors to 116. The six ablations are layout,
timing, traffic, layout+timing, layout+timing+traffic, and all four families.
After timing and traffic individually improved both periods while layout
regressed, two follow-ups were declared in
[phase3_followup_v1.json](../../configs/phase3_followup_v1.json): timing+traffic
and timing+traffic+completed-arrival statistics. This is an adaptive second
batch using the same development sets; its results are not independent tests.
Selection requires improvement over Phase 2 LightGBM on both complete scoring
sets and minimizes the mean of the two RMSE ratios. The fixed refit tree count
is the rounded median of the two October-selected counts.

## Availability and leakage controls

Row-feature readers project an explicit list of input columns. Departure
`TAXITIME_SEC_mvt`, airport `BLOCK_TIME_UTC_mvt`, `AOBT_3_flt`, `LOBT_flt`, and
`ARVT_3_flt` never enter the row-feature transform. Flight IDs are not predictors.
The latter three clocks remain excluded pending original organizer guidance;
this does not assert an official ban.

Traffic windows are **[query time minus window, query time)**, using UTC
movement timestamps. Events exactly at the query time, including the query
departure, are excluded. Counts contain only earlier observed events. Last-event
gaps above six hours are missing. Finite schedule-delay values determine delay
statistics; observed-event counts include rows with missing delay values.

Arrival taxi-in is explicitly separated from departure targets in the phase-1
context. It is used only for arrivals with finite supplied taxi-in and an
in-block timestamp at or after landing. Its availability timestamp is in-block,
strictly before the departure query. A future in-block time cannot contribute
an arrival duration merely because landing already happened.

Traffic uses supplied unlabeled departure context and supplied arrival context
from all allowed development months, including held-out movements. This matches
the retrospective ranking setting, where all movement records are supplied.
It is not a live pre-departure forecasting simulation. All event statistics
remain strictly past-facing relative to the query movement time.

Context is filtered by actual UTC calendar month, including the November
arrival stored in the December file. December departures and December event
context are excluded from feature materialization; December labels are never
scored or fitted. No external data is used in this phase.

## Reproduction

Implementation is in `src/prc2026_phase3/`. The original phase-1 and phase-2
Python packages and their model/feature identities remain unchanged.

On the prepared arcw0 project, use a fresh run directory for reproduction so
the selector sees exactly one run per declared variant/fold:

```bash
bash infra/azure/phase3.sh runs/phase3_reproduction
bash infra/azure/phase3_followup.sh runs/phase3_reproduction runs/phase3_reproduction/report
.venv/bin/python tools/run_phase3.py --config configs/phase3_followup_v1.json --output runs/phase3_reproduction refit --selection runs/phase3_reproduction/selection.json
.venv/bin/python tools/verify_phase3.py --runs runs/phase3_reproduction --output runs/phase3_reproduction/verification.json
```

The runner checks pinned dependencies, runs the substantive test suite,
builds an immutable feature bundle, and trains the twelve initial jobs one at
a time. The follow-up runner adds four jobs and performs selection across all
sixteen results. Models, fit-only encoders, predictions, full metrics, iteration histories,
feature importance, hashes, and resource records are saved under `runs/phase3/`.
The selector independently rescores every saved prediction set and reports
paired day-bootstrap intervals. It writes an eligible refit recipe only when
the declared selection rule is satisfied. Submission is a separate step.

## Checks and portability

**61 tests pass on Windows and Azure.** They cover both earlier phases plus
feature masking, precise traffic boundaries, tied timestamps, empty/gapped
history, airport isolation, arrival completion availability, reserved-period
guards, fit-only vocabularies, and native extended-model persistence.
The one warning comes from the unchanged Phase 2 encoder's pandas deprecation.

The full build asserts that all original 25 predictor values are unchanged
for every development/ranking row. Independent boolean-filter calculations
match **816 real-data traffic/arrival comparisons across 48 departures**.
All 17 run artifact sets and all 12 feature files have saved hashes. Azure
evidence is in [verification.json](verification.json); the immutable feature
receipt is [feature_manifest.json](feature_manifest.json).

The parent Phase 1 manifest includes creation timestamps, per-file elapsed
times, and machine/resource measurements, so its entire-file hash differs
between Windows and Azure even when all 43 data artifacts match. The producing
Azure manifest is retained in [parent_manifest_azure.json](parent_manifest_azure.json).
The local verifier checks its exact archived hash and compares every parent
field except those explicit measurements. It does not change either original
manifest, loosen data/code/split checks, or regenerate the selected model.
All 16 development scores reproduce locally, and loading the final model on
Windows reproduces 1,000 saved ranking predictions with **zero difference**.
These checks are recorded in [local_verification.json](local_verification.json).

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_phase3.py --runs runs/azure_arcw0_phase3 --parent-manifest docs/phase3/parent_manifest_azure.json --output docs/phase3/local_verification.json
```

For a later eligible version, build with `tools/build_submission.py` and verify
the export hash before using `tools/prc_submit.py upload`. To check a submitted
version without changing historical Phase 2 evidence:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/prc_submit.py status --filename vigorous-whistle_v2.parquet --report-dir docs/phase3
```

For this downloaded Azure bundle, use that explicit parent receipt for local
verification. A new native feature preparation writes a new bundle descriptor
with its own producing parent manifest; it must not overwrite this bundle.

Feature preparation took **67.68 seconds** with a sampled peak RSS of **4.61
GiB**. Development jobs took **26.99-51.81 seconds** each, and the refit took
**43.38 seconds** with **3.38 GiB** sampled peak RSS. Only arcw0 ran this work;
no extra VMs, GPU resources, AWS resources, or external data were used.
