# Phase 4: reporting clocks and airport specialists

Phase 4 completed on 13 September 2026 using only the existing Azure `arcw0`.
A **fixed 50/50 global/airport LightGBM blend**, with a conditional schedule
offset for missing-record LIRF departures, is selected. It scores **372.8550488
seconds** seasonally and **245.8626853 seconds** on November, improving Phase 3
by **21.53% / 21.09%**. Official **v3 scored 331.9269 seconds RMSE on all
344,841 rows**, a **21.21% improvement** over v2's 421.3047. Development and
official scores are different evaluations. December remains reserved.

## Official submission and standings

`vigorous-whistle_v3.parquet` was uploaded at **18:07:57 UTC on 13 September**
and processed at **18:08:27 UTC**. The public organizer API confirms
**331.9269 RMSE** and **344,841 used pairs**. The version is immutable.

- File: **1,338,167 bytes**; SHA-256
  `c445bc304ea3f548890fae74825a55ec3d2be6d4f7404e65d4b40ef8883cf7f5`.
- Submission ID: `ff1225ea-b929-4b4b-bac3-0ff6e379c672`.
- Schema: exact template ID order, `MVT_ID_mvt: double` and
  `TAXITIME_SEC_mvt: int32`; nearest-second rounding with ties to even.
- Prediction range: **-843 to 115,075 seconds**, including 28 negative integer
  predictions under the declared no-clipping policy. All rows were scored.

The **18:10 UTC** full leaderboard snapshot ranks the team **74th of 131**,
up from 96th in the earlier Phase-3 snapshot. Third place is **261.9189**,
requiring a further **21.09% RMSE reduction** from v3 to that dated score.
The top-three goal remains work for later phases; this is not a final cutoff.

Evidence: [official result](official_result.json),
[submission and model receipt](submission_v3.json), and
[full leaderboard snapshot](leaderboard_snapshot.json). Local export artifacts
are under `runs/submissions/`; previous versions and their reports are preserved.

## Diagnostic finding

The allowed January-November population contains **1,919,370 departures**,
including **190 labels above three hours**. Of these, 169 are missing-NM LIRF
records. There are **1,399 LIRF departures with both NM origin and airport
aircraft type missing**, with **397 seasonal**, **52 November**, and **383
ranking rows** satisfying the same input-only condition. October contains
115 such rows for early stopping.

For many of these records, movement minus scheduled time is within a few
seconds of the supplied taxi-out label. By the target definition, the residual
is scheduled time minus block time. This is evidence of a reporting convention,
not proof that all long ground times are mistakes. The schedule interval is
already an allowed Phase-1 predictor. Its use as an explicit offset lets a
model extrapolate beyond the taxi times seen in a particular leaf.

The rule is imperfect: some ordinary taxi times have large schedule delays,
and some extreme records have differences close to a day. Flight designator
shapes and missing metadata offer possible input-only clues. No departure
block time or actual NM clock enters the new predictors.

Exploratory replacement of the Phase-3 prediction by movement minus schedule
for this group scored **404.3643679 seasonal / 263.8872323 forward**. This
diagnostic was inspected before declaring the model batch and is explicitly
recorded in its configuration. January-November labels, including development
labels, informed the hypothesis. These evaluations are reused development
data; they are not a new independent test.

Aggregate evidence is in [diagnostics.json](diagnostics.json). The diagnostic
tool reads January-November departure labels separately from feature creation.
Raw records and full flight designators are not published in this report.

## Development results and selection

[phase4_v1.json](../../configs/phase4_v1.json) declares seven recipes before
their model runs. All retain the complete original fit, October stopping, and
scoring populations. A specialist trains on its input-defined subset, and the
composite pipeline still predicts every scoring row.

| Recipe | Seasonal RMSE | November RMSE | Behavior |
| --- | ---: | ---: | --- |
| Phase-3 reference | 475.166887 | 311.570088 | Previously selected timing/traffic model |
| `schedule_gate` | 404.364368 | 263.887232 | Replace the eligible LIRF group by its finite schedule interval |
| `clock_features` | 458.816590 | 299.292852 | Standard LightGBM settings with record-quality predictors |
| `conditional_offset` | 378.897179 | 249.177547 | Fit target minus the conditional interval; restore the interval for prediction |
| `conditional_offset_fine` | 379.414220 | 249.046957 | Same residual target, 31 leaves, 25-row leaf minimum, L2 of 20 |
| `missing_record_specialist` | 375.438598 | 247.980545 | Separate 7-leaf correction for eligible LIRF records |
| `airport_specialists` | 373.914367 | 249.354615 | Ten separately fitted 31-leaf airport models |
| **`airport_blend`** | **372.855049** | **245.862685** | **Fixed 50% global/airport blend; selected** |

All seven recipes improve both complete scoring sets. The selected blend lowers
RMSE by **21.53% seasonally** and **21.09% forward**, and improves **all ten
airports on both folds**. Squared error falls on 57 of 62 seasonal days and
29 of 30 November days. Paired 2,000-replicate day-bootstrap intervals for
candidate minus Phase-3 RMSE are **[-171.24, -43.12] seconds** seasonally and
**[-128.99, -9.79] seconds** forward. These describe reused development data.
The full results and receipts are in [development_results.json](development_results.json)
and [airport_errors.csv](airport_errors.csv).

Global and airport training allows up to 1,800 trees with 120-round October
early stopping. The small LIRF specialist allows 800 trees and 80-round stopping,
with stronger regularization. Each component fits its own categorical vocabulary
on its fitting rows. Unseen categories map to missing.

Selection requires lower complete-set RMSE than the Phase-3 reference on both
folds and minimizes the equally weighted mean of their RMSE ratios. Refit
iterations are the rounded median of the two October-selected tree counts,
separately for each component. No labels or predictions are clipped. Negative
schedule intervals and unusual labels are retained. Missing schedule intervals
fall back to a zero offset or the reference prediction, as declared.

The selected refit uses **1,919,370 January-November departures**, one global
model and ten airport models, and produces **344,841 ranking predictions**.
Its frozen iteration counts are global 1,601; EDDF 692; EDDM 574; EGLL 717;
EHAM 616; LEBL 668; LEMD 616; LFPG 650; LIRF 680; LSZH 332; and LTFM 877.
The run is `20260913T175343734071Z_refit_airport_blend_lightgbm`, saved locally
under `runs/azure_arcw0_phase4/` and under `runs/phase4/` on arcw0.

## Remaining errors and next phase

LIRF RMSE falls from 1,323.14 to **825.58** seasonally and from 829.75 to
**472.57** forward. Missing-NM RMSE falls from 2,997.90 to **1,883.19** and
from 2,292.76 to **1,189.30**. These are substantial gains, but not a solution
to every reporting anomaly. Labels above three hours still account for
**40.83% of seasonal squared error** and **13.66% of forward squared error**.
Seasonal maximum absolute error remains about 84,273 seconds.

Phase 5 should retain this blend as its reference, investigate the remaining
rare clock/reporting regimes using fitting data, and test bounded residual-model
iteration/regularization changes, selected seeds, and saved-prediction blends.
The global seasonal run selected 1,754 of its 1,800 allowed trees, which makes
the iteration cap a reasonable bounded follow-up. Any new gate, clipping rule,
or feature family needs a declared comparison on both complete sets. Do not
tune against individual hidden ranking labels or spend December's audit now.

## Features and provenance

The new bundle has **112 predictors**: the exact 86 selected Phase-3 predictors
plus 26 input-only record-quality and clock predictors. These include aircraft
and flight missingness; flight prefix and character-count shape; prefix lengths
and repetitions; clock seconds, UTC date offsets, and modulo-day intervals;
initial NM clock coverage; and the conditional LIRF indicator/offset. Neither
movement/flight IDs nor full flight designators are predictors.

The raw reader projects an explicit allowlist. It excludes departure targets,
airport block times, `AOBT_3_flt`, `LOBT_flt`, and `ARVT_3_flt`. The last three
remain excluded pending original organizer guidance, without asserting an
official prohibition. No external data or new arrival-label features are used.
Weather and geometry are deferred while this supplied-data hypothesis is tested.

Code is isolated in `src/prc2026_phase4/`; prior source packages are unchanged.
The feature bundle binds all original inputs, source code, dependencies, split
IDs, and parent artifacts. Its Phase-1 scientific identity excludes only creation
time, preparation host/resource metadata, and per-source elapsed time. The
archived producing Phase-1 manifest proves equivalence on Windows; all other
fields and all 43 parent artifact hashes must match. The Phase-3 manifest and
its 12 feature artifact hashes remain exact dependencies. Original manifests
are preserved.

The bundle is `2ceb3449c26767a9731121bdd4988cff7d57f6e9310b84ac680657a131ae9cff`,
stored in `data/phase4/2ceb3449c26767a97311`. The model-code receipt is
`124b752ecc6717ec9ed9f5fcd32237716a637decc41200c3ed411d6122c788c4`.

Each experiment contains a self-contained `pipeline.json`, native component
models, fit-only encoders, component-specific fit/stopping IDs and settings,
iteration histories, importances, full predictions/metrics, resources, and
artifact hashes. Imported model components retain their source-run receipt.

## Reproduction and verification

The prepared `arcw0` project can run the declared batch in a fresh output
directory. This avoids mixing experiments from changed configurations or code.

```bash
bash infra/azure/phase4.sh runs/phase4_reproduction runs/phase4_reproduction/report
.venv/bin/python tools/run_phase4.py --output runs/phase4_reproduction refit --selection runs/phase4_reproduction/selection.json
.venv/bin/python tools/verify_phase4.py --runs runs/phase4_reproduction --output runs/phase4_reproduction/verification.json
```

The runner uses the pinned Phase-3 dependencies, runs all **71 tests**, prepares
the fresh feature bundle, and trains one component at a time on arcw0. The tests
passed on Windows and Azure before this batch. The one existing pandas warning
comes from the unchanged Phase-2 encoder.

The independent verifier checks every artifact hash, all 86 inherited values
and the conditional offset on every row, every complete development score,
component fitting/stopping IDs, and model reloads on 1,000 rows per fold. Reload
samples include all eligible LIRF scoring/ranking rows and random other rows.
The refit excludes December and every flight group protected by that audit.

Windows and Azure verification each cover all **2,264,211 rows**, **15 run artifact sets**,
**14 complete development scores**, and component reloads on 1,000 rows per
fold, with **zero prediction difference**. See [verification.json](verification.json)
and [local_verification.json](local_verification.json).
The checker validates the ordered source configuration and compares its full
canonical hash with saved JSON receipts; sorted-key receipt serialization does
not change the declared execution order. This corrected an initial verifier
bookkeeping failure without changing feature or model source, fits, or predictions.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_phase4.py --runs runs/azure_arcw0_phase4 --output docs/phase4/local_verification.json
```

Preparation took **39.04 seconds**, peaking at **1.11 GiB** sampled RSS.
Individual development recipes took **2.53-77.27 seconds**. Refit took **147.46
seconds**, peaking at **3.99 GiB**. These measurements cover one job at a time
on arcw0. No additional Azure worker or AWS resource was started.

Submission remains a separate, validated step. Status evidence for a new
version belongs under `docs/phase4/`, preserving the older phase reports.
