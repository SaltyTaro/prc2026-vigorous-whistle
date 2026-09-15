# Phase 5: bounded model tuning and prediction ensembles

Phase 5 completed on **13 September 2026 UTC** using
only the existing Azure `arcw0`. It selected a blend of the longer CatBoost
pipeline and a three-seed LightGBM average. Complete development RMSE is
**367.084321 seasonal / 244.554901 November**, improving the v3 development
reference by **1.55% / 0.53%**. The January-November refit and both platform
verifications are complete. Official **v4 scored 326.6895 seconds RMSE on all
344,841 rows**, improving v3's 331.9269 by **1.58%**. December remains reserved
for the final audit.

## Official submission and standings

`vigorous-whistle_v4.parquet` was uploaded at **20:38:40 UTC on 13 September**
and processed at **20:39:14 UTC**. Both the official filename and full
**344,841 used pairs** are confirmed. The version is immutable.

- Size: **1,338,569 bytes**; SHA-256
  `bcde3c25fd38456099e69278e35ccd9d629882e1c742657eab4701a1e36d0f90`.
- Submission ID: `0ae52be7-8efa-4728-9f80-97dbd4f2e23c`.
- Exact template IDs/order; `MVT_ID_mvt: double`, `TAXITIME_SEC_mvt: int32`.
- Predictions use nearest-second, ties-to-even rounding with no clipping.
  The integer range is **-133 to 116,303 seconds**, including 20 negatives;
  all rows were scored under the existing complete-population policy.
- Four submissions were used on 13 September UTC, including v1-v4.

The **20:40 UTC** full leaderboard snapshot ranks the team **68th of 131**,
compared with 74th in the Phase-4 snapshot. Third place is **261.9189 seconds**,
requiring a further **19.83% RMSE reduction** from v4 to that dated score.
The top-three objective is not yet achieved; the snapshot is not a final cutoff.
Development and leaderboard RMSE remain separate evaluations.

Evidence: [official result](official_result.json),
[submission/model/verification receipt](submission_v4.json), and
[full leaderboard snapshot](leaderboard_snapshot.json). The submitted Parquet,
export receipt, upload ledger, and status are under `runs/submissions/`.
v1-v3 and their historical evidence are preserved.

## Declared comparisons

[phase5_v1.json](../../configs/phase5_v1.json) was declared before the initial
batch. It fixes the 112 Phase-4 predictors, offset policy, original fit/October
stopping/scoring IDs, and complete-set RMSE. No labels or predictions are clipped.
All models retain the fixed 50% global / 50% airport structure; a comparison
replaces only the named component group.

| Initial recipe | Seasonal RMSE | November RMSE | Change from v3 |
| --- | ---: | ---: | --- |
| `reference` | 372.855049 | 245.862685 | Import the verified v3 development pipelines |
| `global_long` | 372.855049 | 245.358963 | Raise global LightGBM cap to 4,000; stop after 200 rounds without improvement |
| `global_capacity` | 372.349229 | 245.201786 | 127 leaves, minimum 100 rows per leaf, L2 of 20 |
| `global_regularized` | 373.199396 | 246.910923 | 31 leaves, minimum 200 rows per leaf, L2 of 50 |
| `airport_capacity` | 372.780613 | 245.220873 | 63-leaf airport models, minimum 50 rows per leaf, L2 of 20 |
| `catboost_global` | 368.388898 | 247.008473 | 3,000-tree CPU CatBoost residual model, depth 8 |

The declared anchor rule chooses the best mean relative RMSE among initial
recipes improving both folds and the reference. This selects `global_capacity`.
The two additional seeds are **20260914 and 20260915**, retraining every component
of that recipe. The seed plan was frozen before the adaptive follow-up below.
Full initial metrics and resource receipts are in
[initial_results.json](initial_results.json).

Both initial CatBoost models selected all 3,000 trees, with October error still
falling. A separate [follow-up declaration](../../configs/phase5_followup_v1.json),
written at **19:21:22 UTC on 13 September**, raises only the global CatBoost cap
to 6,000 and stopping patience to 200. It retains the original airport models
and all feature/validation policies. The first twelve complete evaluations
had already been inspected; this is an adaptive development comparison.
No additional follow-up seeds or further tuning are part of this phase.

The longer CatBoost blend scores **366.768133 seasonal / 245.375614 November**.
Its October-selected tree counts are **5,917 / 5,995**. The November improvement
is approximately 0.20%, which is below the promotion threshold on its own.

The final search contains **28 unique weight recipes**: the declared individuals,
an equal three-seed mean, a small grid among the reference and two strongest
initial alternatives, reference/seed-mean mixtures, and seven follow-up recipes.
The follow-up adds its individual prediction and weights of 25%, 50%, and 75%
against either the reference or the three-seed mean. Weights are common across
airports and folds. Every recipe predicts every scoring row.

Promotion requires **at least 0.5% lower RMSE on each complete fold** than v3,
then minimizes the equally weighted mean of the two RMSE ratios. Refit tree
counts use the rounded median of the two October-selected counts, separately
for each component and seed. Identical refit components are fitted only once.
These are reused development evaluations, not a fresh independent test.

## Selected recipe and uncertainty

Four of the 28 recipes pass the declared promotion margin on both complete
sets. The selected recipe is `followup_catboost_long_three_seed_mean_75`:
**75% of the longer CatBoost pipeline and 25% of the equal three-seed LightGBM
average**. Each member pipeline itself retains the fixed global/airport blend.
The four member weights are 0.75 for `catboost_long` and 1/12 each for
`global_capacity`, `seed_20260914`, and `seed_20260915`.

| Qualifying recipe | Seasonal RMSE | November RMSE |
| --- | ---: | ---: |
| **75% longer CatBoost + 25% three-seed mean; selected** | **367.084321** | **244.554901** |
| 50% longer CatBoost + 50% three-seed mean | 368.029138 | 244.244804 |
| 50% longer CatBoost + 50% v3 reference | 368.487334 | 244.613983 |
| 25% longer CatBoost + 75% three-seed mean | 369.597764 | 244.447268 |

The selected model improves squared error on **53 of 62 seasonal days** and
**24 of 30 November days**. Paired 2,000-replicate day-bootstrap intervals for
candidate minus v3 RMSE are **[-8.29, -3.63] seconds** seasonally and
**[-2.76, 0.03] seconds** in November. The November interval includes zero and
its 0.53% gain clears the 0.5% threshold narrowly. These descriptive intervals
do not establish an independent generalization gain; the reserved audit remains
necessary. Neither the threshold nor the selection rule was relaxed.

Complete metrics, all 28 recipes, resource receipts, and the frozen refit
blueprint are in [development_results.json](development_results.json).
The refit global tree counts are **5,956 CatBoost**, and **1,074 / 1,026 / 1,216
LightGBM** for seeds 20260913 / 20260914 / 20260915. Airport counts are frozen
separately in the same receipt. The 44 logical component references share
34 native models, deduplicated only when their full refit recipe is equal.

All ten airports improve seasonally; nine improve in November. November LIRF
worsens by **0.57 seconds RMSE** to 473.14, while seasonal LIRF improves by
21.42 seconds to 804.16. Missing-NM RMSE is **1,840.44 / 1,188.01 seconds**.
Labels above three hours still contribute **41.26% seasonal / 13.99% November**
of squared error, and
the seasonal maximum absolute error remains **83,729.80 seconds**. Model tuning
does not resolve every rare reporting regime. Complete airport and tail slices
remain in the selection receipt; no rows were removed to obtain these gains.

## Fitting-only clock investigation

[fitting_diagnostics.json](fitting_diagnostics.json) uses only the intersection
of the two original fitting populations: **1,226,945 departures**. January,
July, October, November, December, and all hidden ranking labels are excluded
from this diagnostic. Its ID hash is
`5a8ca81854dc7ddc29c0cc0e82c5f3a0a3bc2eb2b22ac17d4da387e7090b63f7`.

The existing LIRF missing-record gate contains 835 fitting rows, including
102 labels above three hours. In contrast, NM-present LIRF records with a
schedule interval above six hours contain 52 fitting examples and **no target
above three hours**. NM-present records across all airports with schedule
intervals above one day contain 82 examples and **no target above three hours**.
Broadening the schedule-offset gate to such groups is not supported by these
fitting data. No feature or offset rule was changed in Phase 5.

## Provenance and verification

New model code is isolated in `src/prc2026_phase5/`. The Phase-1 through Phase-4
packages are unchanged. Native LightGBM/CatBoost components retain their own
fit-only encoders, fit/stopping ID hashes, settings, histories, and importances.
Copied components retain their source-run hashes. Complete predictions,
metrics, resource measurements, and artifact hashes accompany every run.

- Frozen feature bundle:
  `2ceb3449c26767a9731121bdd4988cff7d57f6e9310b84ac680657a131ae9cff`.
- Model-code hash:
  `653e32a3a7d88623e5eb5e07c9762a14f7e185bbe02397a40c4aea747b561ac0`.
- Initial configuration hash:
  `ec09566a952abdb2b6a01d4ed1075b93a3d519a194aa3e4dd05942adda474eea`.
- Follow-up configuration hash:
  `0b1dae14c4ff69379eacf265106bd133328d9ae90120aa07586708ee7f6b1b36`.

All **80 pipeline tests passed on Windows and Azure** before training; the final
suite has **81 passing tests on both platforms**, including the metric-receipt
regression test added during verification. They
include native mixed-model persistence, fit-only CatBoost categories, strict
movement-ID alignment, invalid ensemble weights, and audit/refit guards. One
existing pandas warning comes from the unchanged Phase-2 encoder.

`tools/verify_phase5.py` checks the complete feature/artifact chain, every
component's fit/stopping IDs, exact copied-model hashes, complete development
metrics and all reported slices, independently reconstructed blends, the seed
anchor rule, promotion eligibility, and frozen refit counts. Native reload
samples include every gated LIRF scoring/ranking row plus deterministic random
rows, 1,000 per population. The first verifier attempt exposed JSON's conversion
of integer month-weight keys to strings. The checker now normalizes that key
representation while still rejecting changed values and colliding keys. No
model source, fit, saved prediction, weight, or score was changed by this fix.
Windows and Azure each verify **21 run artifact sets**, all **20 development
evaluations**, and **56 individual/blend evaluations**. Native reload differences
are exactly zero; independent blend reconstruction differs by at most
**1.4552e-11 seconds**. All 2,264,211 frozen feature-row identities and all 43
Phase-1, 12 Phase-3, and 12 Phase-4 artifact hashes are checked. The inherited
feature bytes are unchanged from the fully audited Phase-4 bundle.
See [Azure verification](verification.json),
[Windows verification](local_verification.json), and the
[Windows](tests_windows.xml) / [Azure](tests_azure.xml) test records.

Only arcw0 is used. Credentials remain in the local workspace; no PRC credentials
are copied to the VM. The original raw files and download hashes are preserved.
No external data, GPU, additional worker, or AWS resource is used.

The refit uses **1,919,370 rows**, **34 distinct native models**, and **28,772
total trees**. It took **1,246.13 seconds (20 minutes 46 seconds)**, with
**7.06 GiB peak sampled RSS** and a mean of 14.56 active CPU cores. The completed
run is `20260913T195813824072Z_refit_followup_catboost_long_three_seed_mean_75`.
The final [VM status record](azure_status.json) confirms arcw0 running and
arcw1-arcw7 deallocated; the root disk had approximately 6.4 GiB free.

The [model archive receipt](artifact_archive.json) binds the 647,181,459-byte
Azure transfer to SHA-256
`fec1b54b6f8699bca2c45af1c2d770ea0f49154337e831b06315a5990d5a49e3`.
The local restoration checked that hash and preserved identical existing
evidence. [code_package.json](code_package.json) records the source package;
original native models and historical phase reports remain intact.

## Reproduction

On the prepared arcw0 project, use fresh directories to preserve the original
models and selection receipts. The initial batch, seed anchor, adaptive batch,
replicas, and final blend selection are explicit steps:

```bash
bash infra/azure/phase5.sh runs/phase5_reproduction
.venv/bin/python tools/select_phase5.py --mode initial --runs runs/phase5_reproduction --report-dir runs/phase5_reproduction/report
bash infra/azure/phase5_followup.sh runs/phase5_reproduction_followup
bash infra/azure/phase5_seeds.sh runs/phase5_reproduction
.venv/bin/python tools/select_phase5.py --mode final --runs runs/phase5_reproduction --followup-runs runs/phase5_reproduction_followup --report-dir runs/phase5_reproduction/report
.venv/bin/python tools/run_phase5.py --output runs/phase5_reproduction refit --selection runs/phase5_reproduction/selection.json
.venv/bin/python tools/verify_phase5.py --runs runs/phase5_reproduction --followup-runs runs/phase5_reproduction_followup --require-refit --output runs/phase5_reproduction/verification.json
```

The refit command rejects an ineligible or changed selection. All commands use
the existing Phase-4 reference runs and feature bundle. Component fitting is
sequential and limited to the one authorized 16-vCPU worker. Dependency pins
are in `requirements-phase5.txt`, which inherits the tested Phase-4 environment.

After collecting the models, Windows verification uses the local mirror paths:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_phase5.py --runs runs/azure_arcw0_phase5 --followup-runs runs/azure_arcw0_phase5_followup --reference-runs runs/azure_arcw0_phase4 --require-refit --output docs/phase5/local_verification.json
```

`tools/package_phase5.py` packages the new source, declarations, tests, and Azure
runners without data or credentials. `tools/package_phase5_results.py` archives
only completed, hash-checked models and selection artifacts. Model packages and
bulk predictions are local artifacts under `runs/`, outside the source release.
The generic `tools/build_submission.py` and `tools/prc_submit.py` remain the only
export/upload path.

## Phase 6 handoff

The selected recipe, weights, features, and component counts are frozen. The next
phase is the reserved-period audit and reproduction from the original inputs,
with particular attention to November's small gain, LIRF, missing NM records,
and extreme-label error concentration. Audit code and any newly materialized
December features must have separate provenance; do not weaken the frozen
development guards or alter the Phase-1 through Phase-5 source packages.

Prepare the original-work method report, source attribution, and public GPLv3
release in the audit/release phases. No external data was used in this phase.
There is no further Phase-5 tuning or leaderboard probing after v4's score.
