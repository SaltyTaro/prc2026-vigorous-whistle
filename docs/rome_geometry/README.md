# Rome geometry experiment against v7 — complete

**No candidate qualifies. V7 remains the best official submission at 275.0570
seconds RMSE on 344,841 rows.** All six native fits and four declared candidates
are complete, and both platform verifiers pass. The strongest mean-relative
candidate improves complete development RMSE by only **0.0070% seasonal /
0.0115% forward**, below the fixed **0.5% requirement on each fold**. This
campaign made no refit, export or submission. December remains reserved.

The [model declaration](../../configs/rome_geometry_v1.json) was recorded at
**01:38:03 UTC on 15 September 2026**, after the completed
[source and geometry audit](../source_geometry/README.md) and before any new
geometry-model score. The [complete selection](development_results.json) and
[result analysis](analysis.json) preserve every comparison. These are reused
development sets, separate from official competition scores.

| Complete recipe | Seasonal RMSE, seconds | Forward RMSE, seconds | Seasonal improvement | Forward improvement |
| --- | ---: | ---: | ---: | ---: |
| v7 reference | 322.819323 | 221.138156 | — | — |
| `geometry_v7_50` | 322.796721 | 221.112777 | 0.007001% | 0.011477% |
| `geometry_v7_100` | 322.812719 | 221.109663 | 0.002046% | 0.012885% |
| `geometry_queue_v7_50` | 322.813725 | 221.119916 | 0.001734% | 0.008249% |
| `geometry_queue_v7_100` | 322.847616 | 221.124840 | -0.008764% | 0.006022% |

`geometry_v7_50` has the lowest mean relative RMSE but is **not promoted**:
`selected_candidate` and `refit_iterations` are null. Its absolute gains are
0.022602 and 0.025380 seconds. LIRF itself improves by 0.154798 and 0.175369
seconds; daily squared error improves on 34/62 seasonal and 18/30 forward days.
Queue additions provide no consistent benefit. This result concerns the tested
feature contract and model replacement, rather than every possible geometry model.

## Fixed comparison and complete populations

Each fold fits a reproduction control, geometry, and geometry plus shared-route
queues. Settings match v7's LIRF LightGBM: 31 leaves, minimum leaf 50, L2 20,
learning rate 0.04, 1,800-tree cap, 120-round October stopping and seed 20260913.
The signed NM residual target is unchanged. All original clock-present LIRF
fitting rows remain included, including rows without geometry. The inherited
276 predictors expand to 291 with geometry and 306 with queues.

On LIRF rows with an NM clock and a supported connected route, each candidate is
`v7 + 0.5 * candidate_weight * (new_airport_prediction - original_airport_prediction)`.
The common candidate weights are 0.5 and 1.0 for each new variant. Every other
output is exactly v7. Both controls reproduce the original airport predictions
with **zero difference**, at 816 seasonal and 945 forward trees. Geometry selects
782/666 trees and geometry plus queues selects 754/657, respectively.

All **344,419 seasonal** and **162,332 forward** scoring rows are included,
with no label clipping or filtering. There are 25,999 and 11,071 eligible rows;
the other 318,420 and 151,261 outputs remain identical to v7. No weights, airport
choices or schedule rules were adjusted after inspecting these comparisons.

## Geometry, traffic and availability

The [feature declaration](../../configs/features_rome_geometry_v1.json) assigns
January 2025 maps to January-June movements, July 2025 maps to July-November,
and the matching January/July 2026 maps to ranking departures, using actual UTC
movement dates. Four previously cached LIRF maps are reused. LFPG remains
excluded because its historical parking points failed the earlier controls.

Fifteen static features describe supported parking coordinates, runway-access
coordinates, route/direct distance, detour, edge/junction counts, turns and
availability. Routes use undirected taxiways, taxilanes and linked parking
lead-ins, with no runway traversal. They are mapped geometric minima to any
access on the recorded runway, not actual clearances or a known runway threshold.

Fifteen traffic features describe earlier pushes, active occupancy and completed
service on shared route edges. Windows are 15/60 minutes. Off-block anchors
prefer AOBT then LOBT when the interval to movement is 0–6 hours, otherwise
movement minus 20 minutes, matching the existing retrospective convention.
Pushes and takeoffs must be strictly earlier than the query; the focal movement
and all peers with its nonmissing flight ID are excluded. Active occupancy uses
the supplied takeoff endpoint and is a retrospective proxy, not measured queue
order. Completed-service proxies retain signed and extreme intervals.

No departure label or airport block time enters shared feature generation.
OpenStreetMap contributors retain attribution and **ODbL 1.0** for maps and
derived geometry; the archived parking-tag guide is **CC BY-SA 2.0**. See the
[source license notice](../source_geometry/EXTERNAL_DATA_LICENSES.md).

The extension bundle is `data/rome_geometry/587c526337697ab57602`, full identity
`587c526337697ab57602055f566cdfe06cf8a4bcfd83d841a366f9c62b34fda5`.
It stores 30 new columns for 175,535 LIRF rows and references the unchanged
2,264,211-row timeline parent by hash. Geometry covers 26,503/26,899 LIRF ranking
departures, **98.5278%**. Preparation took 29.09 seconds at 0.69 GiB sampled RSS.

## Verification, execution and preservation

**61 tests pass on each platform**: 27 Rome tests and 34 unchanged geometry-audit
tests. The [Windows](tests_final_local.xml) and [Azure](tests_azure.xml) records
contain the same test identities. Both feature checks verify all **2,633,025
static values** and independently reconstruct queues on 515 rows with zero
difference: [Windows](local_feature_verification.json),
[Azure](feature_verification.json).

Both [Windows](local_verification.json) and [Azure](verification.json) native
verifiers reproduce all eight complete candidate-fold metrics and **112,221
native prediction values with zero difference**. They verify fitting/stopping
IDs, fitting-only encoders, original control settings and unchanged fallback
predictions. The source is isolated in `src/prc2026_rome_geometry/` and is frozen
at model-code SHA-256
`0d2a036a4adc51391b29d0a2c25e9a1fe223aa96befd35be5cc2451bce5766a0`.
The feature manifest SHA-256 is
`9f39f13500ae4c7d4a6619679a9977e1da42c52777067bc365c8a5870e064701`.

Only arcw0 ran model fits. The batch ran from **02:02:42 to 02:04:51 UTC**;
the six native fits total 45.36 seconds and peak at 1.45 GiB sampled RSS.
[Completion](batch_completion.json) records successful tests, preparation
verification, fitting, selection and native verification. The
[final read-only state check](final_execution_state.json) finds no active
project Python jobs and 1,171,894,272 bytes free. The worker remains running.

Original Azure outputs are `runs/rome_geometry/`; the local mirror is
`runs/azure_arcw0_rome_geometry/`. The verified development archive is
`runs/rome_geometry-development.tar.gz`, **44,684,680 bytes**, 104 files,
SHA-256 `7de6023e9551b817eb3e32157fee7d82b03888f8d5dd0c4ea2edb9909d0f37f0`.
See the [mirror receipt](development_mirror.json), [final archives](archive_receipts.json)
and [preservation record](preservation_final.json). Prior sources, data, models
and v1–v7 exports remain preserved; root narrative updates have byte snapshots.

Preparation corrections are disclosed separately from completed model runs.
The timestamp spelling was corrected to the inherited `MVT_TIME_UTC_mvt` before
preparation. An invalid pandas timezone conversion in the initial test fixture
was corrected; the failed test record remains. The first slow preparation was
stopped before a bundle or model existed, and output assembly was vectorized.
[Equivalence](preparation_corrections/assembly_equivalence.json) is exact on
1,200 rows / 36,000 values. The
[effective pretraining declaration](declaration_after_preparation_fix.json)
preserves the final code identity alongside the [original receipt](declaration.json).
A subsequent pandas read-only-array correction affects only the independent
verifier and has a regression test. On Azure, installing NetworkX with pip
failed because pip was absent; the recorded uv installation succeeded without
changing prior library versions. Every completed model was fitted once.

## Reproduction and next hypothesis

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_rome_geometry.py tests/test_source_geometry_audit.py
.\.venv\Scripts\python.exe -X utf8 -B tools/check_rome_geometry_features.py --output docs/rome_geometry/local_feature_verification_recheck.json
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_rome_geometry.py --runs runs/azure_arcw0_rome_geometry --surface-runs runs/azure_arcw0_surface_airports --timeline-runs runs/azure_arcw0_timeline --feature-report docs/rome_geometry/local_feature_verification.json --output docs/rome_geometry/local_verification_recheck.json
```

The [earlier v7 gain review](../competition/gain_review_v7_20260915/README.md)
proposes teaching the fallback with additional fitting rows whose NM inputs
are jointly masked, rebuilding every affected derived feature. This remains
untested. Natural NM absence has exceptional reporting patterns that synthetic
masking may not reproduce. The saved missing-NM-origin slice contributes
50.25%/22.31% of seasonal/forward squared error; it is defined by missing
`ADEP_flt` and must not be relabeled as the exact missing-clock routing slice.
Any new experiment needs its own declaration and complete comparison against
v7. This completed campaign does not begin that follow-up or use December.
