# Frozen timeline features in the airport models

The completed comparison selects **`full_timeline_v6_100`**. It lowers complete
development RMSE from v6's 324.9170674788965 / 223.32713908273132 to
**322.8193230817628 seasonal / 221.13815634167557 forward**, improvements of
**0.6456246% / 0.9801687%**. All ten airports improve on both folds. This passes
the fixed 0.5% margin and has the lowest mean relative RMSE among eligible recipes.
The [selected refit](../surface_refit/README.md) uses the frozen October counts;
these development results are separate from the official v6 score.

Declared at **18:24:05 UTC on 14 September 2026**, after the complete timeline
comparison improved both folds but missed the seasonal promotion margin. This
final bounded follow-up tests the already frozen 276-feature combined contract
in the airport half of the prediction. No new raw feature, external input,
target transformation or validation split is introduced.

The [configuration](../../configs/surface_airports_v1.json) fixes twenty native
LightGBM fits: all ten airports on each original development fold. Settings
match the original NM-clock airport models exactly, including 31 leaves,
minimum leaf size 50, L2 penalty 20, learning rate 0.04, 1,800-tree cap,
120-round October early stopping, seed 20260913 and sixteen CPU threads.
Every signed/extreme label remains in the complete metric. Only clock-present
rows fit these components; every missing-clock prediction remains at v6.

Four complete recipes are declared. The `airport_only` family combines half
the original v6 global model with half the new airport models. The
`full_timeline` family combines 25% timeline LightGBM global, 25% timeline
CatBoost global, and 50% new airport models. Each family receives 50% or 100%
candidate weight against v6. All twenty fits and four recipes must finish;
promotion still requires at least 0.5% lower complete RMSE on both folds, then
the smallest mean relative RMSE. No airport-specific selection or weights are
allowed. The arrival-context experts are excluded after worsening seasonal scores.

This uses `data/timeline/ce42240e5e43eee1713f`, whose feature reconstruction and
complete original global-model scores already pass on Windows and Azure.
New source is isolated in `src/prc2026_surface_airports/`; it calls the frozen,
tested timeline native trainer and encoder with the matched airport settings.
Prior packages and every earlier declaration remain unchanged.

Model-code SHA-256:
`0fe8598a2ac9e5921907cd5fb9f0679f14e1d11be91bc5f28f73a6610b74673c`.
All 184 tests in the original batch passed on arcw0, matching the recorded
Windows test identities. Two later verifier regression tests also pass on both
platforms. The selected refit adds ten more tests; the [combined record](../next_lever/tests_combined.json)
matches all **196 current test identities** to passing evidence on both platforms.
No additional worker was started and December stays reserved.

`tools/batch_surface_airports.py` runs both folds, all four candidate comparisons,
native verification and the summary. Results use `runs/surface_airports/` and
`docs/surface_airports/`; local native mirrors use
`runs/azure_arcw0_surface_airports/`. The verifier checks each component's
fitting/stopping identities and fitting-only categorical vocabulary, native
predictions for every airport including unusual clocks, shuffled airport
routing, complete candidate scores, and the unchanged missing-clock fallback.
The completed batch does not submit a competition file. The separate
January-November refit must pass its own native verification before export.

| Fixed recipe | Seasonal RMSE, seconds | Forward RMSE, seconds | Eligible |
| --- | ---: | ---: | --- |
| `airport_only_v6_50` | 324.02545689343486 | 222.2590354409071 | No |
| `airport_only_v6_100` | 323.59481801431656 | 221.70253412627594 | No |
| `full_timeline_v6_50` | 323.1305120990251 | 221.42625903220988 | Yes |
| **`full_timeline_v6_100`** | **322.8193230817628** | **221.13815634167557** | **Selected** |

The [selection](development_results.json) and [diagnostics](analysis.json) retain
all complete populations and unusual labels. Daily squared error falls on 59 of
62 seasonal days and all 30 November days. Descriptive paired-day intervals for
candidate-minus-v6 RMSE are [-2.6632, -1.6668] and [-2.6944, -1.7655] seconds.
The frozen helper retains an obsolete comparison label in the JSON; the
[annotation](paired_day_label_note.json) records that the actual input reference
is v6. Neither the input arrays nor the interval calculations changed.
Most improvement comes from ordinary departures; seasonal labels above three
hours and negative labels worsen slightly and remain fully included.

The [Azure verifier](verification.json) and [Windows verifier](local_verification.json)
pass all twenty native airport components, complete candidate metrics,
fitting-only encoders, protected IDs and exact missing-clock fallback checks.
Native differences are zero. The first verifier call reversed the observed and
saved-JSON arguments of an inherited metric comparator. Its failure and original
source are preserved in [verifier_fix](verifier_fix/correction.json); the corrected
comparison and two regression tests passed without retraining, prediction changes
or selection changes. The original failed completion receipt is retained alongside
[completion after the correction](completion_after_verifier_fix.json).

Selection SHA-256:
`3a6edb7beb178744acca53c92e269f2ee92fa1524115adbda74bf876cdd4d042`.
The [development mirror](development_mirror.json) restores all 122 files from a
52,732,147-byte archive, SHA-256
`f2572a4fa9751e079a4a3fb038f10b80b2943bd8a8b62b0e4474ef460b717caa`.
The twenty native airport fits used 215.19 seconds in total, with 8.66 GiB
maximum sampled process RSS. The global components remain in the timeline archive.

To reproduce the complete development verification locally:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools\verify_surface_airports.py --runs runs\azure_arcw0_surface_airports --timeline-runs runs\azure_arcw0_timeline --feature-report docs\timeline\local_feature_verification.json --output docs\surface_airports\reproduction_verification.json
```

Use a new report filename to preserve the original verification evidence.
