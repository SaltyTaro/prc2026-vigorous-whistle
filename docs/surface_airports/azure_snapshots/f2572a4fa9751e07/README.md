# Frozen timeline features in the airport models

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
The six additional setting/routing tests pass locally, bringing the current
distinct passing local test set to 184. The existing arcw0 runs the full suite
before training. No additional worker is started and December stays reserved.

`tools/batch_surface_airports.py` runs both folds, all four candidate comparisons,
native verification and the summary. Results use `runs/surface_airports/` and
`docs/surface_airports/`; local native mirrors use
`runs/azure_arcw0_surface_airports/`. The verifier checks each component's
fitting/stopping identities and fitting-only categorical vocabulary, native
predictions for every airport including unusual clocks, shuffled airport
routing, complete candidate scores, and the unchanged missing-clock fallback.
A qualifying recipe would require a separately verified January-November
refit before export. The batch does not submit a competition file.
