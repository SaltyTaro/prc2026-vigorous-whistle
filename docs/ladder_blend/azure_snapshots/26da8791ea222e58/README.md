# Cross-campaign ladder blends

Declared at **16:25:25 UTC on 14 September 2026**, before any cross-campaign
mixture score. This is a final bounded combination of completed model branches.
V6 remains the incumbent at 277.4508 seconds official RMSE.

The first ladder's missing-clock CatBoost improves forward development while
regressing seasonal. The second ladder's scaled combination improves seasonal
by 0.77% and forward by 0.29%, missing the two-fold 0.5% promotion margin. The
new [declaration](declaration.json) tests their complementary errors with
**eight fixed common-weight mixtures**, reusing saved complete predictions.

The left parent is either `missing_ground_v6_50` or
`missing_ground_v6_75` from the first ladder. The right parent is either
`clock300_missing300_v6_25` or `clock300_missing300_v6_50` from the scale
batch. Left weights are 25% or 50%. Every combination uses identical weights
at all airports and on both original folds.

Both parent selections are bound by their file hashes and must already have
passed native verification. Every parent recipe is reconstructed and its
complete metrics checked again. Native checks include all missing-clock rows
and sampled available-clock rows, including large clock disagreements.

All eight candidates must finish before selection. Promotion still requires
0.5% lower complete RMSE than v6 on each original fold, then the lowest equally
weighted mean ratio. There are no new development native fits, features,
external sources, airport-specific weights, threshold changes, or December
labels in this check.

A qualifying refit would fit the three active new components at their original
two-fold October medians: 1,006 trees for the CatBoost fallback, 1,382 for the
scaled-clock LightGBM, and 182 for the scaled missing-clock LightGBM. Every
December flight group remains excluded. Imported v6 models remain byte-identical.
Both platform verifications are required before any refit/export.

Source is isolated in `src/prc2026_ladder_blend/`. Azure evaluation uses
`tools/evaluate_ladder_blend.py`; independent verification adds
`--verify-only` and a distinct report path. There is no unattended upload.

