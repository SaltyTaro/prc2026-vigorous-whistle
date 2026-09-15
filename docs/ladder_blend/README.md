# Cross-campaign ladder blends

Declared at **16:25:25 UTC on 14 September 2026**, before any cross-campaign
mixture score. This is a final bounded combination of completed model branches.
V6 remains the incumbent at 277.4508 seconds official RMSE.

**All eight mixtures are complete and none meets the two-fold 0.5% promotion
margin. V6 is retained.** The candidate closest to the joint margin is
`missing_ground_v6_75__clock300_missing300_v6_50__left25`, with
**323.0088876 seasonal / 222.2824682 forward** RMSE. Its **0.5873% / 0.4678%**
improvements miss the forward threshold. The lowest mean-ratio candidate,
`missing_ground_v6_50__clock300_missing300_v6_50__left25`, scores
322.5884002 / 222.4396326 and also fails promotion. Neither is selected.

These are reused development results. The eight declared weights are unchanged;
no further interpolation was evaluated. There is no refit, v7 export or upload.
Both platform verifiers pass, with identical scientific fields and zero native
prediction difference. The complete 153-test Azure suite passes; the same 153
distinct tests pass in the recorded Windows runs.

Evidence: [selection](development_results.json), [analysis](analysis.json),
[Azure verification](verification.json), [Windows verification](local_verification.json),
[comparison](verification_comparison.json), [completion](completion.json), and
[aggregate result for all 64 recipes](../ladder/CAMPAIGN_RESULT.md).

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

The model-code SHA-256 is
`c064573f92c623ffa9dbb6f9fd23e6365aa3ad51eed06d4d6fb0df27bd367e51`;
selection SHA-256 is
`4e5bc238b4f69136ff449a5a73cc09e500ad50b299cdf488f35d764bedfed2eb`.
The [archive](development_archive.json) contains 142,689 bytes, SHA-256
`26da8791ea222e588918bb49e47503369f5f29e2a01ff38d6ad64712b40c8131`.
All seven archived files are restored and hash-checked locally; native models
remain in the two parent archives. No campaign jobs remain active.
