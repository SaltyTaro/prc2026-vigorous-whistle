# V6 ladder campaign

Declared at **15:04:17 UTC on 14 September 2026**, after the user requested
continuing the ladder climb following v6's official result. V6 remains the
official best at **277.4508 seconds RMSE on all 344,841 rows**. The previous
models, source packages, feature bundle and submissions remain frozen.

The entire follow-up is now closed: this batch, the
[target-scale batch](../ladder_scale/README.md), and the
[final cross-campaign check](../ladder_blend/README.md) comprise **16 new native
fits and 64 declared recipes**. None qualifies for promotion. The closest recipe
improves seasonal/forward RMSE by **0.5873% / 0.4678%**, below the 0.5% requirement
on the forward fold. All 153 distinct tests and all three campaign verifiers pass
on Windows and Azure, with zero native prediction difference. There is no new
refit, export or upload. See the [complete result](CAMPAIGN_RESULT.md) and
[aggregate receipt](campaign_summary.json).

The [public standings](leaderboard_after_campaign.json) at **16:38:49 UTC on
14 September** still place v6 **20th of 134**. Third place scores 252.9192;
reaching that dated result requires another 8.84% RMSE reduction. The
[final integrity check](final_submission_integrity.json) preserves all six
exports and live bucket objects. No campaign jobs remain active on arcw0.

**All eight fits and 36 candidate recipes are complete. No candidate passes
the declared 0.5% improvement margin on both folds, so this batch retains v6.**
Five blends improve both periods by smaller amounts. The leading mean-ratio
recipe, `long50_v6_100`, scores **324.609866 seasonal / 222.966982 forward**,
improvements of **0.09455% / 0.16127%** over v6 development. It is not promoted,
refitted or submitted. These are reused development results.

The complete [selection](development_results.json), [analysis](analysis.json),
[Azure verification](verification.json), [Windows verification](local_verification.json),
[platform comparison](verification_comparison.json), [archive receipt](development_archive.json)
and [completion record](completion.json) preserve the result.

The [declaration](declaration.json) fixes **eight new native fits** and **36
complete candidate recipes** before any new component score. The
[configuration](../../configs/ladder_v1.json) records every model setting,
route, weight and promotion rule. December remains reserved; only the existing
16-vCPU **arcw0** worker is authorized. There is no unattended upload job.

## Why this batch

The [saved-metric review](headroom_v6.json) reads no new labels. Missing clocks
account for **49.6069% of seasonal SSE** and **21.8776% of forward SSE**, on 5,373
and 1,356 rows respectively. The previous generic ground-feature LightGBM
missing-record expert failed promotion; this batch tests a different model family,
an explicit core/ground weather ablation, and stronger LightGBM regularization.
The new experts retain the original conditional LIRF schedule offset.

V6's global CatBoost refit uses 5,968 trees, close to the preceding 6,000-tree
development cap. The longer comparison raises only that cap to 10,000, retaining
October stopping. Three declared comparisons also examine the existing global
and airport components with common weights across airports.

These concentrations describe reused development data. They do not establish
predictability, and slices from different grouping families overlap.

## Declared models and selection

| New component | Route | Predictors | Model |
| --- | --- | ---: | --- |
| cat_missing_core | Both NM clocks missing | 173 | CatBoost, depth 6, cap 4,000 |
| cat_missing_ground | Both NM clocks missing | 232 | Matched CatBoost with ground context |
| lgb_missing_regularized | Both NM clocks missing | 232 | LightGBM, 31 leaves, L2 100, cap 2,400 |
| cat_long | NM clock available | 202 | Original CatBoost settings, cap 10,000 |

Each component is fitted separately on both original development folds, with
October used only for stopping. Clock presence is determined entirely from
supplied covariates. Negative and extreme labels are retained.

Twelve declared families combine the new components and frozen v6 components.
Each family is blended with v6 at 50%, 75% and 100%, producing 36 complete
candidates. Families use the same weights on both folds and at all airports.
Missing and present clock routes cover every scoring row.

All eight fits and all 36 recipes must complete before selection. Promotion
requires **at least 0.5% lower complete RMSE on each original fold** than v6,
then selects the lowest equally weighted mean RMSE ratio. If no recipe passes,
v6 is retained. New refit counts are the ties-to-even rounded median of each
active component's two October-selected counts. Imported v6 refit counts remain
unchanged.

The existing NM feature bundle, `data/nmclock/72e86725e23367ede5b5`, is reused
without new shared features. Source is isolated in `src/prc2026_ladder/`.
Development components have native model/encoder files and full prediction
receipts; their v6 references are hash-bound to the original parent runs.
Any promoted refit will contain a portable complete pipeline.

## Execution and evidence

The batch launcher is restricted to `/home/azureuser/prc-flight-2026` on
`arcw0`. It runs the complete test suite and then both development folds,
one model at a time. Log: `runs/ladder/batch.log`. Azure artifacts belong in
`runs/ladder/`; local mirrors use `runs/azure_arcw0_ladder/`.

The batch launched at **15:10:55 UTC**, PID **44463**. The original **130 tests**
and five subsequent selection/refit guard tests pass on both Windows and Azure.
The three seasonal fallback fits regress v6 individually. All remaining fits
and common-weight evaluations have finished; the selected candidate is null.
Azure verification checks all 2,264,211 inherited feature rows, eight complete
native scores, 72 blend-fold scores and native samples, with zero prediction
difference. Windows independently passes the same checks with zero native
difference. Recomputed complete RMSE differs between platforms by at most
1.14e-13 seconds.

The [controlled-comparison audit](control_comparison.json) confirms identical
fitting/stopping IDs and encoders for the longer CatBoost experiment. Both
learning curves reproduce every value of the original 6,000-tree prefix exactly;
only the cap changed. The longer counts are 9,892 seasonal and 9,966 forward.
The eight fits took 2,952.12 seconds in total, with 44,723.63 CPU seconds and
11.23 GiB maximum sampled RSS. No additional worker was started.

The frozen model-code SHA-256 is
`137bd8c306d044887cbcb6fdd673cddead97c5d81c5156c0d8a1ade6c820f6e6`.
The [initial package receipt](code_package_initial.json) and
[supporting-tools receipt](code_package_review.json) preserve the transfer hashes.
Existing model source was compared byte for byte and preserved when the
selection/refit/verification helpers were added.

`tools/select_ladder.py` freezes the complete promotion decision.
`tools/verify_ladder.py` independently checks populations, all scores and native
reloads. `tools/refit_ladder.py` requires matching Azure and Windows development
verification receipts. `tools/restore_ladder_results.py` preserves separate Azure
document snapshots and refuses to replace differing existing run artifacts.

There is no refit or submission from this batch. No December scoring or training
is authorized by this campaign. The [live capacity snapshot](submission_capacity_20260914_1600.json)
preserves v1-v6 and records the next rolling-window slot after 16:40:24.930 UTC;
capacity and organizer inputs must be rechecked before any later eligible upload.

## Independent follow-up evidence

The [same-flight coverage check](clock_context_coverage.json) reads only supplied
timestamps and matching identifiers. It finds no peer clock for any of the
20,821 missing-clock January-November departures or 5,290 ranking departures.

The [fitting-only residual review](fitting_clock_regimes.json) uses the 1,226,945
IDs shared by both fit sets. It provides no support for a general hour/day repair.
AOBT is the stronger source overall; large disagreement and missing-record
schedule regimes retain substantial residual variation. No scoring, stopping,
December or ranking labels enter that study.

A [separate target-scale declaration](../ladder_scale/README.md), made at
15:49:17 UTC, tests that evidence after this batch's complete verified decision.
It does not change the eight fits, 36 candidates or selection rule above.
