# V6 ladder campaign

Declared at **15:04:17 UTC on 14 September 2026**, after the user requested
continuing the ladder climb following v6's official result. V6 remains the
official best at **277.4508 seconds RMSE on all 344,841 rows**. The previous
models, source packages, feature bundle and submissions remain frozen.

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

The declared training code is frozen once a run exists. Selection, independent
verification, refit and submission evidence will be added here as they complete.
No December scoring or training is authorized by this campaign.
