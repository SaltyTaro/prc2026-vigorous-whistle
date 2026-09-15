# Ladder target-scale follow-up

This separate batch was declared at **15:49:17 UTC on 14 September 2026**, before
any new target-scale development score. It follows the first ladder campaign and
uses only the existing **arcw0** worker. December remains reserved.

The [declaration](declaration.json) and
[configuration](../../configs/ladder_scale_v1.json) fix eight LightGBM fits and
twenty complete candidate recipes. The preceding ladder decision must be complete
and verified before these fits start. A candidate must improve the incumbent by
at least 0.5% on **both** original complete folds. The incumbent is the first
ladder's eligible selection, or v6 if it has none.

## Evidence and hypothesis

The [fitting-only clock review](../ladder/fitting_clock_regimes.json) describes
1,226,945 rows shared by both fitting sets. It reads no scoring, stopping,
December or hidden ranking labels. AOBT is generally stronger than LOBT, but
large source disagreements and missing-record schedule offsets still produce
large residuals. The [same-flight coverage review](../ladder/clock_context_coverage.json)
found no recoverable clocks among missing departure records.

The experiment changes the residual's scale using supplied covariates:

```text
z = (target_seconds - offset_seconds) / scale_seconds
weight = (scale_seconds / 300)^2
prediction_seconds = offset_seconds + scale_seconds * model_z
```

Thus weighted squared residual error equals squared error in seconds divided by
300 squared. October stopping explicitly evaluates complete routed RMSE in
seconds. Regularization acts on the transformed model; this is an experiment,
not an assertion that a transformed learner must improve.

For present clocks the offset remains AOBT, then LOBT. The scale is the larger
of absolute AOBT/LOBT disagreement and a declared 60- or 300-second floor. Missing
disagreements receive the floor. For missing clocks the original conditional
LIRF schedule offset is retained; scale is the larger of absolute schedule
interval and a declared 300- or 900-second floor. No target or prediction is
clipped and every routed fitting label is retained.

## Models and recipes

Two global clock models use the original 202 core/weather/NM predictors and
the original LightGBM settings. Two missing-clock experts use 232 ground/weather
predictors and the first ladder's regularized LightGBM settings. The scale is
computed inside the model and never added as a supervised shared feature.

Five families comprise each of the four new alternatives plus one common
clock-300/missing-300 combination. Present-clock families retain a fixed
50% new-global/50% original-v6-airport mixture. Each family is combined with
the full frozen v6 prediction at 25%, 50%, 75% and 100%. Weights are common
across airports and folds; all twenty recipes must finish before selection.

New refit tree counts are the ties-to-even rounded median of each active
model's two October-selected counts. Imported v6 models retain their original
native files and counts. Any refit excludes every December flight group and
must pass Azure and Windows development verification first.

## Reproduction and preservation

Source is isolated in `src/prc2026_ladder_scale/`; the original ladder and all
earlier model/feature packages remain frozen. The existing NM bundle
`data/nmclock/72e86725e23367ede5b5` is reused. No new external data is fetched.

The twelve new tests check the exact physical loss identity, complete routed
populations, October's physical metric, native reloads and portable blend math.
Full-suite platform results and execution receipts are recorded as they finish.

Azure runs use `runs/ladder_scale/`; local mirrors use
`runs/azure_arcw0_ladder_scale/`. The launcher runs both folds sequentially,
then selection, independent verification, summary and an immutable results
archive. It never uploads a submission.

