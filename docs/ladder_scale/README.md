# Ladder target-scale follow-up

This separate batch was declared at **15:49:17 UTC on 14 September 2026**, before
any new target-scale development score. It follows the first ladder campaign and
uses only the existing **arcw0** worker. December remains reserved.

**All eight fits and twenty candidate recipes are complete. Nine improve both
folds, but none meets the declared 0.5% improvement margin on each fold. V6 is
retained.** The leading mean-ratio recipe, `clock300_missing300_v6_25`, scores
**322.405169 seasonal / 222.687699 forward**, improving v6 development by
**0.7731% / 0.2863%**. These are complete reused development scores. No refit,
submission export or upload was made.

The [selection](development_results.json), [analysis](analysis.json),
[Azure verification](verification.json), [Windows verification](local_verification.json),
[platform comparison](verification_comparison.json), and [completion](completion.json)
preserve the decision. Both verifiers reproduce native predictions and independently
reconstruct the covariate scales with **zero difference**. Complete RMSE differs
between platforms by at most 1.14e-13 seconds. The later
[cross-campaign check](../ladder_blend/README.md) also selects no candidate;
the [aggregate campaign result](../ladder/CAMPAIGN_RESULT.md) retains v6.

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
Two further guards enforce the margin against the preceding incumbent. All
149 tests pass on Windows and Azure; the reviewed 14-test subset also passes
locally. The batch launched at **16:06:44 UTC**, PID **47273**, after the first
ladder's verification completed. Its incumbent is v6 because the first batch
selected no candidate. Log: `runs/ladder_scale/batch.log`.

Model-code SHA-256:
`2c5ee07eea80711b07704e4c68d552161073802456101ea026be8cabb236fe53`.
The initial and reviewed [package receipts](code_package_initial.json) preserve
the transferred source. The [supporting review](supporting_review.json) adds
independent scale reconstruction and coverage of every large clock disagreement;
it changes no training source or configuration.

Azure runs use `runs/ladder_scale/`; local mirrors use
`runs/azure_arcw0_ladder_scale/`. The launcher runs both folds sequentially,
then selection, independent verification, summary and an immutable results
archive. It never uploads a submission.

The eight fits used 566.91 wall seconds, 7,970.76 CPU seconds, and at most
9.71 GiB sampled RSS. All fitting and verification jobs have finished.
Selection SHA-256:
`c23931559347114ec0b529050e1d51991afb439e781b7bfd2e5e431ec585ac19`.
The [results archive](development_archive.json) contains 113,687,436 bytes,
SHA-256 `bb4b8651dc7d02eac09a9114e36c9cb160c6e6ed206f3e1072fec90d70b4811d`;
all 110 files are restored and hash-checked locally.
