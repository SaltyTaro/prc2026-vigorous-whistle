# Flown timeline and off-block traffic experiment

All eight development fits and ten fixed recipes are complete. None meets
the required 0.5% improvement on both complete folds. The leading recipe,
`combined_mean_v6_100`, scores **323.4327816874519 seasonal** and
**221.88908249663194 forward**, improving v6 by **0.4568% / 0.6439%**.
The original selection remains null; this batch made no refit or export.
The separately declared [airport extension](../surface_airports/README.md)
later applied these frozen features to the airport models and qualified.

Declared at **17:07:27 UTC on 14 September 2026**, before any new timeline
development score. The user requested the next lever to reduce the gap after
the completed ladder campaign retained v6. Only the existing arcw0 is used;
December remains reserved.

The [configuration](../../configs/timeline_v1.json) declares **eight native
fits and ten fixed candidate recipes**. The four models are LightGBM with flown
arrival features, LightGBM with surface context, LightGBM with both, and CatBoost
with both. Native settings match the original NM-clock learners. Each new
global model retains the original NM residual target and fits only clock-present
rows; matching original v6 airport models provide half of the candidate prediction.
The fifth family averages the two combined globals while retaining 50% airports.
Every family uses 50% or 100% new-candidate weight against complete v6. Missing
clock rows retain v6 exactly.

All eight fits and ten recipes finished before selection. Promotion requires
at least 0.5% lower complete RMSE on **both** original folds than v6, then the
lowest equally weighted mean relative RMSE. This remains a reused development
evaluation, distinct from official scoring and the reserved audit.

The [fitting-only audit](../next_lever/flown_clock_audit.json) uses the 1,226,945
rows common to both fitting sets. The supplied flown-arrival delay correlates
with the airport-centered raw NM proxy residual (Pearson 0.2950, Spearman 0.3301).
This does not establish additional value beyond existing predictors. The new
field covers 339,551 of 344,841 ranking departures, matching NM clock coverage.

The [new availability review](../next_lever/timeline_availability.json) and
[fresh official sources](../next_lever/field_use_review.json) describe the
retrospective interpretation. `ARVT_3_flt` is a supplied flown-trajectory arrival
timestamp. Its use here is separate from the earlier packages' frozen exclusion. No external message was sent.

The [feature declaration](../../configs/features_timeline_v1.json) preserves
all 261 parent values and adds **14 flown-timeline** and **60 surface-context**
features. Ablations use 216, 262 and 276 predictors respectively, starting from
the same 202-feature NM model contract. Surface features measure other-flight
NM pushbacks, active ground intervals, recently completed NM taxi proxies,
same-stand movements, and movements during the estimated taxi interval.

Completed-event summaries exclude simultaneous and future takeoffs. Pushback
summaries use offblock events strictly before each query and exclude the focal
movement. Active-ground features explicitly use the supplied retrospective
takeoff endpoint and a positive, at-most-six-hour NM interval gate, including
for departures that take off later. This is not a live-availability claim.
Query anchors are takeoff and valid NM offblock, falling back to takeoff minus
twenty minutes for context only. Original proxy offsets remain signed and
unbounded. No departure airport block time, departure label, or arrival block
time enters the new raw projection. December context is excluded by actual UTC
movement date; original protected flight groups and complete scoring IDs remain.

Source is isolated in `src/prc2026_timeline/`. Existing model, feature,
configuration and submission artifacts remain unchanged. New runs belong in
`runs/timeline/`, and mirrors in `runs/azure_arcw0_timeline/`. The batch runs tests,
prepares the new bundle, and fits both folds sequentially. It never uploads.

The bundle is `data/timeline/ce42240e5e43eee1713f`, full ID
`ce42240e5e43eee1713f1f8dbc23cfa4511076e510bdc9aec6853fad30cf134c`.
Preparation retained 2,264,211 departure rows and all parent values exactly,
using 108.66 seconds and 6.87 GiB sampled peak RSS. Independent direct event scans
pass before the resumed training; the report is `feature_verification.json`.

The initial batch launched at 17:10:10 UTC. Its first LightGBM fit finished,
but a missing `ID` import prevented writing the provenance card, before any
complete new development score. The [recorded correction](import_fix/correction.json)
adds that import only, preserves the original source and partial native files,
and repeats the fit in a new directory. Features, stopping, settings and the
declared recipes are unchanged. A complete fit/provenance/reload test now covers
this path; all 166 distinct tests pass on Windows and Azure.

The resumed batch launched at **17:19:56 UTC**, PID **52110**, log
`runs/timeline/resumed.log`. Its model-code SHA-256 is
`3f7f148faf5a4b4b33382b2b66837c007e5520b1abe3718e37f99487977ed584`.
The original failed package and partial run remain separately recorded. The
repeated LightGBM model, encoder and feature-importance files are byte-identical
to the preserved first attempt. The October curve differs only by 5.69e-14
seconds of numerical rounding. Count eight completed declared fits and one
additional preserved fit attempt that failed while writing provenance.

The [Azure verifier](verification.json) and [Windows verifier](local_verification.json)
pass all complete scores and native prediction checks with zero native difference.
Both independent feature checks reproduce 10,656 new feature values exactly and
verify all 2,264,211 inherited feature rows. The [complete results](development_results.json)
and [matched comparisons and diagnostics](analysis.json) retain all ten recipes.

The [development mirror](development_mirror.json) contains 127 verified files,
171,374,259 archive bytes, SHA-256
`de8ff34f1db12bf69109602e221cc8c576d40ceb025cb92ef8ebddd3e49ec6b2`.
The feature archive has 717,332,738 bytes and SHA-256
`d480548d9ad92b1091659114f516a251d129462aacdd6c3895cc6abfa1c10a51`.
The completed native fits used 3,187.57 seconds in total and peaked at 13.55 GiB
sampled process RSS. These totals exclude the preserved first attempt.
