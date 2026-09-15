# Flown timeline and off-block traffic experiment

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

All eight fits and ten recipes must finish before selection. Promotion requires
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
