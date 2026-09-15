# Where a substantial improvement could come from after v5

Reviewed on 14 September 2026 after the weather campaign. The strongest
untested hypothesis is to reconstruct taxi time from the supplied Network
Manager off-block clocks and learn the difference between those clocks and the
airport reporting source. Its development gain is unmeasured. This review does
not declare or run another model campaign.

V5 scores 322.9266 seconds officially. The saved 03:13 UTC standings have third
at 256.3864: another 66.5402 seconds, 20.61% RMSE or 36.96% mean squared error.
This is a dated benchmark, not a forecast of the final qualifying score.

## What the remaining v5 error says

These are shares of squared error on the complete, unchanged development sets.
They are not a decomposition of the hidden leaderboard score.

| Slice | Seasonal error share | Forward error share |
| --- | ---: | ---: |
| Rome Fiumicino (LIRF) and Paris CDG (LFPG) | 66.28% | 42.58% |
| Missing Network Manager origin | 39.27% | 18.84% |
| Target above three hours | 41.74% | 12.91% |

These groups overlap and their shares must not be added. Two records, the
largest error in each of LIRF and LFPG, contribute 30.30% of seasonal squared
error. That concentration is a reason to investigate how the records were
produced; it does not establish that their labels are inferable from predictors.
Corrections must have fitting-period support and cannot be rules for individual
held-out IDs.

Even perfect predictions on every above-three-hour case, with every other
prediction unchanged, would improve forward RMSE by only 6.68%. Routine
departures also need better predictions. No evaluation records are removed in
that upper-bound calculation.

Calculations, protected evaluation-ID hashes and input receipts are in
[error_headroom.json](error_headroom.json). Reproduce with
`python tools/research_prc_v5_gains.py`; it reads saved aggregate metrics only.

## Recommended order

1. **Test the NM off-block clocks as a new information source.**
   `AOBT_3_flt` and `LOBT_flt` are populated for 339,551 of 344,841 ranking
   departures (98.47%). The target can be represented as takeoff time minus
   NM off-block time plus a learned correction. Compare ordinary feature
   augmentation with residual prediction, using v5 as the fallback where the
   clocks are missing. Learn source disagreement from airport, timing,
   runway/stand, aircraft and reporting-consistency information. A raw clock
   subtraction is not an exact target: the original arithmetic audit found
   substantial airport-dependent errors. These fields also permit congestion
   measurements closer to recorded pushback, beyond the fixed 10/20-minute
   earlier queries already tested. Any successful new feature set should reach
   the airport specialists as well as the global model.

2. **Investigate the source and date inconsistencies at LIRF/LFPG.**
   Inspect fitting-period evidence for day offsets, contradictory clocks,
   duplicate or mismatched flight joins and missing source records. A rule or
   expert needs an observable trigger and measured false-positive cost on
   ordinary rows. The generic missing-NM expert already failed to improve the
   two complete folds in Phase 5R; another expert needs additional information
   or a supported measurement model. Broadening the existing LIRF schedule
   offset is also unsupported by the fitting-only evidence already collected.

3. **Test actual stand-to-runway geometry if its coverage is sufficient.**
   OurAirports supplies runway coordinates and length, which v5 already uses.
   Actual stand locations and taxiway graph distances remain untested. An
   openly licensed source such as OpenStreetMap could support an airport
   coverage study, followed by an unimpeded travel-time plus waiting-time
   model. Check stand-code matching before building a large graph pipeline.
   Phase-3 stand-code features and Phase-5R empirical stand/runway baselines
   do not establish the value of real path geometry.

The first clock comparisons fit the current arcw0 CPU allocation. Additional
workers are not needed for this screen. Keep the original scoring IDs, October
early stopping, complete-set metrics and December's independent audit. Declare
the experiment and create a fresh feature package/bundle before fitting; do
not change v5's frozen source, predictions or receipts.

## Field-use evidence

The [official data page](https://prc-data-challenge-2026.netlify.app/data.html)
documents both off-block fields. The [official rationale](https://prc-data-challenge-2026.netlify.app/rationale.html)
describes post-operations analysis. Our latest archived public-page check found
no field-specific prohibition. The existing exclusion was a temporary project decision, not a
verified organizer ban; it should be reviewed explicitly for the proposed new
experiment. The user has already said they do not know of an announcement.
No new clarification request or administrator message is part of this review.
See the [source receipts](../field_use_review_20260914.json) and
[observed schema](../DATA_SCHEMA.md).

## What has already been tried

Phase-5 tuning and blending improved official RMSE by 1.58%; weather added
another 1.15%. Phase 5R's demand/service features, empirical taxi baselines,
target scaling and generic missing-record expert produced no replacement for
v4. Those completed experiments have less claim to a major new gain than an
unmeasured, direct timing signal. No numerical gain is assumed for the proposed
clock or geometry experiments.

The useful lesson from the earlier winners remains to estimate the uncertain
part of a measured operational quantity. The 2024 winner used a physical mass
representation; the 2025 winner estimated fuel flow and reconstructed consumed
fuel. For 2026, a clock-based taxi estimate with a learned source correction is
a closer application of that idea than another generic parameter search.
The earlier targets were different, so their reported gains do not predict
ours. See the [historical review](../history_20260914/README.md).
