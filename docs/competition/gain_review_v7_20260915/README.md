# Where to seek a larger gain after v7

V7 officially
scores **275.0570 seconds RMSE** on 344,841 departures. The last saved third-place
score is **252.9192**, requiring **8.0484% less RMSE**, equivalent to **15.4491%
less mean squared error**. These are dated standings, not a final cutoff.

The next search should focus on **the off-block reporting mechanisms at LIRF
and LFPG**, particularly missing NM records, and **actual taxi-path geometry
combined with runway queues**. Further gains are unmeasured. The large v5-to-v6
official improvement, 14.0824%, came from introducing NM off-block information;
v7's broader traffic representation added 0.8628%.

## What the current error budget establishes

These are aggregates from the selected v7 recipe on the original complete
development folds. They do not describe the unavailable ranking labels.

| Slice | Seasonal share of squared error | November share of squared error |
| --- | ---: | ---: |
| LIRF and LFPG together | 63.62% | 46.45% |
| Missing NM record | 50.25% | 22.31% |
| Taxi-out from zero to one hour | 55.99% | 75.18% |
| Taxi-out above three hours | 33.87% | 15.29% |

The airport and missing-record slices overlap the duration slices; their shares
must not be added. Missing records are only 1.56% of seasonal rows and 0.8353%
of November rows. All 5,290 ranking departures missing clocks, 1.5340% of ranking
rows, retain the frozen v6 fallback.

As an arithmetic scenario, halving squared error on missing-NM rows while
holding every other prediction fixed would reduce complete RMSE by **13.47%
seasonally and 5.74% in November**. Achieving an 8.0484% complete improvement
through that slice alone would require **69.24% less slice squared error** in
November. This is a demanding requirement, not evidence that it is achievable.

Reducing squared error on all clock-present November rows by 20% would instead
lower complete RMSE by **8.10%**. Perfect predictions on every above-three-hour
November case would lower complete RMSE by at most **7.96%**. Ordinary departures
therefore remain essential. These calculations use different development error
mixes and must not be converted into promised official scores.

## Recommended investigations

1. **Establish the recurring reporting mechanisms at LIRF/LFPG.** Start with
   fitting-only evidence for missing joins, disagreement between clocks and
   schedule, and repeated reporting patterns. Distinguish ordinary taxi-outs
   from records following a different timing convention only when observable
   inputs support the distinction. A model could then learn the probability
   of each supported interpretation and its conditional prediction. Existing
   models already contain the clock differences; simply adding them again is
   not a new experiment. Broadening the old LIRF schedule gate remains
   unsupported, and relinking by callsign/route already produced false matches.
   Large error concentration does not establish that the exceptional cases
   are recoverable.

2. **Pilot real taxi-path geometry and ordered queues at those airports.**
   First measure exact stand-code coverage against a documented, openly
   licensed layout, including whether it represents the competition periods.
   Candidate information includes stand-to-runway path length, runway crossings,
   terminal exits and routes shared by departing aircraft. Combine this with
   the order of aircraft ahead at pushback, runway service and wake-separation
   context. The target is travel time plus waiting time. V7 already measures
   pushbacks, active aircraft and recent traffic in windows, so another set of
   similar counts has weaker justification. Phase-5R empirical stand/runway
   baselines also failed promotion; actual geometry is the untested difference.
   OurAirports runway coordinates alone do not supply the required graph.

3. **Use more fitting data to teach the missing-clock fallback.** A bounded
   secondary hypothesis is to train with the NM fields jointly hidden on some
   otherwise complete fitting rows, rebuilding every derived feature under the
   same missing-information contract. The earlier dedicated fallback experts
   fitted only naturally missing rows. Synthetic missingness could increase
   the examples available for learning ordinary taxi-out behavior, but natural
   NM absence is not random and artificial masking does not reproduce its
   exceptional reporting conventions. It requires a controlled comparison
   against v7 and does not justify an expected gain in advance.

The first recommendation is a diagnostic priority. The geometry coverage check
is the clearest remaining search for new physical information. A useful result
must lead to a new declaration and improve both complete development folds
against v7 before any refit or submission. Learned intermediate predictions
must use inner fitting folds grouped by flight; no rule may be chosen around
individual scoring rows. December remains reserved.

The completed ladder's 16 fits and 64 recipes did not qualify. Extra model
length, seed/blend variations, target scaling, generic missing-NM experts and
arrival taxi-in summaries have weaker evidence as the main route to a large
gain. The current review starts no training, submission or additional worker.

## Evidence

- [Recomputed v7 error budget and arithmetic scenarios](error_headroom.json).
- [Frozen surface selection](../../surface_airports/development_results.json).
- [Official v7 submission](../../surface_refit/submission_v7_official.json).
- [Completed next-lever search](../../next_lever/CAMPAIGN_RESULT.md).
- [Completed ladder search](../../ladder/CAMPAIGN_RESULT.md).
- [Fitting-only clock evidence](../../ladder/fitting_clock_regimes.json).
- [Clock relinking quality audit](../../next_lever/clock_relink_quality.json).
- [Earlier research and geometry source notes](../history_20260914/README.md).

`tools/research_prc_v7_gains.py` derives the calculations from frozen aggregate
JSON only, verifies partition totals and receipt hashes, and preserves the
original review file. No additional labels, fitting rows or reserved periods
were read. No new model was fitted or scored, and no prediction was changed.
