# Search for the next lever after v6

The search found a qualifying improvement: **surface activity and flown-timing
features applied to both global and airport models**. The selected recipe scores
**322.8193 seasonal / 221.1382 forward**, improving v6's complete development
RMSE by **0.6456% / 0.9802%**. All ten airports improve on both folds. Its
[separate refit](../surface_refit/README.md) is complete at the frozen tree counts.
Both final verifiers reproduce all 344,841 ranking predictions exactly.
**V7 is exported and ready for submission**; its official effect is unmeasured.
The [campaign result](CAMPAIGN_RESULT.md) summarizes the complete evidence.

V6 remains the official best at **277.4508 seconds RMSE**. The dated leaderboard
[snapshot](leaderboard_snapshot_final.json) at 19:18:25 UTC on 14 September places it 21st of 135, with third at
252.9192. Reaching that dated score requires another 8.84% RMSE reduction.
These investigations use the original development protocol and keep December
reserved. They do not infer hidden labels from leaderboard submissions.

All **32 declared development fits and 22 fixed recipes** are complete across
four declarations. One additional initial timeline fit attempt is preserved;
it failed while writing provenance, before a complete development score.
The [development summary](development_summary.json) binds the complete results,
both platform verifiers, archive hashes and test evidence.

| Investigation | New native fits | Fixed recipes | Main question |
| --- | ---: | ---: | --- |
| [Flown timeline and surface activity](../timeline/README.md) | 8 | 10 | Do flown arrival clocks and other flights' actual surface activity improve NM-clock residual predictions? |
| [Supplied arrival taxi-in context](../arrivalctx/README.md) | 4 | 6 | Do completed arrivals add information for departures missing an NM clock? |
| [Complementary route combinations](../nextlever_joint/README.md) | 0 | 2 | Do the predeclared equal learner means combine into an improvement across the full dataset? |
| [Frozen timeline features in airport models](../surface_airports/README.md) | 20 | 4 | Does extending the useful global feature signal to every airport model clear the promotion margin? |

Only the existing arcw0 trains competition models, sequentially. Each declaration
requires every candidate to finish before selection and the same 0.5% improvement
margin on both complete folds. Signed and extreme labels stay in the metric.
The four declarations preserve every earlier campaign. The first three found
no eligible candidate; the airport extension has two eligible recipes and
selects the full-weight one. Both platform verifiers pass with zero native
prediction differences, and all [196 current test identities](tests_combined.json)
have passing records on each platform. No new official submission has been
made and December remains reserved.

The successful lever addresses ordinary taxi-outs. Useful context includes
other flights' NM pushbacks, active ground movements, recent completed NM taxi
proxies, and recent activity at the departure's stand. The controlled global
comparisons do not support using flown-arrival timing alone as the next model;
surface context and the combined contract are more useful. Feature importance
is descriptive and does not isolate causal effects. Applying the combined
features to the airport models produces the qualifying complete recipe.

The [flown-clock audit](flown_clock_audit.json) used all 1,226,945 common fitting
departures. Its raw NM-proxy residual correlations motivate a controlled feature
comparison, rather than establish additional value over v6. The fresh
[official-source review](field_use_review.json) and the
[timeline availability policy](timeline_availability.json) document the
post-operations interpretation of supplied timestamps and other-flight events.

The [arrival-context audit](arrival_context_audit.json) covers 111,177 common
fitting departures, including all 12,179 with missing NM clocks. Same-stand and
same-runway taxi-in means have within-airport rank correlations of approximately
0.233 and 0.254 with the missing-clock raw offset residual. Stand mean context
covers 342,148 of 344,841 ranking departures. This is not a v6 prediction-error
comparison. The completed native experiment worsens seasonal RMSE for every
declared recipe and is excluded from the selected model. Only supplied arrivals completed strictly before each query enter
the summaries; focal flight IDs and actual December context are excluded.

Conservative callsign/route matching to supplied arrival records was also
investigated without changing any prediction. The
[coverage audit](clock_relink_coverage.json) found 344 unique possible matches
among 5,290 ranking departures missing NM clocks. A 30-minute schedule-agreement
condition retains 271; requiring a known matching aircraft type retains 244.
However, three ranking positive-control matches still point to a different NM
flight ID and clock, even with those conditions. These are checks against supplied
covariates, with no hidden target access. The
[quality audit](clock_relink_quality.json) therefore does not support automatic
clock substitution. Raw arithmetic errors on fitting rows are distinct from
improvement over v6. This branch remains an audit, not a submitted model.
