# PRC 2026 project plan

Proposed on **13 September 2026** for team **vigorous-whistle**. Objective:
top three. Compute: **Azure only, maximum eight VMs**. The published deadline
is **11 October 2026, 23:59:59 CET**; target completion by **10 October**.

Initial research, phases 1-5, the initial Phase-5R campaign, weather and NM-clock
campaigns are complete as of **14 September UTC**. The best official submission,
**v6**, combines CatBoost global and LightGBM airport NM-clock residual models,
with v5 fallback for missing clocks. It scores **277.4508 seconds RMSE on all
344,841 rows**, improving v5 by **14.08%**; see [NM-clock results](../nmclock/README.md).
The team ranks **20th of 134** at 16:38:49 UTC; third place scores 252.9192,
requiring a further **8.84%** reduction to that dated score. December remains reserved.
Phases 6-7 remain a plan. Dates are working
targets and can move if a phase finishes early or an experiment fails to justify
more work. Keep the final completion buffer.

The [historical review](history_20260914/README.md)
is complete and recommends a further development campaign before the final audit:
clock/source experts, an operational taxi baseline, ground demand/runway-service
features, then bounded weather/geometry experiments. The initial
[Phase-5R campaign](../phase5r/README.md) completed on **14 September UTC**:
ten new fits and twenty recipes, with none improving both complete folds.
The matched LightGBM comparison improves 0.50%/0.67%, but that batch produced
no replacement for v4. A separate [weather coverage audit](weather_20260914/README.md) prepared
190,466 public-domain observations with 99.99% departure coverage. The separately
declared [weather campaign](../weather/README.md) completed six fits and sixteen
common blend recipes on arcw0. Its 61 new weather/runway-wind features retain
all prior values and every scoring ID. The selected pipeline improves both
complete folds: **365.1711 seasonal / 240.6522 forward RMSE**. Refit, verification
and official v5 scoring are complete; 102 tests pass on each platform. The v4
recipe and all Phase-5 receipts remain frozen. At **03:13 UTC on 14 September**,
the team ranks **68th of 133** and third place scores **256.3864**. Reaching that
dated score needs a further **20.61% RMSE reduction**, so the competitive
objective remains outstanding.

The user then approved the [NM off-block clock campaign](../nmclock/README.md).
Its 03:47 UTC declaration fixes four direct/residual model recipes on both folds,
twenty common blend candidates and a 0.5% improvement requirement on each fold.
The new contract adds 29 supplied clock predictors and retains v5 for every row
without a clock. All eight runs and twenty
candidate comparisons are complete. Selected `cat_residual_v5_100` scores
**324.9171 seasonal / 223.3271 forward**, improving v5 development by
**11.02% / 7.20%**. Both platform verifiers reproduce the full scores and native
predictions; 117 tests pass per platform. The fixed January-November refit began
at 05:22 UTC on arcw0 and completed at 05:44 UTC. Both final verifiers reproduce
6,450 ranking samples and the complete missing-clock fallback exactly. The
**v6 export is validated for all 344,841 rows**. The complete submission
receipt verifies the uploaded bytes and preserves prior versions. December remains
reserved. Exact artifacts, scoring evidence and standings are in the campaign report.

The [subsequent ladder campaign](../ladder/CAMPAIGN_RESULT.md) is also complete:
sixteen new native fits and 64 declared recipes across longer/fallback models,
covariate-scaled residuals and fixed cross-campaign mixtures. None meets the
unchanged 0.5% improvement margin on both original folds. The closest recipe
improves seasonal/forward RMSE by 0.5873% / 0.4678%, so v6 remains the incumbent
without another refit, export or upload. All three platform verifications and
153 distinct tests pass on Windows and Azure. December remains reserved.

## Phase schedule

| Phase | Target dates | Main work | Completion milestone |
| --- | --- | --- | --- |
| **0. Understand the competition** | 13 Sep — complete | Retrieve rules, data, field dictionary, leaderboard, and public research; inspect all 14 files | Research pack and verified input hashes available |
| **2. Establish strong baselines** | Completed 13 Sep; originally 16–18 Sep | Compare airport means, LightGBM, and CatBoost; build and validate a complete submission file | v1 scored 436.8976 RMSE on all rows; reproducible baseline and local-to-official comparison recorded |
| **3. Improve airport and traffic features** | Completed 13 Sep; originally 19-23 Sep | Tested eight combinations of airport/layout, schedule timing, past traffic, and completed-arrival statistics on both folds | Timing + traffic selected: 475.1669 seasonal / 311.5701 forward RMSE; official v2 421.3047; 61 tests pass on Windows/Azure |
| **4. Address difficult records and airport differences** | Completed 13 Sep; originally 24-28 Sep | Tested record-quality/clock features, conditional schedule offsets, and airport specialists; weather/geometry deferred | Fixed global/airport blend: 372.8550 seasonal / 245.8627 forward; official v3 331.9269; 71 tests and both platform verifiers pass |
| **5. Tune and combine the strongest models** | Completed 13 Sep UTC; originally 29 Sep–4 Oct | Tested six initial recipes, two selected seeds, one declared CatBoost cap follow-up, and 28 individual/blend recipes | Selected blend: 367.0843 seasonal / 244.5549 forward; official v4 326.6895; 81 tests and both platform verifiers pass |
| **5R. Research substantial new gains** | Historical review and initial experiment batch completed 14 Sep | Tested demand/service features, cross-fitted stand/runway baselines, target scaling, and a missing-NM expert | Ten fits and twenty recipes; none improve both folds, so this batch retained v4. 91 tests pass |
| **Weather. Test surface weather and runway wind** | Completed 14 Sep | Added 61 features from licensed IEM observations and OurAirports runway data; tested six fits and sixteen recipes | Selected CatBoost/airport blend: 365.1711 seasonal / 240.6522 forward; official v5 322.9266. Refit and both full verifiers pass; 102 tests pass per platform |
| **NM clocks. Test supplied off-block information** | Completed 14 Sep, including official scoring | Four direct/residual pipelines on both frozen folds; 29 new features; v5 fallback for missing clocks | Selected 324.9171 seasonal / 223.3271 forward; both final verifiers and 117 tests per platform pass; official v6 277.4508 on all rows, 20th of 134 at 14:52:58 UTC |
| **Ladder. Test capacity, fallbacks, residual scaling and mixtures** | Completed 14 Sep | Sixteen new native fits and 64 declared recipes across three separate batches | No candidate meets the two-fold 0.5% margin; v6 retained. All three verifiers and 153 distinct tests pass on both platforms; no new refit or upload |
| **6. Audit and reproduce the selected recipe** | 5–8 Oct | Evaluate the reserved period, check error concentration and feature availability, reproduce from raw files, and prepare GPLv3 source/documentation | A reproducible final candidate with documented validation and exact submission checks |
| **7. Finish and verify scoring** | 9–10 Oct | Refit the selected recipe on eligible full training data, generate all 344,841 predictions, submit, confirm scoring, and complete the public release | A successfully scored final version linked to code, configuration, data hashes, and model artifacts |

**11 October is contingency time** for processing or technical problems. Do not
schedule a major new experiment or first reproduction attempt for the last day.
There is a one-hour CET/CEST ambiguity in the published wording, so finishing
on 10 October also avoids relying on its interpretation.

## Validation design comes first

Use January/July 2025 development evaluations to test the seasonal mixture,
alongside a chronological evaluation to test forward transfer. A seasonal split
that trains on later months is not a forward-time test; report the two separately.
December 2025 is now reserved from model selection; score it after freezing the
candidate recipe. The initial descriptive audit covered all months. Airport-mean,
LightGBM, and CatBoost scores have been inspected on the seasonal and forward
development folds; no December model score has been inspected.

Choose early-stopping data inside the development protocol. Do not tune on the
reserved audit period. Keep flight groups separated where required and ensure
target encodings, group means, and any supervised preprocessing use only the
fitting rows of each split.

For held-out departures, mask airport block time and the target before feature
generation, including neighbor and traffic joins. Keep arrival context distinct
from departure targets. Measure overall RMSE, per-airport RMSE, monthly scores,
missing-record slices, and the contribution of unusually large errors.

Score the **complete** validation set, including negative and extreme labels.
Training filters, clipping, robust objectives, or special treatment are experiments
to compare on that same evaluation population. A gain should remain useful across
the seasonal and chronological checks; inspect changes dominated by one or two
records and use paired day-level uncertainty estimates when differences are small.

## Where the improvement effort goes

Start with the supplied data and features that are cheap to compute. It contains
2,085,047 labeled departures and rich airport/runway/stand information. Preserve
the arrival movements for traffic context. Airport and runway keys must use the
correct movement phase.

The active user-approved NM campaign tests timestamp differences and residual
models under its documented official-source interpretation. The audit found strong
coverage but substantial airport-dependent timestamp errors. Missing NM records
retain v5; this route has 5,290 ranking rows. Authenticated organizer announcements
remain unverified. No outgoing organizer message is part of this work.

Prioritize the difficult regimes before a large parameter search. Of 202 training
labels above three hours, 188 occur at LIRF. Determine whether their patterns can
be learned across periods before introducing specialized models. Avoid copying
another team's rules or fitting corrections to leaderboard feedback.

Try external weather and geometry only as bounded experiments with documented
licenses, airport/month coverage, and validation gains. If a source adds complexity
without a repeatable improvement, stop that branch. GPU experiments are optional
and should follow a measured CPU bottleneck or a promising model hypothesis.

## Azure allocation

Phases **1-5, the initial 5R campaign, and weather used one VM**. Expansion toward four to eight workers in later
phases is useful only when enough independent experiments and measured runtime
justify it. Reduce capacity during the final audit and release. Actual worker
counts depend on measured resource use and the user's current allocation.

The user has now allocated `arcw0`–`arcw7` and authorized starting **only arcw0
for now**. All eight are `Standard_D16ds_v7` in `eastus`. Additional worker
starts require a change to that current instruction. Phase 1 used arcw0 only.
Phase 2 also used arcw0 only: LightGBM runs took 14–17 seconds including
loading/scoring, and CatBoost took 93–117 seconds with peak RSS below 4 GiB.
Phase 3 completed sixteen development runs in 27-52 seconds each on arcw0,
with feature preparation peaking at 4.61 GiB. Its selected model scored
421.3047 officially. Phase 4's selected refit took 147.46 seconds and peaked
at 3.99 GiB, scoring 331.9269 officially. These measurements support continuing
on the single authorized worker for bounded Phase-5 experiments. Phase 5's
selected 34-model refit took 1,246.13 seconds and 7.06 GiB peak RSS. Only arcw0
was used, and arcw1-arcw7 remained deallocated at the final check.

Phase 5R's ten new fits total 956.60 seconds, including their own transforms
and predictions, with peak RSS of 6.52 GiB. Feature preparation takes 81.77
seconds and 4.84 GiB. The single authorized worker remains sufficient for
these bounded comparisons. No new refit or submission was promoted by that batch.

The subsequent weather campaign's six fits total 2,126.16 seconds and peak at
7.29 GiB. The selected refit takes 1,155.32 seconds and 8.08 GiB, using 16 CPU
threads on arcw0. No additional workers were started. These measurements still
support the single authorized VM for bounded experiments.

An example at full utilization:

| VM | Role |
| --- | --- |
| 1 | Shared feature preparation, input manifests, and experiment coordination |
| 2–3 | LightGBM variants and feature ablations |
| 4–5 | CatBoost variants and feature ablations |
| 6 | LIRF, missing-record, and other airport specialists |
| 7 | Weather/geometry experiments or an additional training worker |
| 8 | Validation, ensemble evaluation, and reproduction checks |

These are independent jobs, not a requirement for distributed training of one
model. Build common features once, save them with hashes, and have all workers
use identical split definitions and row IDs. Match model thread counts to each
VM's CPU allocation. Store artifacts on the provided VM disks or Azure storage.
Do not use the inherited AWS launch scripts or AWS artifact buckets.

The pilot recommendation is **8–16 vCPUs and 32–64 GiB RAM**, subject to the
actual VMs available. See [AZURE_COMPUTE.md](AZURE_COMPUTE.md) for sizing details.

## Experiment and submission discipline

Each run should record input and feature hashes, code version, split definition,
parameters, random seed, runtime, memory use, and complete validation predictions.
Use those predictions to compare errors and fit blends without repeatedly
retraining every member. Ensemble weights are selected on development predictions,
then frozen for the reserved audit.

Use only submissions supported by local evidence, typically **one to three on
days when candidates are ready**, within the official maximum of five per day.
There is no need to fill the daily allowance. Resolve reset/counting details before
operating near the limit. Retain every submission's version, file hash, recipe,
official result, and `usedPairs` count.

The validator must enforce the exact 344,841 movement IDs, template alignment,
two expected columns, valid prediction values, and the chosen compatible dtypes.
Keep submission creation separate from upload and verify successful scoring.

Write reproduction notes, source attribution, and external-data provenance as
the models are developed. Finalize the public GPLv3 code release and method report
by the completion target. JOAS publication can follow as encouraged by the
organizers unless a later official announcement adds a specific requirement.

## Competitive benchmark and decision points

The current [post-campaign snapshot](../ladder/leaderboard_after_campaign.json)
at **16:38:49 UTC on 14 September** places v6 **20th of 134**, with **277.4508**
seconds RMSE. Third place scores **252.9192**, requiring a further **8.84%**
reduction. The three completed ladder batches produced no qualifying replacement.
The next development direction should test new taxi-interval activity and
stand/runway information under a separate feature-availability review and
declaration. December's audit and the GPLv3 release remain reserved Phase-6/7 work.

The 13 September snapshot placed third at **261.9189 seconds RMSE** and first
at **245.0207 seconds**. These are reference points, not predicted final qualifying
scores. Refresh the team standings at milestones and measure progress against
the current third-place result.

The [weather snapshot](../weather/leaderboard_snapshot.json) at **03:13 UTC on
14 September** places the team **68th of 133**, with a best RMSE of **322.9266**.
Third place scores **256.3864**, requiring a further **20.61% reduction** from
v5. Weather improves seasonal/forward RMSE by 0.52%/1.60%; both paired-day
intervals favor the new model, but these are reused development sets. Preserve
December for an independent final audit. Seasonal extreme-label errors remain
a priority for further substantial gains. The public GPLv3 release and final
reproduction remain Phase-6/7 work.

- **By 18 September:** working pipeline, reliable baseline, first official score.
- **By 23 September:** evidence about which feature families deserve more compute.
- **By 28 September:** main error regimes understood and promising model branches identified.
- **By 4 October:** model search largely finished; final recipe shortlist frozen.
- **By 8 October:** reserved-period audit and reproduction complete.
- **By 10 October:** final scoring confirmed and reproducible source release complete.

If a phase stalls, cut low-value experiments and retain the best validated
candidate. Keep the final audit, reproducibility, and submission buffer intact.

Supporting material: [competition brief](BRIEF.md), [data audit](DATA_AUDIT.md),
[public-methods review](RESEARCH.md), and [unresolved rules](OPEN_QUESTIONS.md).
