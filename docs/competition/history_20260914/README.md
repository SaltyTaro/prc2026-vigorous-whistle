# Where to seek substantial gains after Phase 5

This extends the initial public-methods review with full-text
reports and selected source files from earlier competitions. It is research,
not a new model result. No participant implementation was executed or added to
our training packages. December remains untouched.

Our strongest next directions are **record-source and clock modeling**, followed
by **a model of airport ground congestion and unimpeded taxi time**. Surface
weather and actual stand/runway geometry are further bounded experiments. The
historical evidence favors representing the operational process explicitly;
neither another algorithm nor external data is sufficient by itself.

## The size and location of the remaining error

The saved official snapshot at **2026-09-13 20:40:07 UTC** has v4 at **326.6895
seconds**, ranked **68/131**, with third place at **261.9189 seconds**. Reaching
that dated score requires **19.83% less RMSE, or 35.72% less mean squared error**.
It is not a final cutoff. Our Phase-4 structural changes improved the official
score by 21.21%; Phase-5 tuning and blending improved it by 1.58%.

The following shares use the selected Phase-5 predictions on the **complete,
unchanged development sets**. They are local diagnostics, not a breakdown of
hidden competition labels.

| Slice | Seasonal rows | Seasonal squared-error share | Forward rows | Forward squared-error share |
| --- | ---: | ---: | ---: | ---: |
| LIRF + LFPG | 66,140 / 344,419 (19.20%) | **66.23%** | 29,974 / 162,332 (18.46%) | **43.04%** |
| Missing Network Manager origin | 5,373 (1.56%) | **39.21%** | 1,356 (0.84%) | **19.71%** |
| Target above three hours | 60 (0.0174%) | **41.26%** | 6 (0.0037%) | **13.99%** |

These slices overlap; their shares must not be added. The largest error at each
of LIRF and LFPG together accounts for **30.11% of seasonal squared error**.
Error concentration does not establish that those cases are predictable.

Even an oracle that predicted every above-three-hour case perfectly, while
leaving every other prediction unchanged and retaining every scoring row, would
reduce forward RMSE by only **7.26%**, from 244.55 to 226.81 seconds. This is a
diagnostic upper bound, not an achievable forecast. It supports working on
ordinary taxi/congestion errors as well as exceptional records.

The previous fitting-only study is also important: large schedule intervals on
NM-present LIRF records did **not** identify large taxi labels. For example, the
52 fitting rows with intervals above six hours had no targets above three hours.
Broadly extending our current schedule-offset rule is unsupported. Any new
correction needs observable evidence and fitting-period support, not a rule
chosen around an individual held-out error.

Calculations and input hashes: [error_headroom.json](error_headroom.json).
Existing fit-only evidence: [Phase-5 diagnostics](../../phase5/fitting_diagnostics.json).

## What the earlier editions actually predicted

The annual series has different targets. The 2024 and 2025 scores below are in
**kilograms**, and cannot be compared with the 2026 scores in seconds.

| Edition | Target | Verified leading solutions |
| --- | --- | --- |
| 2024 | Actual takeoff weight, using 2022 flights and ADS-B trajectories | `team_likable_jelly`: **1,561.63 kg**, first; `team_tiny_rainbow`: **2,217.75 kg**, second |
| 2026 | Departure taxi time, from airport/NM movement records | Our current v4: **326.6895 seconds** |

Primary results: [2024 final leaderboard](https://ansperformance.eu/study/data-challenge/dc2024/rankings.html),
[2024 prize announcement](https://ansperformance.eu/study/data-challenge/dc2024/),
[2025 final outcome and organizer solution reviews](https://ansperformance.eu/study/data-challenge/dc2025/outcome.html).

The 2024 leaderboard's third-scoring team was `team_gentle_elephant` at 2,252.89
kg. The third **prize** went to `team_brave_pillow`, fourth on score at 2,270.06
kg. The published pages establish this distinction, but do not establish the
reason; we do not infer it. Score ranking and award eligibility are distinct.

## Methods worth transferring

### 1. A useful intermediate target can matter more than a new learner

The [2024 winner's implementation](https://github.com/PRC-Data-Challenge-2024/team_likable_jelly/tree/fd8f0c45e737848b9da53cb4739fd463c9775504)
uses LightGBM. Its features align trajectories into climb-altitude slices and
relative-time slices; derive energy rate and an equivalent aircraft mass; and
reconcile reported times using independent trajectory observations.

It trains on `(TOW - empty_weight) / (maximum_weight - empty_weight)`, with
**squared scale weights** to retain the absolute-mass squared-error objective.
We checked the target transformer and training code as well as the README.
The transferable idea is to make a model learn the uncertain part of an
operational quantity while keeping the actual competition loss.

For taxi-out, an independently built version could learn the difference from
a supported clock proxy or from a stand/runway taxi baseline. A possible
formulation is:

```text
z = (taxi_seconds - baseline(x)) / airport_scale
training_weight = airport_scale ** 2
prediction_seconds = baseline(x) + airport_scale * predicted_z
```

Then `weight * (z - predicted_z)^2` equals squared error in seconds. Any
baseline or scale learned from labels must be fitted within the fitting fold;
supervised baseline predictions used for fitting another model need inner
cross-fitting. The baseline must retain our existing supported clock offset,
with sparse stand/runway groups falling back to broader airport groups.

This is a proposed experiment. A simple airport normalization may add little
because we already have airport specialists. A better operational baseline is
the more substantial opportunity. Neither takeoff-mass predictions nor the
2024 timestamp-correction code can be transplanted directly: 2026 does not
supply those trajectories, and the old code's fallback uses taxi time, which
is a known input in 2024 but our target in 2026.

The [2025 winning report](https://github.com/PRC-Data-Challenge-2025/resourceful-quiver/blob/2484125852ebb6cfb1fe55c253fb3f7539c924ff/PRC_2025_report.pdf),
sections 5-6 and Table 4, provides a second example. It predicts **fuel flow**
and multiplies by interval duration to produce fuel consumed. With its full
recursive-mass feature set, the authors report ranking RMSE **203.75 kg** for
flow training versus **220.56 kg** for direct amount training. The inspected
notebook converts predictions back to kilograms before computing RMSE.
The report's ablations reused the ranking set; they are author-reported
development evidence, not independent tests or estimates of our possible gain.

### 2. Model the underlying state and preserve measurement provenance

The 2025 winner derived aircraft-mass and fuel-flow features, using a mass
estimator trained with the previous challenge's data. Its best reported
feature configuration used a stable mass interpolation; a more elaborate
recursive calculation could accumulate errors or produce negative mass when
trajectory observations were poor. The simpler version performed better.

The [2025 runner-up]
likewise models mass, handles empty trajectory segments explicitly, includes
neighboring-segment context, and combines CatBoost, LightGBM, and XGBoost.
Its README's older approximately 224 kg / top-eight claim is superseded by the
organizer's final **211.99 kg / second-place** result. Source files corroborate
the architecture, but this review did not reproduce the scored version.

For us, the analogous latent state is **ground demand relative to available
runway service**, together with the distinction between actual movement delay
and a source's reporting convention. Candidate signals include:

- Departure banks from scheduled/initial off-block clocks, paired with observed
  takeoff throughput and headways.
- Runway arrival/departure mix, active-runway changes, wake-category mix, and
  periods of reduced service rate.
- Congestion measured near an estimated pushback time as well as takeoff;
  present traffic features are centered on the takeoff observation.
- Stand/runway baseline duration, followed by an additional-delay model.
- A partially pooled expert for missing or conflicting records, with a
  probability of using each supported clock interpretation and a global fallback.

We already have strict-past event counts and airport models. The new work would
model demand, service, and the taxi interval jointly. It should first use the
supplied fields and existing past-event policy. Any extension to retrospective
future-movement context requires its own availability review and matching
training/ranking construction. No departure target or airport block time may
enter shared features; any predicted block time must come from fold-safe
predictions.

### 3. Operational features have stronger evidence than indiscriminate model growth

The [2024 runner-up paper](https://journals.open.tudelft.nl/joas/article/view/7963),
Table 8, reports **24.83% higher RMSE after removing operational features**.
Removing meteorological features increased RMSE by 1.76%. Removing its neural
network or SAINT component had no measured effect on the final ensemble.
These are that team's task-specific ablations, not forecasts for taxi-out.

The 2024 winner's README reports a single-model score of **1,612.41 kg**,
ten-seed average **1,564.09**, and twenty-seed average **1,561.63**. The last
ten seeds bought approximately **0.16%**. It also documents changes in retrieved
weather data and nondeterministic training that affected reproduction.

For this project, weather observations and operational features should be
snapshotted with hashes; another seed campaign should follow new predictive
information. We have already tried larger LightGBM models, a longer CatBoost
run, seeds, and 28 blend recipes in Phase 5.

### 4. Treat weather, data cleaning, and augmentation as hypotheses

The 2025 winning report found limited value from additional gridded weather;
in its best mass configuration, removing wind improved the reported ranking
RMSE from **201.04 to 199.91 kg**. Attempts to reconstruct missing trajectory
ends worsened results. This supports keeping missingness visible and testing
each enrichment separately.

For 2026, surface weather has a different mechanism: visibility, precipitation,
freezing conditions, and wind relative to runway heading affect ground
capacity and deicing. Start with hourly/subhourly METAR joins and explicit
report age/missingness. Test January and July separately as well as the complete
folds. Add stand/taxiway geometry only after checking actual stand-code coverage.
Our Phase-3 layout features were code prefixes and stand-number groups; their
failure does not test actual taxi-path distance. Completed-arrival summary
features were also already tried and should not simply be rerun unchanged.

Two newer papers provide useful ideas but need careful interpretation:

- [Grigoriou et al., article 8764](https://journals.open.tudelft.nl/joas/article/view/8764)
  combines physics, METARs, and targeted widebody augmentation. The July 2026
  paper reports **214.92 kg**, while the organizer's final outcome lists its
  competition team `bright-lobster` at **303.02 kg, rank 11**. A later research
  result and the awarded competition version are not interchangeable. Its
  augmentation/validation evidence does not justify making synthetic copies of
  our rare clock anomalies. The archived competition README also identifies
  LLM-generated aircraft fuel constants; we would require independently sourced
  values for any analogous physical constants.
- [Kačar and Ćulibrk, article 8766](https://journals.open.tudelft.nl/joas/article/view/8766),
  published 26 August 2026, was missing from our initial review. It separates
  ADS-B and ACARS aggregation, derives latent physical quantities, and reports
  a 227.32 kg blind-test result. This is an additional research solution, not
  one of the verified three prize winners. Its source-aware representation is
  relevant to our airport-versus-NM records; its airborne physics is not a taxi
  model.

The 2025 [organizer data paper](https://journals.open.tudelft.nl/joas/article/view/8750)
also documents ambiguous fuel units, variable source timing, and residual label
noise. The general lesson is to understand how observations became labels. It
does not establish that any particular 2026 extreme is wrong, or authorize
dropping it from evaluation.

## Proposed next campaign

This is a new development campaign before the final audit, preserving the
completed Phase-5 recipe. No experiment in this table has been run here.

| Order | Experiment | Difference from existing work | Evidence required |
| --- | --- | --- | --- |
| 1 | **Clock/source experts and residual target** | Learn when each allowed clock is informative; share information across airports and missingness regimes; test an operational baseline with optional scale-corrected loss | Fitting-only evidence for routing; complete-fold improvement; false-positive cost on ordinary rows; stability across dates |
| 2 | **Demand, runway service, and taxi-interval context** | Go beyond takeoff-centered past counts to departure-bank/throughput imbalance and context near estimated pushback | Fresh feature bundle; fold-safe intermediate predictions; masked departure labels/block times; feature-family ablation |
| 3 | **Unimpeded taxi baseline and real geometry** | Smoothed stand/runway baselines and actual distance/geometry, rather than stand-code prefixes | Cross-fitted supervised baselines; stand mapping coverage; explicit fallbacks; gain beyond airport models |
| 4 | **Surface weather and operating-regime interactions** | Add METAR visibility, precipitation/freezing, report age, and runway-relative wind to the operational model | Licensed source snapshot; ten-airport/month coverage; separate weather ablation; January/July/November results |
| 5 | **Combine complementary new models** | Blend independently useful feature mechanisms with frozen v4 | Full-set saved predictions, day-level uncertainty, and complementary errors; original December audit still reserved |

First screen each hypothesis using the same fixed LightGBM recipe on both
development folds, preserving October stopping and all existing scoring IDs.
Only promising feature families justify another full CatBoost/LightGBM ensemble
refit. Compare any proposed submission with frozen v4 on both complete periods,
all airports, missingness slices, and date-level errors. Declare recipes before
scoring; disclose adaptive follow-ups and do not optimize around individual IDs.

The first diagnostic and screening jobs fit the existing **arcw0 CPU worker**.
No additional VM or GPU is needed to begin. Before any large enrichment download
on Azure, account for arcw0's approximately 6.4 GiB of free root-disk space at
the end of Phase 5. The other seven workers remain unauthorized to start under
the current allocation.

The dated top-three gap justifies continuing development before spending the
December audit. The audit should assess a final frozen candidate after this
work, while retaining the planned October release/submission buffer.

### One unresolved, potentially consequential field-use question

`AOBT_3_flt` and `LOBT_flt` are supplied and defined on the official 2026 data
page, but our models exclude them pending original organizer guidance. A
confirmed eligible off-block clock would support a direct clock-proxy residual
model and could materially change the priority order. Its value is **unmeasured
in our validation**. Public competitors using it do not establish eligibility.

This review freshly retrieved the official data/ranking pages and the public
website repository's six issue/PR entries. We found no field-specific ruling
there. The user answered
that they do not know of an announcement; the clarification request is answered,
and no administrator message was sent. A further
[public-page review](../field_use_review_20260914.json) confirms the documented
fields and post-operations rationale without finding a field-specific ban.
Other experiments above can proceed with the current feature contract. The
exclusion is a temporary project decision, not a verified organizer restriction.

## External data and implementation rights

- [IEM's own usage statement](https://mesonet.agron.iastate.edu/disclaimer.php)
  says its website materials are **public domain** and may be used freely;
  attribution is appreciated. This is stronger evidence than a participant's
  license claim. Actual airport/date coverage and weather quality remain to be
  measured before modeling.
- [OurAirports](https://ourairports.com/data/) releases its data to the public
  domain. It provides runway/airport coordinates, not a complete stand/taxiway
  graph. [OpenStreetMap](https://www.openstreetmap.org/copyright) provides ODbL
  data with attribution/share-alike conditions and variable airport coverage.
- The [2024 organizer paper](https://journals.open.tudelft.nl/joas/article/view/8252)
  describes a CC BY 4.0 archive of 527,162 flights. Its flight table includes
  **taxiout_time in minutes**, so a small-table study of historical taxi-time
  priors is possible. The flights are from 2022, airlines are obfuscated, and
  the listed table lacks stand/runway detail. The equivalence of that NM-derived
  quantity to the 2026 airport-reported target needs checking. Treat it as a
  lower-priority transfer experiment, not an extra set of interchangeable 2025
  labels. Downloading the approximately 286 GiB trajectory archive is unnecessary
  for that assessment and has not been done.
- Retrieved LICENSE files identify GPLv3 for the 2024 winner/runner-up and
  the 2025 winner/third-prize team. The 2024 `brave_pillow` license is AGPLv3. Our plan is an original implementation of the useful concepts, with
  attribution and the competition's GPLv3/public-data requirements.

## Evidence and reproduction

The archive contains **65 public source objects**, including **five journal
PDFs, the 2025 winner's full report, seven repository commit/tree snapshots**, and
selected READMEs/source/license files. This was a methods/results review, not a
reproduction or exhaustive audit of each repository. Repository source URLs
use exact commit IDs. Raw downloads and full-text extracts remain in the
Git-ignored `docs/competition/sources/history_20260914/` directory.

- [requests.json](requests.json): exact public GET request register.
- [source_manifest.json](source_manifest.json): response URLs, UTC retrieval
  times, SHA-256 hashes, content sizes, PDF page counts, extraction versions.
- [retrieval_notes.json](retrieval_notes.json): the initial 2024 singular
  `ranking.html` URL returned 404; the linked `rankings.html` URL succeeded.
- [error_headroom.json](error_headroom.json): complete-set calculations and
  hashes of the frozen Phase-5 selection and standings inputs.
- [last_pull.json](last_pull.json): most recent archive verification outcome.
- [verification.json](verification.json): all 65 response hashes, 32 downloaded
  files matched to their Git blob IDs, seven commit/tree relationships, Python
  syntax, document links, and unchanged Phase-5 input hashes checked successfully.

Install the research dependencies plus the PDF parser when needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-history.txt
.\.venv\Scripts\python.exe -X utf8 -B tools/pull_prc_history.py
.\.venv\Scripts\python.exe -X utf8 -B tools/pull_prc_history.py --offline
.\.venv\Scripts\python.exe -X utf8 -B tools/research_prc_gain_headroom.py
```

The pull tool only performs public reads and caches text/PDF evidence; it does
not load `.env`, execute downloaded code, access competition storage, or submit
predictions. The headroom tool reads saved aggregate metrics only, checks
partition counts and squared-error totals, and does not read new labels or
fit/score a model. No training packages, feature bundles, original raw competition
files, model predictions, or submissions were changed by this research.
