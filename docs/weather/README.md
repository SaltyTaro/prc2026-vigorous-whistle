# Surface-weather experiments

This separately declared development batch adds licensed surface observations
to the existing taxi-time models. It was declared at **01:59:25 UTC on
14 September 2026**, before any weather model was fitted or scored. Training
started at **02:01:02 UTC on arcw0**. All six fits and sixteen declared recipes
have completed. **CatBoost with weather qualifies for promotion**, with 0.52%
lower seasonal RMSE and 1.60% lower forward RMSE than v4. Its refit, both-platform verification and official scoring are complete.
**V5 scores 322.9266 seconds RMSE on all 344,841 departures**, improving v4
by **3.7629 seconds / 1.1518%**. It is the new best official submission. December remains reserved.

## Inputs and availability

The [feature declaration](../../configs/features_weather_v1.json) binds the
30 verified IEM source CSVs: **190,466 observations**, covering January-November
2025 and January/July 2026 at the ten competition airports. Source URLs, hashes,
retrieval dates and public-domain usage evidence are in the
[input inventory](../competition/weather_20260914/inventory.json).

The latest eligible observation is at or before movement time minus 15 minutes,
and no more than 90 minutes old at movement time. A second lookup applies the
same rule at movement time minus one hour. Means and maxima use every report in
the closed interval `[movement - 195 minutes, movement - 15 minutes]`; these
longer historical summaries can remain available when the latest report is
stale. No current or future report is used. Provider publication times are
absent: the 15-minute delay is a conservative convention, not evidence of exact
live availability. All movements remain in the prediction population.

The 61 new predictors include temperature, dew point, humidity, pressure,
visibility, wind direction/speed/gusts, cloud cover and ceiling, reported weather
conditions, recent changes and three-hour summaries. Units are Celsius, hPa,
kilometres of visibility, metres for cloud height and metres/second for wind.
Unreported gusts and unknown sky coverage remain missing. An empty weather-code
field means no condition was reported; it is not independent proof of clear sky.

`p01i` is omitted because all 190,466 downloaded values are zero. Rain/drizzle,
snow, freezing weather and storms instead come from reported weather codes.
The wholly missing `mslp` column is omitted. Raw inputs are preserved.

A later label-free [covariate audit](covariate_audit.json) found four finite
but implausible source pressure reports outside 850-1,100 hPa. The already
declared first batch retains those source values: 24 training rows and 23
ranking rows inherit one as their latest pressure. Neither complete scoring
period contains an outlying latest pressure or a pressure change over 100 hPa.
The [completed pressure review](pressure_review.json) bounds the resulting
submission-score sensitivity at 0.42 seconds and retains the validated recipe.
This remains a recorded input-quality limitation. Numerical cloud fractions are representative
encodings (FEW 0.125, SCT 0.375, BKN 0.75, OVC/VV 1), not precise measured cover.

[OurAirports runway coordinates](../competition/weather_20260914/runway_sources.json)
provide geodesic runway bearings for headwind, tailwind and crosswind features.
The saved [usage page](../../data/external/ourairports/data_terms.html) releases
the data to the public domain. This is a static snapshot retrieved on
14 September 2026, not historical runway-operation information. Exact
airport/designator matches include now-closed runways, since movements are
historical. An unmatched but valid runway number supplies a flagged ten-degree
bearing approximation; there is no invented runway-name remapping. The flag
distinguishes external geodesic bearings from that approximation. The snapshot
also supplies runway length. No stand or taxiway geometry is claimed.

## Frozen comparisons

The [model declaration](../../configs/weather_v1.json) specifies **six fits**:

| Variant | New global model | Predictors |
| --- | --- | ---: |
| `lgb_weather` | Existing high-capacity LightGBM settings plus weather | 173 |
| `lgb_ground_weather` | Same LightGBM settings, ground demand/service plus weather | 232 |
| `cat_weather` | Existing 6,000-tree-cap CatBoost settings plus weather | 173 |

Each is fitted on both original development folds. October supplies early
stopping. The seasonal score covers all **344,419** January/July departures;
the forward score covers all **162,332** November departures. Negative and
extreme labels are retained. The existing conditional LIRF schedule offset is
unchanged. No new supervised baseline, target scaling, target clipping or
airport-specific blend tuning is introduced.

Each candidate pipeline uses 50% new global model and 50% frozen original-seed
airport models. Four families are evaluated: each individual pipeline and a
75% CatBoost / 25% ground-weather LightGBM combination. Each family is blended
with v4 at new-family weights 25%, 50%, 75% and 100%, giving **16 recipes**.
Weights are common to both folds and every airport. Promotion requires at least
**0.5% lower complete RMSE on both folds** than v4; eligible candidates are
ranked by the equally weighted mean of the two RMSE ratios. This is reused
development evidence, not an independent final audit.

V4's development references are **367.0843209597658 seasonal** and
**244.5549008776592 forward**, distinct from its official leaderboard score.
Only a qualifying recipe can proceed to a fixed-iteration January-November
refit, verification and a later versioned submission.

## Complete development results

| Recipe (global/airport pipeline) | Seasonal RMSE, seconds | Forward RMSE, seconds |
| --- | ---: | ---: |
| Frozen v4 | 367.084321 | 244.554901 |
| LightGBM + weather | 370.918802 | 242.898321 |
| LightGBM + ground traffic + weather | 370.012892 | 242.025602 |
| **CatBoost + weather (selected)** | **365.171140** | **240.652220** |
| 75% weather CatBoost + 25% ground/weather LightGBM | 365.484624 | 240.168675 |

Only `cat_weather_v4_100` meets the 0.5% margin on both complete folds. This
name means the complete proposed pipeline receives weight 100%; that pipeline
is **50% weather CatBoost plus 50% unchanged original-seed airport models**.
The mixed CatBoost/LightGBM family has the lowest mean RMSE ratio but misses
the seasonal margin (0.4358% improvement), so it is not selected. No weights
were adjusted by airport or fold after seeing results.

The selected seasonal gain is **1.913181 seconds / 0.52118%**, improving 40 of
62 scoring days. Its paired-day bootstrap 95% interval for candidate-minus-v4
RMSE is **[-3.4463, -0.5117] seconds**. Forward improves **3.902680 seconds /
1.59583%**, with 25 of 30 days improving and interval **[-7.5349, -1.3649]**.
These intervals describe reused development sets and do not correct for the
full history of model selection. They do not replace December's reserved audit.
The selected model improves nine of ten airports on each fold: EGLL worsens
by 0.29 seconds seasonally and LEBL by 0.68 seconds forward. Seasonal gains
come primarily from departures with labels below one hour; the rare one-to-three
hour and over-three-hour groups still worsen slightly in aggregate. Forward
gains include both ordinary departures and the six over-three-hour records.
[Selected diagnostics](selected_diagnostics.json) records every group delta.

October selected **5,991 / 5,998 CatBoost trees**. The predeclared rounded-median
rule gives **5,994 refit trees** (ties to even). The final fit must retain all
1,919,370 January-November departures while protecting December flight groups.

Every source run, complete metric, declared blend, iteration count and paired
day comparison is in [development_results.json](development_results.json),
SHA-256 `1228b7b16d3bea90229da139fb51ee714cf323472f3796604d40b1a7178351f3`.
[analysis.json](analysis.json) compares each new model with its matched earlier
model and records feature importances. Both LightGBM variants and seasonal
CatBoost improve their matched no-weather models. This is a modest gain. The remaining competition gap needs stronger signals,
particularly for source/clock regimes and long-duration records.

Six fits total **2,126.16 seconds**, including each fit's own transforms and
predictions, with **7.29 GiB** maximum sampled process RSS. Only arcw0 ran
training. The [live team check](team_status_before_decision.json) at 02:32 UTC
confirmed only v1-v4 exist and v4 is the best official result.

## Artifacts and checks

New source is isolated in `src/prc2026_weather/`. Prior feature/model packages,
configurations, predictions and submissions remain unchanged. The feature bundle
is **`data/weather/827d828796d9cb9c4581`**, with full identity
`827d828796d9cb9c45811d48b0d9647659c187dd719d6ba3caa771a16bfe1ce8`.
All 171 inherited feature values and all protected IDs are preserved exactly.

Preparation took **46.83 seconds**, peaking at **1.42 GiB** sampled process RSS.
The independent pre-model coverage audit matches the prepared features exactly:
186 of 1,919,370 training rows and 19 of 344,841 ranking rows lack a recent
report. Missing-weather flags and native missing-value handling retain them.

**100 tests pass on Windows and Azure.**
Two additional refit-selection tests pass on each platform, rejecting altered
iterations/weights or a failed two-fold margin even if an eligibility flag is
incorrectly set. The two groups total **102 passing tests per platform**.

Both the [Azure verifier](verification.json) and the [Windows verifier](local_verification.json)
pass all 2,264,211 inherited rows, 13,979 recomputed weather rows, six complete
validation scores and 32 blend-fold evaluations. Every native prediction sample
reproduces with zero difference. [Platform comparison](platform_comparison.json)
records the sub-picosecond rounding differences in recomputed RMSE.

The selected refit completed on arcw0 with the fixed 5,994-tree count. Runtime
was **1,155.32 seconds** and sampled peak RSS **8.08 GiB**. The executing pipeline
has **11 native models / 12,416 total trees**: the new CatBoost and ten unchanged
airport models. Byte-identical unused parent artifacts are also preserved for
traceability; see [refit_summary.json](refit_summary.json).

Both [Azure final verification](verification_refit.json) and
[Windows final verification](local_verification_refit.json) pass, including
1,400 ranking rows covering every conditional LIRF and missing-weather case,
with **zero native prediction difference**. The verifier separately reconstructs
the final weights from the original parent models and selected native node.

The [pressure sensitivity check](pressure_sensitivity.json) affects 49 ranking
rows. The [review with integer-rounding allowance](pressure_review.json) bounds
the possible change in full-ranking RMSE by **0.419944 seconds** for any fixed
hidden labels. The validated recipe is retained unchanged. This calculation
is label-free and does not measure a pressure-filtered retraining experiment.

The exported [v5 receipt](submission_v5.json) records **344,841 rows**, exactly
the template IDs/order and `double` ID / `int32` prediction columns. Its
1,340,031-byte Parquet has SHA-256
`25ddc5554ce2fc059b920fbd43e58e149facc7e837c49b6e292ce434f66fa3bd`.
Predictions are rounded to nearest integer, ties to even, with no clipping;
19 are negative. **V5 uploaded at 03:12:10 UTC on 14 September and scored at 03:12:45 UTC.**
The [official result](official_result.json) confirms **322.9266 seconds RMSE /
344,841 used pairs**. Do not overwrite or re-upload v1-v5. Check live versions and limits before any later submission.

At **03:13:54 UTC on 14 September**, the [standings snapshot](leaderboard_snapshot.json)
places the team **68th of 133**. Third place has improved to **256.3864 seconds**,
requiring a further **20.6054% RMSE reduction** from v5 to that dated score.
Weather provides a measured improvement; this is not a top-three result or a
final qualifying cutoff.

Azure originals are under `runs/weather/`; the local evidence mirror is
`runs/azure_arcw0_weather/`. Initial deployment is recorded in
[code_package_initial.json](code_package_initial.json); the complete support
code and input snapshot is [code_package_final.json](code_package_final.json).
The [feature archive](feature_archive.json), [development archive](development_archive.json)
and [refit archive](refit_archive.json) are hash-checked local mirrors. Verification reconstructs
weather features from source samples, checks every inherited feature row,
reproduces complete validation scores and every declared blend, and checks
native model predictions including all conditional LIRF and missing-weather
scoring rows.

On the existing Azure workspace:

```bash
.venv/bin/python -u tools/prepare_weather.py
.venv/bin/python -u tools/run_weather.py --fold seasonal
.venv/bin/python -u tools/run_weather.py --fold forward
.venv/bin/python -u tools/select_weather.py
.venv/bin/python -u tools/verify_weather.py --require-refit --output docs/weather/verification_refit.json
```

Local verification after restoring the hash-checked archives:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_weather.py --runs runs/azure_arcw0_weather --prior-runs runs/azure_arcw0_phase5 --followup-runs runs/azure_arcw0_phase5_followup --require-refit --output docs/weather/local_verification_refit.json
```

Requirements inherit the existing tested `requirements-phase5r.txt` stack.
There are no additional libraries, credentials or cloud workers. Only arcw0 is
used. All external objects are verified offline against the frozen manifests;
the feature/training tools make no weather API requests. Preserve the cached
source snapshots: public download URLs can change over time. Re-downloading or
rewriting source inventories requires a new identity rather than silently
replacing these scientific receipts.
`tools/pull_weather_geometry.py --offline` verifies the cached public runway
snapshot without changing its receipt; omitting `--offline` can restore missing
files only while the public source still matches the saved hash.

The [completion record](completion.json) binds the final evidence hashes,
official result and selected recipe. At the [03:19 UTC VM check](azure_status.json),
arcw0 has no PRC training or verification jobs running; the VM remains available
and other services are preserved. The project filesystem has 3.56 GiB free.
Plan artifact storage before another large batch. Only arcw0 was used.
