# Next lever after v6

The qualifying lever is **surface-traffic and flown-timing context in the
airport models as well as the globals**. The selected recipe,
`full_timeline_v6_100`, improves complete development RMSE by **0.6456%
seasonally and 0.9802% in November**, with improvement at all ten airports
on both folds. The original 0.5% promotion margin and complete scoring
populations were retained.

| Complete development evaluation | v6 | Selected surface recipe | Improvement |
| --- | ---: | ---: | ---: |
| Seasonal, 344,419 departures | 324.9171 s | **322.8193 s** | **0.6456%** |
| Forward, 162,332 departures | 223.3271 s | **221.1382 s** | **0.9802%** |

These are reused development evaluations. The selected refit was subsequently
submitted as **v7**, which officially scores **275.0570 seconds on all 344,841
ranking rows**, improving v6 by **2.3938 seconds / 0.8628%**. It was uploaded
once at 20:01:56 UTC on 14 September and processed at 20:02:30 UTC. The dated
[public snapshot](../surface_refit/leaderboard_snapshot.json) at 20:04:02 UTC
places the team **19th of 135**, up from the preceding v6 snapshot's 21st.
Third is 252.9192 seconds: a remaining **8.05%** reduction from v7 to that score.
The [official receipt](../surface_refit/submission_v7_official.json) binds the
complete result, uploaded bytes and submission checks.

## What supplied the gain

The new context measures other flights' NM pushbacks, active ground movements,
recent completed NM taxi proxies, and movements at the same stand or runway.
It also adds the supplied flown-arrival timestamp and flight-duration
consistency features. The combined model contract contains 276 predictors,
including the unchanged original NM and weather predictors.

The controlled global comparisons showed that flown timing alone did not
improve the v6 recipe. Surface activity and the combined contract were more
useful, but the best global-only recipe missed the seasonal promotion margin.
The separate airport extension used the same frozen features and original
airport learner settings. That extension made the full recipe qualify.
Feature-importance summaries support inspection but do not establish causal
effects or replace the complete model comparisons.

Where NM clocks are present, the selected prediction is **25% new CatBoost
global + 25% new LightGBM global + 50% new airport LightGBM**. Every missing-clock
row retains the original v6 prediction. The native target is the original
signed NM residual; all signed and extreme labels remain in fitting and metrics.

Most improvement comes from ordinary taxi-outs. The selected recipe lowers
daily squared error on 59 of 62 seasonal days and all 30 November days.
Descriptive paired-day RMSE intervals against v6 are [-2.6632, -1.6668] and
[-2.6944, -1.7655] seconds. These use the same reused development data. An
inherited display label in the saved interval JSON still says Phase 3; the
[annotation](../surface_airports/paired_day_label_note.json) records the actual
v6 reference and preserves the original calculations.

The clock-missing slice keeps its original error and contributes 50.25% of
seasonal and 22.31% of forward squared error after this change. Those slice
fractions describe the saved complete metrics and overlap other diagnostic
groups. They are not added to airport or duration-band fractions.

## Completed search

| Declaration | New development fits | Fixed recipes | Result |
| --- | ---: | ---: | --- |
| Timeline globals | 8 | 10 | Leading recipe 323.4328 / 221.8891; seasonal margin missed |
| Arrival taxi-in context | 4 | 6 | Every seasonal result worsens; rejected |
| Complementary route combinations | 0 | 2 | Both seasonal results worsen; rejected |
| Timeline airport extension | 20 | 4 | Two qualify; full-weight recipe selected |

The search totals **32 completed declared native fits and 22 recipes**. One
additional initial timeline fit attempt is retained: a missing import prevented
writing its provenance card before a complete development score. The corrected
repeat produced a byte-identical native model, encoder and importance file.
No prior declaration, prediction, source package or submission was overwritten.

The arrival-context fitting-only correlations did not translate into complete
model gains. Conservative callsign/route relinking also remained an audit:
positive controls still found incorrect clock associations after the proposed
gates. No automatic clock fill was used and no hidden labels were accessed.

All four complete development verifiers pass on Azure and Windows, with zero
native prediction differences. Their scientific fields agree; only report
timestamps, hashes of independently produced feature reports, and numerical
score rounding below 1e-9 differ. All **196 current test identities** have
passing records on both platforms. The [development summary](development_summary.json)
binds every selection, verification report and hash-checked archive.

## Selected refit and export

The [separate refit](../surface_refit/README.md) was declared only after both
development verifiers passed. It fits twelve native components at the frozen
October median counts, totaling 17,187 new trees. The fitting population must
match v6's **1,919,370 January-November rows**, including **1,898,549** rows with
NM clocks. Actual December movements and every December flight group remain
protected; no reserved-period score or stopping set is used.

The refit completed on the existing arcw0 in **1,800.21 seconds**, with
**15.15 GiB** sampled peak RSS. Both final platform verifiers reconstructed
**all 344,841 ranking predictions with zero difference**, including a separate
native reconstruction of the fixed weights. They also checked every fitting-only
encoder and the byte-identical imported v6 artifacts. The final export retains
exactly the template's two columns, movement order and types, with the original
nearest-integer rounding and all **5,290** missing-clock values unchanged.

The output is `runs/submissions/vigorous-whistle_v7.parquet`, SHA-256
`534e1031a068a743515eeb44b4b8f13676c5626d6c18c9416da092d58b0b924e`.
The later authorized upload repeated those live checks, and remote readback
matched the reviewed hash exactly. The readiness receipt remains unchanged;
the separate official receipt records upload and scoring.
The refit archive contains 156 hash-verified files and is mirrored locally;
the Azure originals are preserved. No campaign training or evaluation jobs remain.

Only arcw0 was used. No other Azure worker was started, and no AWS compute or
artifact storage was used. The new features reuse supplied competition data
and the already documented weather/runway sources. Their declared availability
is retrospective, including supplied flown timestamps and active-ground
endpoints; the method makes no live prediction availability claim.

The [final preservation check](preservation_final.json) verifies the original
v1-v6 submissions, all fourteen raw download hashes, frozen source packages
and passing test records. The [machine-readable campaign summary](campaign_summary.json)
binds the completed development search, final refit, both complete ranking
verifiers and v7 export. The [final source archive receipt](final_source_package.json)
records the reproduction source and supporting evidence separately from the
original data and model archives. These receipts retain their historical
pre-upload state. The subsequent [submission integrity check](../surface_refit/submission_integrity_20260914.json)
confirms the unchanged exports, source archive and original receipts; earlier
narratives are archived before adding the official result.

December's final audit remains available for an explicit final-audit workflow.
