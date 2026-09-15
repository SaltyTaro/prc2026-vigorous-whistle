# Phase 5R: new operational signals and residual targets

The initial campaign completed on **14 September 2026 UTC with no promoted
candidate**. None of the 20 declared recipes improved both complete development
folds. The current official best remains **v4: 326.6895 seconds RMSE on all
344,841 rows**. No new refit or submission was produced. December remains
reserved, and all previous phases and submitted versions are preserved.

The work follows the historical-methods review and the user's instruction to
continue. All full-data feature preparation and ten new model fits used only
the authorized Azure **arcw0**. Windows checks use the mirrored artifacts.

The initial declaration is [phase5r_v1.json](../../configs/phase5r_v1.json).
It specifies five new models per development fold and a common four-weight
grid against v4, for **20 candidate recipes**. Promotion requires at least
**0.5% less RMSE on each complete fold** than v4, then minimizes the mean of
the two RMSE ratios. This is a new development campaign on reused validation
data, not an independent audit. No submission is selected from leaderboard
probes.

## New comparisons

| Recipe | New information or learning change |
| --- | --- |
| `demand_service` | 59 planned-demand and past-service predictors added to the existing 112 |
| `baseline_residual` | Six flight-group cross-fitted stand/runway baseline features; learn a residual around the pooled 20th percentile |
| `baseline_scaled` | Same baseline plus airport IQR scaling, with squared-scale weights preserving squared-seconds loss |
| `combined` | Demand/service features and the unscaled baseline residual together |
| `missing_record` | A regularized model for all airports' missing-NM records, using the new context features and the existing clock offset |

Global comparisons retain the fixed 50% new global / 50% original-seed airport
model structure used in Phase 5. The missing-record comparison patches v4 only
on its declared route. Candidate weights of 25%, 50%, 75%, and 100% are common
across folds and airports. October alone chooses iteration counts. A selected
new model refits January-November at the rounded median of its two selected
counts; old components reuse their verified native refits.

## Availability and leakage controls

[features_phase5r_v1.json](../../configs/features_phase5r_v1.json) declares the
raw projection. Shared features read no departure labels or airport block
times. `AOBT_3_flt`, `LOBT_flt`, and `ARVT_3_flt` remain outside this campaign's
working contract; that exclusion is not a verified organizer prohibition.

Observed service uses movements strictly before a query, with equal-time
events excluded. Fixed 10/20-minute earlier queries describe plausible taxi
interval conditions. Headway statistics use recent prior runway/airport
movements, plus runway usage, last movement phase/wake, and stand departures.

Planned demand uses supplied scheduled and first-filed EOBT clocks. It includes
plans for movements that take off later, without using their eventual takeoff
time as the planned event clock. The focal departure's own plan is excluded
from these counts. Plans' publication times are absent: this is a retrospective
competition feature, not a claim that the same inputs were available in a live
pre-departure forecast. This distinction is explicit in code and tests.

Supervised baselines are separate model artifacts, never shared bundle columns.
Five inner folds group by flight ID. Every fitting row gets a baseline built
without its entire flight group. October, scoring, and ranking rows use only
the relevant outer fitting map. Quantiles, means, and IQRs use all fitting
labels, including unusual and negative values, with partial pooling from the
global residual through airport, reporting quality, runway, and stand groups.
No label or prediction clipping is applied. Saved inner assignments, maps,
feature values, model encoders, predictions, and hashes make the transforms
reviewable.

For scaled residual `z = (y - offset - baseline) / scale`, sample weight
`(scale / 600)^2` makes weighted squared residual error equal squared-seconds
error divided by the common constant `600^2`. Early stopping explicitly scores
unweighted RMSE in seconds. Scaling can still change the finite regularized
tree fit; it is an experiment, not an assumed improvement.

## Reproduction

The new package is `src/prc2026_phase5r/`. Prior model and feature packages are
unchanged. `requirements-phase5r.txt` inherits the tested Phase-5 pins.

```bash
.venv/bin/python -m pytest
.venv/bin/python tools/prepare_phase5r.py
.venv/bin/python tools/run_phase5r.py suite --fold seasonal
.venv/bin/python tools/run_phase5r.py suite --fold forward
```

All full-data feature preparation and training runs on arcw0. Local verification
uses mirrored artifacts. The scores below are from the completed comparisons.

## Results and promotion decision

All scores in this table are **development RMSE in seconds**, using 344,419
seasonal rows and 162,332 November rows, including unusual labels. They are
distinct from official leaderboard RMSE.

| Recipe | Seasonal | November |
| --- | ---: | ---: |
| **Frozen v4 reference** | **367.084321** | **244.554901** |
| Matched Phase-5 `global_capacity` baseline | 372.349229 | 245.201786 |
| `missing_record` | 369.820774 | 247.608002 |
| `demand_service` | 371.801922 | 244.820652 |
| `baseline_residual` | 371.100684 | 244.509704 |
| `baseline_scaled` | 370.511573 | 244.782326 |
| `combined` | 370.485572 | 243.557187 |
| 50% `combined` candidate + 50% v4 | 367.850800 | 243.212566 |

The new information has a modest benefit in the matched LightGBM comparison:
`combined` improves `global_capacity` by **0.5005% seasonal / 0.6707% November**.
It does not replace the stronger v4 ensemble. All twenty common-weight blends
were evaluated; **zero improve both folds, even without the 0.5% margin**.
The promotion rule was retained, and `selected_candidate` is `null`.

The lowest mean relative RMSE among the tested blends is `combined_v4_50`.
Its actual composition is 50% v4, 25% new global model, and 25% frozen
original-seed airport models. November improves **1.3423 seconds (0.5489%)**,
but seasonal RMSE worsens **0.7665 seconds (0.2088%)**. It improves all ten
airports in November and eight seasonally. Seasonal LIRF worsens by **6.7018
seconds**, outweighing improvements elsewhere; EGLL worsens by 0.6144 seconds.

The paired-day 95% interval for candidate minus v4 is **[-0.4724, 1.9829]
seconds seasonally** and **[-2.0655, -0.6598] in November**, with 38/62 and
25/30 days improving respectively. These are descriptive intervals on reused
development data. They do not replace the reserved independent audit.

The best November-only recipe, with 75% weight on `combined`, reaches
243.173633 seconds but worsens seasonal RMSE to 368.936486. Choosing that result
alone would contradict the declared two-fold protocol. No additional
airport-specific weights were fitted after inspecting these results.

Complete evidence: [development_results.json](development_results.json) has
all ten runs, all twenty recipes, full airport/date/tail slices, and the
no-promotion decision. [analysis.json](analysis.json) summarizes the matched
comparison and resource totals. Neither report consumes December labels.

## Artifacts and verification

- Feature bundle: `data/phase5r/d4decaaa6d8e7915d821`, with **171 shared
  predictors** and 2,264,211 departures. Baseline variants add six supervised
  features inside their model pipeline, making 118 or 177 predictors.
- Full bundle ID:
  `d4decaaa6d8e7915d8210a0aaa644af5caa8bb6924c646d9a56f74b8a6f9e2cc`.
- Model-code hash:
  `7023c24353cdef70c42df2bfd051be2d9bdfb4b067dec50f9e02b2ff92616118`.
- Preparation: **81.77 seconds**, **4.84 GiB** peak RSS, 397,259,619 Parquet
  bytes. Ten model runs total **956.60 seconds (15.94 minutes)**, including
  their own transforms/predictions, and peak at **6.52 GiB**. This excludes
  shared loading, verification and transfer time.
- **91 tests passed on both Windows and Azure**, including exclusion of
  simultaneous/future service events, masked-input invariance, whole-flight
  cross-fitting, squared-seconds loss equivalence, and native model reloads.
- [Azure verification](verification.json) and [Windows verification](local_verification.json) check every shared row against
  the frozen prior features, inner fold membership, independently regenerated
  baseline values, native prediction samples including the LIRF gate, all ten
  complete development scores, and all forty fold/blend scores. Both reports
  reproduce native prediction and baseline samples with **zero difference**;
  their scientific fields and protected population hashes agree exactly.
- Local model mirror: `runs/azure_arcw0_phase5r/`. The two immutable transfer
  archives contain features and model evidence separately. Their hashes and
  file receipts are in [feature_archive.json](feature_archive.json) and
  [development_archive.json](development_archive.json). The initial worker
  source archive is recorded in [code_package_initial.json](code_package_initial.json).
  [code_package_final.json](code_package_final.json) adds the completed selection,
  verification and reporting tools; the fitted feature/model code is unchanged.

Local verification command:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_phase5r.py --runs runs/azure_arcw0_phase5r --prior-runs runs/azure_arcw0_phase5 --followup-runs runs/azure_arcw0_phase5_followup --output docs/phase5r/local_verification.json
```

## Inputs prepared for the next experiment

The independent [surface-weather audit](../competition/weather_20260914/README.md)
downloaded **190,466 openly licensed IEM observations** across all ten airports
and all development/ranking periods. A report at least 15 minutes before
takeoff, no more than 90 minutes old, is available for **99.9903% of training
departures and 99.9945% of ranking departures**. Per-variable gaps still need
explicit handling. Only 19 ranking rows lack a matched observation under this
rule, and they must still be predicted.

Weather was not used in these twenty recipes. A useful next batch can test
wind/runway direction, visibility, low cloud, freezing conditions, and recent
weather changes, using a fresh declaration and feature bundle. The present
results support small benefits from the new baseline/context features; they
do not demonstrate the substantial gain needed for the top-three goal. The
working exclusion of late NM off-block fields remains unchanged, as documented
in the field-use review. No administrator message was sent.
