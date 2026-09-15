# v10 accumulation (15 September 2026; scored 267.9646, worse than v9; v9 retained)

Two declarations on top of the official best **v9 (267.4454)**, following the
plan in [`../v10_plan.md`](../v10_plan.md). Both are scored against the exactly
reconstructed v9 development predictions (314.2323 seasonal / 219.8750
forward, reproduced to 1e-6 from the frozen v7 runs plus a refit of the v9
classifier on each fold's fitting months).

## What the plan's items turned out to be worth

Same-day scratch prototypes on the two folds, disclosed in both declarations:

| Item | Seasonal | Forward | Verdict |
| --- | ---: | ---: | --- |
| `cat_long` at 25% on the clock route | −0.137% | −0.184% | kept |
| negative-proxy routing to the v5 no-clock recipe | −0.113% | −0.048% | kept |
| LIRF clock-present schedule-fill mixture | −0.10% | −0.02% | dropped: the main models already hedge these fills from the same clock features (classifier AUC 0.87 with or without stand turnaround); paired-day intervals include zero |
| EGLL/LTFM/LFPG clock-present fills | oracle bound 0.28% / 0.32% of SSE | | dropped: too small to survive imperfect separation |

A new item was found on the fitting months: **day-level density** for the LIRF
population classifier. The schedule-fill share of population rows is 65–69% on
days with one or two population rows and 30–37% on days with eight or more
(67% versus 32% at low versus high LIRF no-NM shares). Six label-free day
quantities (departures, no-NM departures, share delayed over an hour, population
rows, their mean schedule interval, population rows within two hours) were added
to the v9 classifier with everything else unchanged.

The plan's 0.5%-per-fold margin was set for native model replacement; it would
reject any honest accumulation of ordinary-row items here by construction, on a
seasonal fold whose SSE is 30% two unpredictable LFPG rows (labels 84240 and
58206 s with schedule intervals under 35 minutes and no NM record). The
declarations therefore fix, before scoring, a **0.25% margin on each fold plus a
paired-day 95% interval entirely below zero on each fold**.

## Declaration 1 — `configs/v10_v1.json` (05:46 UTC): nothing eligible

Single-seed classifier. [`development_results.json`](development_results.json)
(`runs/v10/selection.json`, SHA-256 `26c1d6c3…9069`):

| Candidate | Seasonal | Forward | Eligible |
| --- | ---: | ---: | --- |
| `cat_long` | 313.8031 (−0.137%) [−0.64, −0.26] | 219.4705 (−0.184%) [−0.59, −0.22] | no (margin) |
| `route` | 313.8782 (−0.113%) [−0.59, −0.14] | 219.7693 (−0.048%) [−0.63, +0.48] | no |
| `density` | 314.0717 (−0.051%) [−0.99, +0.72] | 219.0566 (−0.372%) [−1.58, −0.28] | no |
| `cat_long_route` | 313.5015 (−0.233%) [−1.02, −0.47] | 219.4397 (−0.198%) [−1.03, +0.22] | no (margin) |
| `density_cat_long` | 313.6423 (−0.188%) [−1.44, +0.29] | 218.6506 (−0.557%) [−1.97, −0.65] | no |
| `density_route` | 313.7175 (−0.164%) [−1.42, +0.40] | 218.9505 (−0.420%) [−1.81, −0.08] | no |
| `density_cat_long_route` | 313.3406 (−0.284%) [−1.79, **+0.02**] | 218.6196 (−0.571%) [−2.15, −0.37] | no (interval) |

The full stack missed the seasonal interval condition by 0.022 s. The reason is
classifier variance: with feature subsampling, the single-seed classifier's
seasonal density-only score moves between −0.04% and −0.24% against v9 when
only the seed changes (forward −0.35% to −0.41%).

## Declaration 2 — `configs/v10_v2.json` (05:53 UTC): `density_cat_long_route`

Identical items, candidates and rule; the classifier becomes an **eight-member
seed bag** (v9 seed + 0…7, each member with its own date-grouped inner rounds,
probabilities averaged). [`development_results_v2.json`](development_results_v2.json)
(`runs/v10/selection_v2.json`):

| Candidate | Seasonal | Forward | Eligible |
| --- | ---: | ---: | --- |
| `density` | 313.7757 (−0.145%) [−1.36, +0.48] | 219.0386 (−0.380%) [−1.61, −0.29] | no |
| `density_cat_long` | 313.3459 (−0.282%) [−1.82, +0.03] | 218.6326 (−0.565%) [−1.99, −0.67] | no |
| `density_route` | 313.4211 (−0.258%) [−1.81, +0.16] | 218.9325 (−0.429%) [−1.82, −0.10] | no |
| **`density_cat_long_route`** | **313.0439 (−0.378%)** [−2.22, −0.21], 49/62 days | **218.6016 (−0.579%)** [−2.18, −0.38], 24/30 days | **yes** |

(`cat_long`, `route` and `cat_long_route` do not involve the classifier and
score exactly as in declaration 1.) Member rounds: seasonal 192–281, forward
337–419; scoring fill AUC 0.896 / 0.984 against v9's 0.893 / 0.981. Population
SSE by true mode, selected candidate versus v9: seasonal fills 8.16e8 → 1.14e9,
normal/other 2.48e9 → 2.06e9, one-day 1.25e9 unchanged; forward fills
1.31e8 → 0.69e8, normal/other 0.52e8 → 0.54e8, one-day unchanged. The
seasonal gain of the density item therefore comes from disruption days on which
the old classifier over-called fills; the clock blend and routing add their
ordinary-row gains on top.

Honest expectation for the official score: about −0.4% to −0.5% (≈ 1.2 s, to
roughly 266.2), which buys one or two positions around 10th–11th. It cannot
approach top 5 (260.4) or top 3 (246.6).

## Refit, export and verification

- **cat_long refit** (`src/prc2026_v10_refit/`, `tools/refit_v10_cat_long.py`,
  launched by `tools/launch_v10_refit.py` on arcw0 at 06:01 UTC after the
  five-test suite passed there): CatBoost `cat_long` at the declared **9,929
  trees** on the 1,898,549 December-protected clock-present Jan–Nov rows
  (identity `89005020…af95`, the v6 refit's population), 1,988 s wall at
  14.9 active cores, 9.5 GB peak RSS; reload difference on 1,171 rows 0.0.
  Run `runs/azure_arcw0_v10/20260915T060119134146Z_refit_cat_long/` (run.json
  SHA-256 `453beaa4…8149`; ranking clock predictions
  `5b75c87a…f93a5`, 339,551 rows), streamed back hash-checked by
  `tools/mirror_v10_refit.py` ([`refit_mirror.json`](refit_mirror.json),
  archive `edb35882…be4b`, 13 files). arcw0 had 0.99 GB free afterwards; no
  other worker was started.
- **Export** `tools/export_v10.py --version 10` →
  `runs/submissions/vigorous-whistle_v10.parquet` (SHA-256
  **`45690fdb30297ff3ea17c96d92ca8a3db738136256226596cae07855097152f0`**,
  344,841 rows, minimum −12, maximum 120,549 — the latter inherited unchanged
  from v7/v9 on a long-schedule LIRF row). The export first refits the v9
  classifier on Jan–Nov and reproduces the **uploaded v9 integers on every
  row from scratch** (its booster is byte-identical to the v9 export's,
  SHA-256 `c8ed40b1…058e`), then fits the eight density members (rounds
  288–497, 1,399 population rows: 655 fill / 733 normal or other / 11 one-day)
  and applies the selected items: 372 mixture rows, 339,551 clock-blend rows,
  169 routed rows, 14 floored rows; 327,154 rows differ from v9 (median
  absolute change 7.3 s, 99.9th percentile 217 s, 46 rows above 1,000 s, all
  population or routed rows). Members and record under
  `runs/v10/export_v10/` (record SHA-256 `225179be…7db2`). Report:
  [`export.json`](export.json).
- **Verification** `tools/verify_v10.py --version 10`
  ([`verification.json`](verification.json)): an independent
  re-implementation (per-row filtered turnaround features, per-day pandas
  groupbys and a per-row window filter for the density features, normal
  component and floor from refit labels, the eight saved boosters averaged,
  the clock blend from the mirrored refit predictions, routing from raw NM
  clocks) reproduces every exported integer; rows outside the declared items
  equal the uploaded v9; saved probabilities and density features differ by
  0.0; the refit record and receipts are re-checked.
- Code receipts: development model code SHA-256
  `817cb72c9cd4312798a2c91f95fdf9959195d24389968934f089e442a6c54d6f`
  (`src/prc2026_v10/`, `src/prc2026_lirf_mixture/`, `tools/run_v10.py`,
  `run_lirf_mixture.py`, `run_postprocess.py`); refit code
  `4df5c820258d9f508514082d20c8e3257ddb1224a8099ff13e97d2dea3e474d6`
  (ladder code plus `src/prc2026_v10_refit/` and the refit tool). Selection
  receipt `runs/v10/selection_v2.json` SHA-256
  `ef1aed5a84b1844c10df6c2973c52aa027a10c0178e0eacded441ada3b0f6f6e`.
  Declarations are copied here as [`declaration_v1.json`](declaration_v1.json)
  and [`declaration_v2.json`](declaration_v2.json).
- Tests: `tests/test_v10.py` (five, also run on arcw0 before the refit:
  `azure_snapshots/edb3588230e9b1f9/tests_azure.xml`) and
  `tests/test_v10_classifier.py` (one); **313 tests pass** locally
  ([`tests_local.xml`](tests_local.xml)).

No Azure worker other than arcw0, no external data, no December label and no
leaderboard-derived rule was used.

## Official result: 267.9646 — a loss of 0.52 s against v9

With the user's approval v10 was uploaded at **06:41:14 UTC** (ETag
`a04b3b42e19faf2bbf75b11693287d85`; read-back SHA-256 matches:
[`submission_remote_verification.json`](submission_remote_verification.json))
and processed at 06:41:48 UTC: **267.9646 seconds RMSE on all 344,841 rows,
0.5192 s (+0.194%) worse than v9's 267.4454**
([`official_result.json`](official_result.json)). The organizer's result
object reports `Succeeded`, 344,841 used pairs. **v9 remains the official
best**; the standing is unchanged at 12th of 142
([`leaderboard_snapshot.json`](leaderboard_snapshot.json), 06:43 UTC; second
place moved to 245.2932).

The development folds predicted −0.378% / −0.579%; the ranking year gave
+0.194%, an SSE increase of about 9.6e7 (≈ 0.13% of v9's ranking SSE). No
row-level attribution is possible without labels and none is attempted; the
prior facts are: the clock blend moves 339,551 rows by a median of 7 s
(paired-day intervals excluded zero on both folds), the routing moves 169
rows, and the day-density classifier moves 372 LIRF population rows by large
amounts — the item whose seasonal-fold evidence was weakest (declaration 1
failed on exactly that interval, and the item's single-seed seasonal score
ranged from −0.04% to −0.24%). Under the plan's original 0.5%-per-fold rule
this candidate would not have shipped; the smaller margin declared here let a
candidate through that did not transfer. Lesson recorded with v8's: changes
concentrated in a few hundred LIRF population rows need evidence that the
seasonal fold cannot supply, and a 0.25% margin is not protection for them.

Never re-upload v1–v10.
