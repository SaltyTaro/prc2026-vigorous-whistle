# NM-input masking on the eight Azure workers

All **eight model/fold jobs** and **twelve declared candidate recipes** are
complete. **No candidate qualifies** at the fixed requirement of at least 0.5%
lower RMSE on each complete development fold. V7 remains the official best at
**275.0570 seconds on 344,841 rows**. This campaign made no refit, export or
submission, and December's reserved audit remains unused.

The leading mean-relative recipe, `cat_augmented_v7_50`, improves seasonal RMSE
by **0.098084%** and forward RMSE by **0.416614%**. These reused development
scores do not establish a leaderboard gain.

| Complete development population | V7 RMSE, seconds | Leading blend | Improvement |
| --- | ---: | ---: | ---: |
| Seasonal, 344,419 rows | 322.819323 | 322.502689 | 0.098084% |
| Forward, 162,332 rows | 221.138156 | 220.216864 | 0.416614% |

[Development results](development_results.json) contain every candidate and
slice metric. [Analysis](analysis.json) compares augmentation with its matched
controls. [Campaign summary](campaign_summary.json) records the final decision,
resources and worker states.

## Declared comparison

The user explicitly requested parallel testing on all eight existing Azure
VMs. [The authorization receipt](../../infra/azure/parallel_authorization_20260915.json)
supersedes the earlier start-only-arcw0 restriction. No new VM was provisioned.
All jobs used the isolated `/home/azureuser/prc-flight-2026` directory, with
16 native threads and the same pinned packages on each worker. Credentials
remained local.

[The model declaration](../../configs/nm_mask_v1.json), dated
**15 September 2026 at 02:31:16 UTC**, precedes all new model scores.
[The feature declaration](../../configs/features_nm_mask_v1.json) defines the
same 232-predictor NM-blind view for controls and augmented models. Only rows
with both a missing NM origin and no usable NM off-block clock can change;
every other prediction remains exactly at v7. The route contains **5,373
seasonal, 1,356 forward and 5,290 ranking rows**.

The feature transformation reads eleven airport-side columns. It removes NM
information from every movement, including peer movements, rebuilds 80 row
and timing/quality predictors plus 59 ground-context predictors, and reuses
93 airport-only traffic/weather predictors unchanged. No departure target or
block time enters this shared view. The masked conditional schedule offset
matches the existing offset on all 26,111 naturally eligible feature rows.
Full movement and flight IDs are not predictors.

Controls fit only naturally missing records, with unit weights. Augmented
models fit all original fitting rows after the same NM removal. Natural and
synthetically masked cohorts each receive half the fitting weight; total
weight equals the control's natural-row count. Cohort membership is not a
predictor. Both groups use the same original October stopping rows restricted
to naturally missing records, and the same native settings and seed.

| Model variant | Seasonal worker | Forward worker | Seasonal fitting rows | Forward fitting rows |
| --- | --- | --- | ---: | ---: |
| LightGBM natural control | arcw0 | arcw1 | 12,179 | 17,552 |
| CatBoost natural control | arcw2 | arcw3 | 12,179 | 17,552 |
| LightGBM augmented | arcw4 | arcw5 | 1,226,945 | 1,571,364 |
| CatBoost augmented | arcw6 | arcw7 | 1,226,945 | 1,571,364 |

The four individual variants and two equal LightGBM/CatBoost means are each
tested at common 50% and 100% replacement weights on the eligible route.
All twelve recipes must finish before selection. Every original scoring row,
including negative and extreme labels, contributes to complete RMSE. There is
no clipping, hidden-label probing or adaptive airport weighting.

## Complete candidate scores

Lower is better. The 50% CatBoost augmented blend has the lowest mean of the
two ratios to v7, but fails the promotion margin on both folds.

| Candidate | Seasonal RMSE | Forward RMSE |
| --- | ---: | ---: |
| `cat_augmented_v7_50` | 322.502689 | 220.216864 |
| `cat_augmented_v7_100` | 324.321444 | 220.073511 |
| `lgb_augmented_v7_50` | 324.986652 | 222.334914 |
| `lgb_augmented_v7_100` | 329.020129 | 224.450234 |
| `augmented_mean_v7_50` | 323.372140 | 221.093793 |
| `augmented_mean_v7_100` | 325.190514 | 221.536516 |
| `cat_natural_v7_50` | 325.420163 | 220.037691 |
| `cat_natural_v7_100` | 330.785286 | 220.157264 |
| `lgb_natural_v7_50` | 324.051662 | 222.965769 |
| `lgb_natural_v7_100` | 328.313862 | 225.838107 |
| `natural_mean_v7_50` | 324.315862 | 221.344581 |
| `natural_mean_v7_100` | 327.890284 | 222.371515 |

Augmentation improves the full-replacement CatBoost control by 1.9541%
seasonally and 0.0380% forward. It improves LightGBM forward but worsens it
seasonally. This supports a limited benefit from extra masked fitting examples,
not promotion of this recipe. The experiment removes peer NM context as well
as focal NM fields. **Focal-only masking that retains other flights' NM
context remains untested**; no follow-up campaign was launched.

## Execution and verification

The initial launch completed arcw0's native fit. Seven fresh workers failed
during pytest import because their source search path was missing, before any
competition fit. The separate resume runner added an explicit `PYTHONPATH`;
original logs and status files remain preserved. The seven resumed workers
started their job commands at a common 02:57:13 UTC barrier. Every declared
native model was fitted once. See [launcher correction](launcher_fix.json),
[initial launch](launch.json) and [worker mirrors](mirrors/).

All **34 campaign tests pass locally and on every VM**. Four further verifier
regression tests pass on Windows and arcw4. A JSON month-key representation
mismatch was corrected only in the verifier, with collision and changed-value
controls. [Its original failure and correction](verifier_fix/) are preserved;
no model, feature, configuration or prediction changed.

Both [Windows](local_verification.json) and [Azure](verification.json)
independent verifiers reproduce **26,916 native prediction values exactly**
and all **24 complete candidate/fold scores**. They check fitting-only
encoders, fitting weights, protected split identities, original routing,
October stopping IDs and counts, native settings, full populations and
unchanged fallbacks. Independent blend arithmetic differs by at most
9.10e-13 seconds. [The feature verifier](local_feature_verification.json)
checks 2,264,211 rows, 210,571,623 unchanged inherited values, all 26,111
eligible offsets and 186,816 reconstructed row/ground values with zero
difference. [Combined test identities](tests_combined.json) distinguish the
34 campaign checks from the four later verifier checks.

The eight measured native fits total **369.29 seconds** of wall time; the
longest takes **126.58 seconds**. Maximum sampled memory across the jobs is
**7.83 GiB RSS**. This sum is not an elapsed-time speedup estimate.
[Final worker checks](pool_final_checks.json)
verify each worker's source, inputs and preserved results and require no active
project Python jobs before release. [Release receipts](pool_release.json) and
[final VM states](pool_final_state.json) record restoration of arcw1–arcw7 to
their original deallocated state, with arcw0 kept running. VM disks and all
artifacts are retained.

## Reproduction and immutable evidence

- Source: `src/prc2026_nm_mask/`, model-code SHA-256
  `10b798a7b702149ab86c25d6967c4c37f6296f08bd8f4a9ece9c40820c81b1d0`.
- Separately bound job preparation/loader SHA-256:
  `933ece59e925d90cd71596bf1f64c11ba61f6bb01e1a7ef798742a5ceda73979`.
- Feature bundle: `data/nm_mask/aa7dc17c8794d1b5e1d1`, full identity
  `aa7dc17c8794d1b5e1d10d286720d9ce64a80cf46b5cbb6ecd7be31271d9061d`;
  manifest SHA-256
  `bf45c3ab762d2a7e16ef12c82cae5328c76c1188d741b0fca008a45906c187ca`.
- Selection SHA-256:
  `c4f2add99e36393835878f8d7fb2cce5a880bd8a3c226fe53090a651afeff018`.
- Azure originals: each assigned worker's `runs/nm_mask/<job_id>/`.
  Local mirror: `runs/azure_parallel_nm_mask/`; worker archives:
  `runs/nm_mask_results/`; portable inputs: `runs/nm_mask_inputs/`.
- Use [jobs_bound.json](jobs_bound.json). The initial job manifests are
  retained; a pre-fit overlay adds the explicit loader-source hash without
  changing any data, split, model setting or reference prediction.

To verify an existing local mirror, use a fresh output filename:

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/verify_nm_mask.py --output docs/nm_mask/verification_reproduction.json
```

Reproduction of an individual job uses `tools/run_nm_mask.py --job` with its
path from `jobs_bound.json`, the assigned host and a clean isolated output
directory. Preserve existing native run directories. The deployment helper
refuses differing existing files and verifies package and member hashes.

[Archive receipts](archive_receipts.json) bind the final source/evidence,
NM-blind feature bundle, eight portable job packages, verification inputs and
eight result archives. [Preservation evidence](preservation_final.json) checks
all fourteen raw files by byte hash, v1–v7 exports, v7 official artifacts and
earlier scientific archives. Previous narrative versions remain in
`overviews_before/`. Prior model packages and submissions are unchanged.
