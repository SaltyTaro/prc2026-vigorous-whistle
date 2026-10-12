# Development-month reading of final candidate 4's six-month learners (12 October 2026)

**Question.** Final candidate 4 (`8fc8828d...`, [`final_candidate4.md`](final_candidate4.md)) uses C6 guarded stacker and C14 second pass
learners also fitted on Tier B's February / June 2025 rows. Their only readings were on February / June 2025, yet they also move January and
July 2026 rows: squared change 1.28e7 and 1.47e7 against candidate 3. No reading existed there because the development months sit inside
both fits. The analogue: does adding the Tier B rows help a development month that the learner did not see?

**Declaration.** [`configs/sixmonth_devmonths_v1.json`](../../configs/sixmonth_devmonths_v1.json), commit `4d3c8c1`, made before any fit. A reporting
addendum, `3a21984`, was also committed before any fit: it adds by-airport figures and LIRF's level shift on non-population rows. It does
not change the rule. Tool: `tools/run_sixmonth_devmonths.py`. No official score was used, and nothing was uploaded.

**Method.** For each development month m, each learner was fitted on the other three development months, (a) without and (b) with both
Tier B months' rows. The Tier B rows are selected exactly as the final file's fits select them (323,246 rows for C6, the same count as
candidate 4's six-month fit). Both variants then predict month m. The fitting code is `tools/run_sixmonth.py`'s, unchanged:

- C6: `stack_fit_predict`, ten seeds x p2 / p3 / po bags, guards from the three-month support. Values come from `run_stack_weight.rank_value`
  at 0.65.
- C14: `pass_fit_predict` with `sp_hicov7`. Value = v71 + clip(0.5 x prediction, ±120 s), floored at -12 s.

Each comparison is (b) against (a) at the layer's position, with a paired-day 95% interval (2,000 resamples, seed 20261004).

## Readings ([`devmonths_results_v1.json`](devmonths_results_v1.json), SHA-256 `44f13229...`)

| held month | C6 change [95% interval] | C6 dSSE (s²) | C14 change [95% interval] | C14 dSSE (s²) | C6 + C14 dSSE |
|---|---|---:|---|---:|---:|
| **2025-01** | -0.0205% [-0.0917, -0.0022] | -6.50e6 | -0.0199% [-0.0628, -0.0093] | -6.24e6 | **-1.27e7** |
| **2025-07** | -0.0264% [-0.0517, +0.0025] | -6.50e6 | -0.0399% [-0.0648, -0.0152] | -9.55e6 | **-1.60e7** |
| 2025-11 | -0.0080% [-0.0422, +0.0282] | -8.66e5 | -0.0323% [-0.0485, -0.0192] | -3.44e6 | -4.31e6 |
| 2025-12 | +0.0225% [-0.0312, +0.0772] | +2.00e6 | -0.0121% [-0.0257, +0.0033] | -1.06e6 | +9.44e5 |

Rows moved: C6 112k / 135k / 117k / 114k, C14 77k / 104k / 75k / 77k. The January and July C6 figures (-6,497,278 and -6,496,972 s²) are
close by coincidence; the RMSE and moved-row counts differ.

**Control.** C14's (a) equals the chain's stored out-of-month development predictions (`preds_dev_stored.parquet`) exactly in all four months
(max |difference| 0). C6 has no stored analogue on this table: `run_stack_weight`'s development predictions were made on the pubrefit ship
table, whose `new` differs on 358,372 rows. Both C6 variants share the code and the table and differ only in the extra rows.

## Decision (pre-set rule)

The C6 + C14 squared-error change is negative on both analogue months: January -1.27e7 s² and July -1.60e7 s². So **January and July 2026
keep candidate 4's values**. February / June 2026 keep candidate 4 by declaration. **No candidate 5 is built: candidate 4
(`8fc8828d1f5338bda08972f16af61c0f5f288b5d50992c1750d4d768d89b0a42`) stands.** Candidate 3 (`b673791d...`) remains the fallback.

Supporting months: November is better (-4.31e6). December is slightly worse overall (+9.44e5): C6 is worse there (+2.00e6, of which LIRF
+2.08e6 and EGLL +0.83e6), while C14 is better (-1.06e6).

## Addendum: by airport, and LIRF's level

C6 lowers LIRF's non-population rows in every held month: by -3.89 s in January, -3.78 s in July, -2.43 s in November and -1.95 s in
December. The squared error on those rows moves the other way in two months:

| month | LIRF non-population rows | mean change | dSSE (s²) |
|---|---:|---:|---:|
| 2025-01 | 11,254 | -3.89 s | -1.41e6 |
| 2025-07 | 14,856 | -3.78 s | **+2.40e6** |
| 2025-11 | 11,272 | -2.43 s | -4.35e5 |
| 2025-12 | 11,980 | -1.95 s | **+2.08e6** |

So adding the Tier B rows (built from a 4-seed model part) shifts LIRF's level down on full-seed rows too, and LIRF's net over the four months is
+2.6e6 s². The shift seen on candidate 4's LIRF June / July 2026 rows (-9.3 / -10.2 s) is larger than any month here, and those months have
no ADS-B at LIRF. C14 does not touch LIRF.

The C6 gains are elsewhere: July LSZH -4.29e6 (-1.20%), LTFM -2.56e6 and LFPG -1.80e6; January EGLL -1.92e6, LFPG -1.58e6 and LSZH -1.26e6.
Every month's per-airport dSSE, % and mean change are in the results JSON under `by_airport`.

## VMs and files

- **arcw0** fitted C6 for January / July, **arcw1** C6 for November / December, and **arcw3** all of C14 and the reading. The work ran
  09:50-10:20 UTC on 12 October. Each VM's repository was overlaid on its local NVMe ([`devmonths/mount.sh`](devmonths/mount.sh)).
- Code was copied by `git archive` at `3a21984` plus `.git`. The fit records hash `run_sixmonth_devmonths.py` as `3c97fed5...`, the committed
  LF blob; the Windows working copy is CRLF.
- Inputs: 88 files, 1.68 GB ([`devmonths/want.txt`](devmonths/want.txt)), hash-equal on every VM.
- Outputs: 16 fit files ([`devmonths/fits.sha`](devmonths/fits.sha)) plus logs (`devmonths/logs_*`), mirrored and hash-checked. Every
  `.enoent` lists only outputs that did not exist yet. **All eight VMs are deallocated.**
- **Duplicate launch, disclosed.** The local launch command hung, and its `&&` chain later started `jobs.sh` a second time:
  - on arcw1 at 10:11:50 and on arcw3 at 10:13:04;
  - the never-overwrite guard refused every output that already existed;
  - the duplicates' arcw1 b fits and arcw3 `ss_2025-12_a` fit were killed within seconds to about a minute, before writing;
  - every file comes from the first launch;
  - those jobs shared strace trace files, so their `.enoent` lists are incomplete. They are diagnostic only.
- Fit outputs are kept under `runs/sixmonth/devmonths/` (git-ignored).
