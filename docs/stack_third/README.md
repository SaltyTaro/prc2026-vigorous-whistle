# Third guarded residual pass on v72 with raw ADS-B inputs (`stack_third_v1`, 8 October 2026) — ACCEPTED: `t3_ro_3`; **v73 = 214.5851 official**

Declaration `configs/stack_third_v1.json` (a26f565), committed before the five-seed bags were fitted; tool `tools/run_stack_third.py`
(a4994a5; `--build` locally, `--fit` on Azure arcw0, `--score` locally); results [`results_v1.json`](results_v1.json); the exploration that
led here is [`../audit_v72/`](../audit_v72/) (aud5). No official score was read or used. A build under `runs/submissions` and any upload are
the user's word.

## What it is

The stacker and the second pass (v72) only ever combined the components' predictions; neither saw the raw ADS-B match inputs. This pass
learns clip(label - v72, ±1800) from those inputs (features_v19own's 26 columns, 17 matches2 columns: first point, segment / motion start,
stops, path, gaps, surface traffic, own-aircraft points), the second pass's prediction and change, v72's value, the clock columns, row
class and airport: new = max(v72 + clip(0.5 x bag, ±120 s), -12), five seeds, 400 rounds, fitted and applied on non-fixed rows of EDDF,
EHAM and LSZH. Every other row keeps v72.

## Not a clean test (disclosed in the declaration)

The three airports were chosen after a single-seed exploration in which EGLL worsened January / July (its July 2025 coverage collapsed:
77% of rows unmatched, 3% in July 2026) and EDDM / LEBL / LEMD were mixed. No fresh-month check is possible.

## Result (complete-month RMSE change against v72's development values, paired-day 95% interval)

| candidate | January | July | November | December | eq mean | gain / squared change | transfer |
|---|---:|---:|---:|---:|---:|---|---|
| **t3_ro_3** | -0.0191 [-0.075, -0.006] | -0.0203 [-0.031, -0.011] | -0.0617 [-0.093, -0.039] | -0.0550 [-0.070, -0.042] | **-0.0315** | 1.78 / 1.31 / 3.14 / 2.09 | EDDF 1.61, EHAM 1.24, LSZH 1.08 |
| t3_ro_h7 | -0.0159 [-0.065, -0.004] | -0.0197 [-0.031, -0.010] | -0.0513 | -0.0437 [-0.060, -0.030] | -0.0264 | 2.17 / 1.76 / 4.01 / 2.52 | 1.69 / 1.35 / 1.12 |

Both pass clauses 1-4; the rule selects **t3_ro_3** (lower eq mean).

## Ranking dry run (version provisional)

`runs/stack_third/rank/dry_v73/vigorous-whistle_v73.parquet`, SHA-256 `649e99725ee3f700041b07a231dcd2fa656fdbeef3aa8db5b5b9c9fc341af3ba`:
v72 with 86,351 rows changed (EDDF 33,839, EHAM 31,759, LSZH 20,753), squared change 1.04e7, max |change| 113 s, mean -0.6 s (per
airport-month mean -3.4..+1.8 s, RMS 5-15 s). Independently checked: no other airport, no fixed row, guard rows 25,499 / 47,705 unchanged,
minimum -12, int32, template order. Predictions copied back from arcw0 with matching SHA-256 (`runs/stack_third/vm_preds.sha256`).

**Expected effect:** about -0.07 x g s: -0.09 to -0.22 s at development's g = 1.3-3.1; +0.07 s if the change were pure noise.

## Build (only at the user's word)

`docs/stack_third/build_v73.py` (fill in USER_DECISION), then `tools/audit_export.py --version 73 --compare 72 --report
docs/export_audit/v73_audit.json`, then `tools/prc_submit.py upload --file runs/submissions/vigorous-whistle_v73.parquet --expected-sha256
649e99725ee3f700041b07a231dcd2fa656fdbeef3aa8db5b5b9c9fc341af3ba --calendar-day`; status `--report-dir docs/stack_third`.
## Official result (8 October 2026)

Built (`build_v73.py`, USER_DECISION "'upload it'") and uploaded at the user's word; audit
[`../export_audit/v73_audit.json`](../export_audit/v73_audit.json) (flag sets identical to v72's). **v73 = 214.5851 official on all
344,841 rows** (processed 04:46:33 UTC 8 October), **-0.3419 s against v72 (214.927); the official best.** Scored squared-error gain
50.6e6 = 4.87 x the layer's squared change (1.04e7), above development's 1.3-3.1; nothing re-tuned from the score. Record
[`official_result_v73.json`](official_result_v73.json). Never re-upload v1-v73.
