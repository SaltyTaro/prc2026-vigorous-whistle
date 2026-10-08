# Winter-season (W25) late pass on v73 (`season_pass_v1`, 8 October 2026) — ACCEPTED: `w25_e3`; **v74 = 214.4666 official**

Declaration `configs/season_pass_v1.json` (11b29f2), committed before the five-seed bags were fitted; tool `tools/run_season_pass.py`
(7da8df3; `--build` locally, `--fit` on Azure arcw0, `--score` locally); results [`results_v1.json`](results_v1.json); exploration
[`../audit_v72/`](../audit_v72/) (aud11, aud12). No official score was read or used. A build under `runs/submissions` and any upload are the
user's word.

## Finding

Every late pass so far gained strongly in November / December 2025 and little or nothing in January / July 2025. Fitting the raw-input pass
on ONE development month and reading another shows why: adjacent months gain (EGLL Nov->Dec -0.067%, Dec->Nov -0.053%; LIRF -0.058% /
-0.043%), every non-adjacent pair loses, including the same calendar season a year apart (EGLL Dec->Jan 2025 +0.016%). Interpretation:
operations are organised by IATA scheduling season; November / December 2025 and January 2026 are all winter season W25 (26 October 2025 -
28 March 2026). January 2026, a ranking month, sits in the same season as the last two development months.

## What it is

One LightGBM bag per airport (raw ADS-B + second / third passes' predictions + v73 + clocks + class; target clip(label - v73, ±1800); five
seeds, 400 rounds), fitted on November + December 2025 and applied to the airport's non-fixed **January 2026** rows only:
new = max(v73 + clip(0.5 x bag, ±120 s), -12). July 2026, all other airports, the LIRF population and the guard rows keep v73.

## Not a clean test (disclosed in the declaration)

The airports (EGLL, LIRF, LEBL) and the rule (only W25 months can read a W25 layer) were chosen after the single-seed exploration; LFPG /
LTFM lose even on adjacent months, EDDM / LEMD are small, EDDF / EHAM / LSZH have nothing left after v73's third pass.

## Result (complete-month RMSE change against v73's development values; each month predicted by the bag fitted on the other)

| candidate | November (from December) | December (from November) | mean | gain / squared change | Dec -> Jan 2025 (cross-season, descriptive) | transfer (Jan 2026) |
|---|---:|---:|---:|---|---:|---|
| **w25_e3** (EGLL, LIRF, LEBL) | -0.1026 [-0.156, -0.063] | -0.1412 [-0.188, -0.100] | **-0.1219** | 1.47 / 0.96 | +0.031 | EGLL 0.95, LIRF 1.17, LEBL 1.03 |
| w25_e2 (EGLL, LEBL) | -0.0603 [-0.094, -0.035] | -0.0822 [-0.116, -0.053] | -0.0712 | 1.49 / 0.77 | +0.018 | 0.95 / 1.03 |

Every airport-month improves (EGLL -5.66 / -6.15e6 s², LIRF -4.50 / -5.13e6, LEBL -0.76 / -1.01e6). Both pass; the rule selects
**w25_e3**. The cross-season reading worsens, as the mechanism predicts; the layer never applies outside W25.

## Ranking dry run (version provisional)

`runs/season_pass/rank/dry_v74/vigorous-whistle_v74.parquet`, SHA-256 `85e57d05322009d886d7423ba818ec6435f023bb8fd054463dd8adca6983f761`:
v73 with 40,796 January 2026 rows changed (EGLL 18,553, LEBL 11,536, LIRF 10,707 non-population), squared change 1.03e7, max |change| 91 s,
mean -3.6 / +0.5 / -5.9 s (EGLL / LEBL / LIRF). Independently checked: no July row, no fixed row, no LIRF population row (raw rule), guard
rows 25,499 / 47,705 unchanged, minimum -12, int32, template order. Predictions copied back with matching SHA-256 (`runs/season_pass/vm_preds.sha256`).

**Expected effect:** about -0.07 x g s: -0.07 to -0.10 s at development's g = 1.0-1.5; +0.07 s if the change were pure noise.

## Build (only at the user's word)

`docs/season_pass/build_v74.py` (fill in USER_DECISION), then `tools/audit_export.py --version 74 --compare 73 --report
docs/export_audit/v74_audit.json`, then `tools/prc_submit.py upload --file runs/submissions/vigorous-whistle_v74.parquet --expected-sha256
85e57d05322009d886d7423ba818ec6435f023bb8fd054463dd8adca6983f761 --calendar-day`; status `--report-dir docs/season_pass`.
## Official result (8 October 2026)

Built (`build_v74.py`, USER_DECISION "'upload it'") and uploaded at the user's word; audit
[`../export_audit/v74_audit.json`](../export_audit/v74_audit.json) (flag sets identical to v73's). **v74 = 214.4666 official on all
344,841 rows** (processed 07:03:35 UTC 8 October), **-0.1185 s against v73 (214.5851); the official best.** Scored squared-error gain
17.5e6 = 1.70 x the layer's squared change (1.03e7), development 1.0-1.5; nothing re-tuned from the score. Record
[`official_result_v74.json`](official_result_v74.json). Never re-upload v1-v74.
