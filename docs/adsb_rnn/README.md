# adsb_rnn: a neural-network member of the residual stage (24 September 2026)

**Result: nothing selected under the declared rule** -- every candidate improves all four recipe months and the pooled twelve-month
residual scope, but January 2025's paired-day interval includes zero for every candidate, which the rule does not allow. **The user
then chose a recorded override** (`override.json`): `rnet_w90` was exported as **vigorous-whistle_v26**, verified, audited and
uploaded, and **officially scored 234.1438** (-0.7137 s / -0.304% vs v25's 234.8575), **1st of 186** at 11:20 UTC on 24 September
(`leaderboard_after_v26.txt`). v26 is the incumbent; never re-upload v1-v26.

## Override, export and official result

- `override.json`: the user's answer after the null selection was reported ("Override: upload w90"); the only failed condition is
  January's interval ([-0.341, +0.114] s); w90 is the rule's own tie-break (lowest mean ratio) among the three. Not a precedent for
  relaxing the rule in later rounds.
- Export (`tools/export_adsb_rnn.py --version 26 --override docs/adsb_rnn/override.json`, `export.json`):
  `runs/submissions/vigorous-whistle_v26.parquet`, SHA-256 `7357eb742f89d79f3756ae8f01e24fe1c68e7a03ca2062badfb0e0f4c9d2b00f`; the rebuilt
  control equals the uploaded v25; 197,069 rows change (RMS 12.19 s, mean -0.42 s, none above 600 s; LIRF untouched).
- Verification (`tools/verify_adsb_rnn.py`, `verification.json`): all 16 refit networks reload and re-predict every in-scope ranking
  row within 2.1e-3 s; plain arithmetic reproduces every exported integer; rows outside the scope equal the uploaded v25.
- Audit (`tools/audit_export.py --version 26 --compare 25 24 9`, `export_audit_v26.json`): v25's shape (same January airport-month
  flags, negative-proxy and LIRF-population rows; 851 predictions above one hour against v25's 884).
- Uploaded with the user's approval (`tools/prc_submit.py upload --calendar-day`, second upload of 24 September UTC); processed
  11:19:41 UTC: **234.1438 on all 344,841 rows** (`official_result.json`). The official gain (-0.304%) is at the December read
  (-0.28%) and above the recipe-month mean (-0.129%), as in the earlier network rounds.

## What was tested

The residual stage learns `label - v9` (cross-fitted v9) on matched rows outside LIRF and enters v25 as
`0.3 x (v9 + residual) + 0.7 x committee mix`. v25's residual is the mean of two trees, `adj2` (month-weighted LightGBM) and `cat`
(CatBoost). This round adds the committee networks' architecture (`src/prc2026_nn`, unchanged) on the same target, rows and
twelve-month leave-one-month-out (`tools/run_adsb_rnn.py`).

- **Design** (`configs/adsb_rnn_design_v1.json`, `design_report.json`, October only): `rnb` (the trees' 33 columns) and `rwide`
  (+ the committee's 287 timeline/v19 columns), 4 seeds each. rnb's 4-seed mean alone beat the tree pair on October's scope
  (113.85 vs 114.29 s) and read -0.075% on the October recipe at w 0.9; rwide blended better on the scope but read only -0.039%
  (it overlaps the committee). Go, rnb.
- **Declaration** (`configs/adsb_rnn_v1.json`, 09:33:52 UTC, commit `54b2fbe`): network `rnet` = 16 rnb seeds (501-516);
  candidates `rnet_w50/w70/w90` = `(1 - w) x mean(adj2, cat) + w x network`; control v25 (reproduced to 1e-9 s on every recipe
  month); rule = no recipe month worse, mean at least 0.02% below the control, every touched month's paired-day 95% interval below
  zero, pooled twelve-month scope RMSE below the control's.

## Development result (`development_results.json`, `runs/adsb_rnn/dev/`)

| candidate | Jan 2025 | Jul 2025 | Nov 2025 | Dec 2025 | mean | pooled scope | intervals < 0 |
|---|---|---|---|---|---|---|---|
| rnet_w50 | -0.0232% | -0.0552% | -0.0920% | -0.1881% | -0.0896% | -2.030% | 3/4 (Jan [-0.232, +0.026] s) |
| rnet_w70 | -0.0262% | -0.0722% | -0.1117% | -0.2412% | -0.1128% | -2.259% | 3/4 (Jan [-0.295, +0.063] s) |
| rnet_w90 | -0.0256% | -0.0862% | -0.1217% | -0.2815% | -0.1287% | -2.149% | 3/4 (Jan [-0.341, +0.114] s) |

- The 16-seed network alone beats the tree pair on the residual scope in **all twelve months** (e.g. February 96.45 vs 100.04 s,
  December 113.87 vs 117.59 s). The pooled scope gain (-2.26% at w 0.7) is the largest of any residual-stage round (adsb_v25's
  selected pair: -1.96%).
- Days with lower squared error (w70): January 23/30, July 29/31, November 25/30, December 29/31.
- January (descriptive, after the read): the residual scope carries only 7.3% of January's squared error; the candidate lowers the
  month's SSE by 8.6 M s^2 of 16,516 M, but a handful of long EGLL rows (labels 4,700-10,600 s, moved by a few hundred seconds either
  way) make the day-level interval straddle zero.

## Disclosures

- A first smoke test of the scorer (before a fix) printed recipe-month readings of a committee-mix weight 0.65 with v25's own
  residual (Jan -0.056, Jul -0.046, Nov -0.073, Dec +0.038%); recorded in the declaration; no such candidate was declared.
- An interim read of the four recipe months (scratch script importing the declared scorer's functions) was taken at 10:00 UTC, before
  the other eight months finished; it equals the declared scoring exactly.

## Artifacts

- Networks: `runs/adsb_rnn/rnb_s501..s516/<month>/` (192 development fits) and `/refit12/` (16 twelve-month refits on 1,113,566
  rows, predicting the 216,435 in-scope ranking rows from `runs/adsb_rnn/ranking_nb.parquet`, whose scope equals v25's row for row).
- Ship tools prepared before selection: `tools/export_adsb_rnn.py`, `tools/verify_adsb_rnn.py` (not run: nothing was selected).
- Compute: arcw0-arcw7, two 8-thread lanes each, 09:36-10:47 UTC; all workers deallocated afterwards.
