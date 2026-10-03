# No-NM rematch at EDDF / EHAM / LSZH (`configs/rematch_nonm_v1.json`) — NOT ACCEPTED by its rule (3 October 2026)

Governing tool `tools/run_rematch_nonm.py`; results [`results_v1.json`](results_v1.json); ranking dry run
[`ranking_dry_run.json`](ranking_dry_run.json). No official score, no leaderboard, nothing under `runs/submissions`, nothing uploaded.

## Mechanism

No-NM departures are keyed to ADS-B on `FLIGHT_mvt`, which at EDDF / EHAM / LSZH is the flight number, not the ATC callsign, so they
almost never match there (v35's IATA-prefix fix misses alphanumeric callsigns and leading zeros). A key learned from the same flight
number's NM-present departures (label-free) finds a gate segment at the known stand for 1,176 such 2025 rows and 343 ranking rows. The
model part moves halfway to the ADS-B taxi plus a per-airport offset (fitting months: EDDF +84..+91, EHAM +1, LSZH +48 s), before v48's
reconciliation, the extreme-day layer and the variance re-allocation, all with their shipped parameters.

## Governing run (control = the unchanged v55m through the same chain; reproduces `runs/scout/refit/ctl_v55m.parquet` exactly)

| month | layer rows | control RMSE | candidate | change | paired-day 95% (s) | days better |
|---|---:|---:|---:|---:|---|---:|
| January 2025 | 73 | 322.0588 | 321.8975 | −0.0501% | [−0.470, −0.014] | 22 / 32 |
| July 2025 | 241 | 255.9946 | 255.9450 | −0.0194% | [−0.097, −0.016] | 24 / 32 |
| November 2025 | 39 | 184.0052 | 183.9852 | −0.0109% | [−0.038, −0.007] | 19 / 31 |
| December 2025 | 77 | 166.2473 | 166.2119 | −0.0213% | [−0.073, **+0.003**] | 24 / 32 |

Equal-footing mean −0.0302%. Twelve months (older predictor, no chain): 12 of 12 better, every interval below zero, mean −0.0236%.

**Rule:** (1) no month worse: yes; (2) mean ≤ −0.02%: yes; (3) every equal-footing interval below zero: **no** (December's upper end
+0.003 s); (4) twelve months: yes. **Not accepted.** Shipping it would be the user's override.

The gain is concentrated: the best ten rows carry 91 / 144 / 94 / 99% of each month's gain (January 2025's EHAM winter day 5 January
alone carries about half of January's). The chain absorbs little: the squared change after the chain is 101 / 89 / 96 / 98% of the
model part's.

## Ranking dry run (label-free)

The chain on the unchanged base `runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet` (`266b8c93...`) reproduces v57 byte for
byte (`d1c390d4...` -> `aa02ff59...` -> `31d909ff...`). On the base with the layer (`runs/rematch_nonm/base/vigorous-whistle_v55m_rematch.parquet`,
`75957f82...`, 342 rows changed, squared change 1.21e8) the chain gives
`runs/rematch_nonm/rank/Y/vigorous-whistle_v58.parquet`, **SHA-256 `c67f20d61f8a85770a68695268e9f49449cc0157970ffc2b9e83044c458201ca`**:

- against v57: 8,982 rows differ (342 layer rows; 8,640 others through the day shifts and the re-allocation), squared change
  **1.231e8** (layer rows 1.226e8, others 5.3e5), max 2,196 s; EHAM January 1.147e8, LSZH January 4.7e6, EHAM July 2.9e6, EDDF 0.7e6;
- published daily totals after the chain, on the 11 airport-days with at least five layer rows: |published − implied| 52.9 h with v57,
  38.8 h with the layer, closer on 9 of 11 days (all days 802.7 -> 788.5 h);
- arithmetic only: at the 2025 gain per unit of squared change (0.66–2.6 on the development months, 0.74–2.95 on the twelve) the
  change is worth −0.5 to −2.4 s; it breaks even at a gain per unit of 0 (the moves half right in the squared sense) and costs +0.8 s if
  the moves carried no information at all (gain per unit −1, never seen in any 2025 month or airport-month cell, lowest 0.46).

The intruder layer is not applied here.

## v59: user override, built and uploaded ("Override: build + upload v59", 3 October 2026)

The declared rule was not met (December's paired-day interval [-0.073, +0.003] s); the user overrode it after seeing the label-free
published-total check (gap on the 11 most affected airport-days 52.9 h -> 38.8 h, 9 of 11 closer) and the precedent of v35's callsign fix.
v59 = v55m model part + this rematch -> v48 -> extreme-day layer -> variance re-allocation (`runs/rematch_nonm/rank/Y`, c67f20d6...) ->
the intruder stand bound last (`tools/run_intruder_bound.py --build --version 59 --base ...`; receipt `docs/intruder_bound/export_v59.json`,
audit `docs/export_audit/v59_audit.json`, clean). SHA-256 `2a84b71361f8244faa251e3f3606d61f2f21d28ea84c660851be936652fa570e`;
8,980 rows differ from v58 (EHAM 6,494, LSZH 1,941, EDDF 545), squared change 1.231e8.

**Official: 218.8483 on all 344,841 rows (processed 08:58:43 UTC 3 October 2026), -2.0729 s against v58 (220.9212); the official best.**
Record [`official_result_v59.json`](official_result_v59.json). Squared-error gain 3.14e8 = 2.55 x the squared change, inside the 2025
range quoted before the upload (0.74-2.95). Nothing is re-tuned from the score. Never re-upload v1-v59.
