# v30 error map (27 September 2026) — descriptive, nothing selected or exported

`tools/v30_error_map.py` → [`v30_error_map.json`](v30_error_map.json). v30's recipe is rebuilt per row on the four recipe months
(committee mix `c_e24x32_e32h` + round B's LIRF blends `lctx_b100_up`) after three exact reproductions (v28, `c_e24x32_e32h` with v25's
LIRF blends, v27 + `lctx_b100_up`; all to 1e-9 s). v30 development RMSE: January 325.68, July 262.11, November 185.44, December 168.80.
Rows are split into label-free cells (airport × ADS-B gate/dark/unmatched × NM clock × LIRF fill population); each ranking departure
gets its cell's development mean squared error (season-matched month first, then all four months, then coarser cells). No ranking label
exists or is inferred.

**Projected ranking RMSE from ordinary development error rates (labels ≤ 3 h): 213–221 s** (season-matched 220.75, recent 213.36;
January 2026 178–197, July 2026 238). With the development months' extreme rows included it is 242–310 s. The official v30 is 229.60:
**8–14% of its squared error is not explained by ordinary development error rates** (1.4–2.5e9, 1.9–3.4× the 7.4e8 gap to first place)
— extreme rows the ranking contains and ranking-only regimes; the map cannot separate them.

Projected share of the ranking's ordinary squared error (season-matched / recent):

| group | ranking rows | projected RMSE (s) | share |
|---|---:|---:|---:|
| LIRF unmatched (15,528 of them July 2026, the ADS-B-dark month) | 17,731 | 401–406 | 17.0–18.7% |
| LIRF fill population (v9 mixture rows) | 383 | 2,661 | 16.1–17.3% |
| LTFM unmatched (no ADS-B at all) | 47,579 | 228–233 | 15.4–15.7% |
| LFPG unmatched | 31,557 | 233–244 | 10.9–11.1% |
| EGLL dark + gate | 37,034 | 191–217 | 9.4–9.6% |
| LIRF dark (matched) | 7,375 | 272–346 | 3.5–5.3% |

LIRF is about 40% of the projected ranking error on 8% of the rows.

**LIRF clock-present schedule fills** (block time = scheduled time ±5 s, label = takeoff − schedule): 13–19% of LIRF clock-present rows in
every month of 2025 (July lowest, 13%). On the development months they carry **5.0% of v30's squared error** (January 1.7%, July 7.8%,
November 5.4%, December 8.1% of the month), mostly where the schedule interval is 30 min–2 h (fill RMSE 649–2,099 s against 277–487 s for
non-fills in the same bands). Operator separates them strongly (fill share 1.5%–73% across operators with ≥500 rows) but `airport_operator`
is already a model feature, and the v10 prototype of a clock-present fill mixture (intervals > 1 h only) gave −0.10% / −0.02%.
EOBT_1 = schedule in 68% of fills against 35% of non-fills. A perfect separator is a ceiling of −5% SSE; an imperfect one is worth a
fraction of it.
