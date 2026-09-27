# Weather ceiling on v30's development residuals (27 September 2026). Descriptive: nothing downloaded, fitted or selected

This note covers step 3 of the gap plan: is an ERA5 or other open-weather screen worth a VM round? It uses
`tools/weather_ceiling_check.py`, which writes [`weather_ceiling.json`](weather_ceiling.json). The inputs are:

- v30 rebuilt per row on the four recipe months (`runs/error_map/dev_rows.parquet`, from `tools/v30_error_map.py`);
- the frozen METAR weather columns of the timeline bundle, with December's taken from the December build.

Only ordinary labels (at most 3 h) are used, as in the error map: 672,350 rows, RMSE 196.88 s.

## 1. Ceiling for any covariate that varies only by airport and time

If v30 were corrected by the true mean residual of each group, this much squared error would disappear. The figures are
noise-adjusted: each group's term is n·mean² − within-group variance.

| group | groups | share of SSE removable |
|---|---:|---:|
| airport × day | 1,230 | **1.02%** (each month 0.92–1.08%) |
| airport × day × 3-hour block | 9,073 | **3.13%** |
| airport × day × hour | 25,075 | **5.37%** |

These shares are upper bounds. They cover weather, and also every traffic or operations effect that is shared by the departures
of an airport-hour. The bias is diffuse:

- The largest single airport-day is worth 0.05% (LTFM, 13 July).
- The 100 largest 3-hour blocks together are worth 1.6%. Most are July afternoons (convective) at EGLL, LTFM and LEBL, plus
  a few LFPG morning blocks.

## 2. v30 is already unbiased in METAR weather states

ERA5 would add intensity (precipitation amount, snowfall rate) and snow on the ground to the METAR states the models already see.
The bias-removal column below is n·mean² as a share of SSE.

| state | rows | mean residual (s) | bias removal |
|---|---:|---:|---:|
| snow now | 2,810 | −15.2 ± 4.5 | 0.002% |
| snow in the last 3 h, not now (snow-on-ground proxy) | 5,456 | −3.4 ± 2.8 | 0.000% |
| freezing weather | 2,531 | +37.5 ± 7.1 | 0.014% |
| temperature ≤ 0 °C, no snow in 3 h (frost / de-icing proxy) | 40,253 | +3.2 ± 1.0 | 0.002% |
| thunder in the last 3 h | 6,834 | +21.2 ± 4.5 | 0.012% |
| fog | 17,790 | +6.8 ± 1.7 | 0.003% |
| rain | 71,503 | −3.0 ± 0.8 | 0.002% |

In winter months, cold hours (≤ 1 °C) against warm hours per airport differ by at most 35 s. The largest case is LFPG: +31 s on
9,191 cold rows, 0.03% of SSE.

## Reading

For open weather to help, it would have to explain variance *within* these states, and at the scale of an airport-hour. The total
ceiling there is 3–5% of SSE, and most of it is shared traffic and operations rather than weather. Realistically ERA5 would be worth
hundredths of a percent, and it would still need a new-source licence record and a full development round. **Recommendation:
drop the ERA5 part of step 3.** Spend the Azure capacity after round E on the unmatched-row specialist (Option A) instead.
