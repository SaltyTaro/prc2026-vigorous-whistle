# Development table of the shipped model part (2 October 2026)

`tools/build_dev_model_part.py` -> `runs/dev_model_part/rows.parquet` (SHA-256 `f4d75869db5e61eabe92308a55c74536a8b1a397debba35bd9ed3e68e97d25de`,
672,428 rows; record [`build.json`](build.json)). No model is fitted and nothing is selected.

Rows and footing are `runs/lomo_dev/rows.parquet`'s (January / July 2025 eleven-month leave-one-month-out, November the forward fold,
December the January-November refit), identical to `runs/daily_totals/rows.parquet`'s rows; the `v46` column equals that table's `stack`
exactly (max difference 0.0). Columns: `MVT_ID_mvt`, `month`, `day`, `airport`, `label`, `matched`, `lirf_m`, `lirf_u`, and each model-part
layer cumulatively: `v46`, `v51m` (combination member w 0.40), `v52m` (masked member w 0.40), `v53m` (LIRF masked member w 0.55 on
`lirf_u`), `v54m` (dark re-read w 0.70 / share 0.75 on `lirf_u` in dark months: January 2025 only, match share 0.0), `v55m` (matched-row
LIRF member w 0.40 on `lirf_m`). Every layer reproduces its round's recorded reading before the table is written (month RMSEs of v46 /
v51m / v52m; route RMSEs of v53m / v54m / v55m; tolerance 1e-6 s), and the stack's LIRF unmatched values equal the gated control exactly.

| month | v46 | v51m | v52m | v53m | v54m | v55m |
|---|---:|---:|---:|---:|---:|---:|
| 2025-01 | 322.8585 | 322.6450 | 322.5887 | 322.5542 | 322.5451 | 322.5451 |
| 2025-07 | 257.7185 | 257.5356 | 257.4131 | 257.4116 | 257.4116 | 257.4099 |
| 2025-11 | 184.9050 | 184.8231 | 184.6605 | 184.6852 | 184.6852 | 184.6512 |
| 2025-12 | 167.5137 | 167.3137 | 167.1763 | 167.1447 | 167.1447 | 167.0696 |

Use: refitting the published-data steps (v48's cells and scales, the extreme-day betas, v57's variance model and k) on the shipped model
part's residuals instead of the v46-era stack's.
