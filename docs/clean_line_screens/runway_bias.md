# Per-runway monthly taxi-out figures: screened, not worth fetching (2 October 2026)

Question: EUROCONTROL's AIU airport dashboard shows monthly taxi-out per runway (never fetched). The daily totals (v48's reconciliation)
already carry each airport-day's level, so a per-runway monthly figure could only add the split between runways inside an airport-month.
Does the stack carry a per-runway bias inside airport-months after the reconciliation?

Screen (`tools/screen_runway_bias.py`, development months only, labels read; nothing declared or exported): v46 development stack
(`runs/daily_totals/rows.parquet`) with v48's reconciled values (`runs/daily_totals/daily_totals_v1/`), runway from the supplied
`RUNWAY_mvt`. Oracle upper bound: within each airport-month, shift every row of a runway by minus (that runway's mean residual on valid
rows, labels in (0, 7200], minus the airport-month's), i.e. a perfect same-month per-runway figure.

| month | month SSE | oracle change |
|---|---:|---:|
| 2025-01 | 1.60e10 | −0.014% |
| 2025-07 | 1.26e10 | −0.061% |
| 2025-11 | 5.51e9 | −0.119% |
| 2025-12 | 4.61e9 | −0.155% |

- This is an in-sample bound: it fits each runway-month mean from the same labels, so part of it is noise (about 45 groups per month).
  The largest single cell is LIRF December (one small runway group 196 s off, −0.09% of the month), not something a published
  valid-flight mean would fix.
- Runway biases do not persist: month-to-month correlations of the within-airport runway bias are −0.49 to +0.03, so the model part
  carries no systematic per-runway error.
- At most about 0.1 s in the best case before the published figures' own noise, against a Power BI scrape. **Closed.**
