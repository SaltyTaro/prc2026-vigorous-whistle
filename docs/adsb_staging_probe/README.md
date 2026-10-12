# adsb.lol staging-0 vs prod-0 surface coverage probe (12 October 2026)

Question: do adsb.lol's `vYYYY.MM.DD-planes-readsb-staging-0` releases (repo `adsblol/globe_history_2026`) hold airport-surface positions
that the `prod-0` releases we extract lack, at the known gaps (EDDF February 2026 ground ~3,000 rows/day against ~25,000 in January;
LIRF June / July 2026 dark; LFPG / LTFM thin)?

Staging receipts: `surface_<day>_staging-0.json` in this folder
(the parquet files stayed on arcw7's disk). 0 corrupt members.

**Result: no gain.** Staging carries the same coverage as prod, within -0.6% to +6% per airport-day (June 10 staging is up to 5% larger
everywhere: 434 more trace members). EDDF February ground 3,182 vs 3,189; LIRF ground 0 in June / July in both; LFPG within 2%;
LTFM ground 0 in both. Not a lever; nothing to re-extract.

| day | airport | rows prod | rows staging | ground prod | ground staging |
|---|---|---:|---:|---:|---:|
| 2026-02-09 | EDDF | 63,567 | 63,602 | 3,189 | 3,182 |
| 2026-02-09 | EGLL | 109,410 | 109,903 | 42,670 | 42,626 |
| 2026-02-09 | LFPG | 24,419 | 24,355 | 4,511 | 4,507 |
| 2026-02-09 | LIRF | 31,433 | 31,422 | 13,281 | 13,276 |
| 2026-02-09 | LTFM | 1,208 | 1,204 | 0 | 0 |
| 2026-06-10 | EDDF | 116,219 | 119,623 | 18,734 | 19,282 |
| 2026-06-10 | EGLL | 204,574 | 212,273 | 124,798 | 129,509 |
| 2026-06-10 | LFPG | 31,426 | 32,104 | 6,253 | 6,344 |
| 2026-06-10 | LIRF | 577 | 574 | 0 | 0 |
| 2026-06-10 | LTFM | 1,525 | 1,538 | 0 | 0 |
| 2026-07-10 | EDDF | 96,020 | 95,968 | 11,730 | 11,747 |
| 2026-07-10 | EGLL | 221,280 | 221,365 | 145,242 | 145,288 |
| 2026-07-10 | LFPG | 31,612 | 31,615 | 8,270 | 8,265 |
| 2026-07-10 | LIRF | 4,284 | 4,254 | 0 | 0 |
| 2026-07-10 | LTFM | 7,854 | 7,859 | 0 | 0 |

The other five airports (EDDM, EHAM, LEBL, LEMD, LSZH) are within the same margins (full counts in the staging receipts).
