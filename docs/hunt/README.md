# Fork M: search for a find the size of the rematch (7 October 2026) — nothing reaches 1e7 s², stopped early

Question (coordinator): is there a class of ranking rows where information we already hold (adsb.lol surface days, OPDI, organiser
fields, published aggregates) could pin the off-block as tightly as the no-NM rematch did (about 3e8 s² of ranking squared error,
the 2.23 s gap to rank 1), but no shipped layer uses it? Label-free on the ranking file; 2025 labels only through the v68 development
proxy (`runs/opdi_stack_lszh/dev_v68_proxy.parquet`, 938a91cd...). Nothing declared, built or uploaded.

Tools: `tools/hunt_census.py` (class census from `data/external/adsblol/matches/matches2_<month>.parquet`, the raw files, the v66 chain
step files under `runs/ctx_ownpoints_weight/rank/dry_v66/steps/` and the own-point caches `runs/stack_resid/v3_features/`),
`tools/hunt_pools.py` (pool table, `runs/hunt/pools.csv`). Light table joins only (seconds); no VM was needed or started.

## Classes

`gate` = departure segment traced from the stand; `pinned_layer` = a shipped hard layer moved the row on the ranking file (no-NM
rematch 341, intruder bound + identity 46, own-motion v1 52, own-motion v2 2,313; 2,686 rows); `dark` = segment starts already moving;
`unm_seen` = no segment, but the identified own aircraft has adsb.lol points at the airport between its inbound in-block + 120 s and
take-off; `unm_unseen` = no segment and no own point. Ranking projection = ranking rows x the 2025 mean squared error of the same
(airport, class) on the v68 proxy (LIRF population and guard rows excluded: pinned).

| pool | 2025 rows | 2025 share of squared error | ranking rows | projected ranking share | perfect-pin bound (projected s²) |
|---|---:|---:|---:|---:|---:|
| unm_unseen | 242,506 | 57.4% | 109,953 | 69.8% | 1.06e10 (LFPG 6.4e9, LTFM 1.9e9, LIRF 1.4e9) |
| dark | 257,792 | 19.8% | 140,197 | 22.4% | 3.40e9 (LIRF 8.4e8, EGLL 7.8e8, LEBL 4.5e8, LFPG 3.9e8, LEMD 3.1e8) |
| gate | 153,595 | 3.7% | 82,923 | 5.3% | (already traced) |
| unm_seen | 17,973 | 2.1% | 9,082 | 2.5% | 3.8e8 (LFPG 1.6e8, LIRF 8.6e7, EDDF 6.1e7) |
| LIRF population + guard | 562 | 17.0% | pinned | | |

Ranking class shifts (label-free, by airport-month): EDDF gate share 27% in 2025 -> 4-8% in 2026, unmatched 20-24% (63-70% on 29-31
January 2026); LEMD July 2026 unmatched 26% (day-to-day 10-55%); LIRF July 2026 unseen 99.8%; LFPG unmatched 79%; LTFM 100%. The own
aircraft is identified at the airport on about 90% of rows every day (57-73% on the first of each month: no previous-day inbound in the
ranking file). Every gap follows receiver coverage, not a key: where the aircraft is identified but unmatched it has no departure-taxi
points (tunnel 12, `docs/dig/hexmatch.md`, and below).

## Why no pool holds a pin

- **unm_unseen (70%)**: no adsb.lol or OPDI surface observation exists (LFPG, LTFM, LIRF July 2026); nothing to pin with.
- **dark (22%)**: the aircraft is seen only after leaving the stand; first point, motion start, distance to stand, pre-gap own points and
  the stand stop are already learner / stacker / member inputs (`src/prc2026_adsb/features.py`, tunnel 18, v63-v66). Hard bounds from
  them were tested on v61 and closed (tunnel 15, `docs/dig/own_bounds.md`): upper bounds fail on early records, lower bounds 58-90% valid.
- **unm_seen (2.5%, bound 3.8e8)**: on 2025 these rows carry a few mid-taxi points (median 2-113 per row; first moving point median
  380-690 s before take-off against labels of 750-1,320 s); "parked" points are mostly taxi holds, not the stand (LFPG's last parked
  point median 18,143 s before take-off = the previous evening). Of 2,345 rows with both a moving and a parked point, the label lies
  above the first moving point on 78% and below the last parked point on only 27%: no bracket. The same points are already v3 / member
  inputs (q_mv_vs_pred, q_park_vs_pred). Plausible gain far below 1e7 s².
- **pinned layers / gate**: already pinned.

Closed angles not repeated: hex rematch (12), replica union (16), position intruders (11), hard own bounds (15), OPDI everywhere,
destination witness and OPDI identity, case files 4/5 (early records, minute-grid fills at schedule).

## Verdict

No class reaches 1e7 s² of plausible ranking gain. The rematch-sized pots are the unseen rows (no observation in any open source we
hold) and the early-record recording mode (not observable). Stopped early, as the directive allows.
