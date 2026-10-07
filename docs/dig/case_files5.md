# Case files round four: 2026-like blind spots on the v68 development proxy (7 October 2026) — closed, no GO

Lever #10 of the rank-1 hunt (fork L). Diagnostic, 2025 development months only; labels are read only to pick and classify rows and to
score screens. No official score, no leaderboard, no upload, nothing under `runs/submissions`, nothing declared. Scratch (git-ignored): `runs/dig/case5/` (`census.py`, `bounds.py`,
`build.py`, `context.py`, `turn.py`, `dump.txt`). Hand-off log: [`case5_HANDOFF.md`](case5_HANDOFF.md).

## Question and control

The ranking months lose the surface trace where 2025 had it: LIRF July 2026 has no adsb.lol surface feed, EDDF's gate-traced share falls
from 0.14-0.41 to 0.04-0.08. Is there a label-free mechanism that works **without** a surface trace? Control: the v68 development proxy
`runs/opdi_stack_lszh/dev_v68_proxy.parquet` (938a91cd...; 320.0400 / 250.8650 / 181.9933 / 163.5496 s). The 2026 condition is read on the
2025 rows that already lack a trace (state dark = matched but not gate-traced, unm = unmatched): exact v68 values, no re-prediction.
(A v68-chain re-prediction of the thinned sets would need every later member re-run on thinned inputs; the natural blind rows carry the
same condition with exact values.)

## Census of the v68 proxy (ordinary rows, label <= 3 h, outside the LIRF population; 54.9% of all squared error)

| cell | rows | RMSE (s) | share of ordinary SSE | OPDI matched / stand exit |
|---|---:|---:|---:|---|
| LTFM unm | 91,338 | 202.5 | 18.1% | 0.1% / 0 |
| LFPG unm | 64,894 | 231.7 | 16.9% | 23.8% / 0.1% |
| LIRF dark | 26,700 | 340.5 | 15.0% | 20.2% / 0 |
| EGLL unm | 32,852 | 219.2 | 7.6% | 56.0% / 0 |
| LIRF unm | 18,463 | 286.3 | 7.3% | 14.9% / 0 |
| EGLL dark | 35,605 | 177.8 | 5.5% | 69.5% / 0 |
| LEBL dark | 37,016 | 155.7 | 4.3% | 65.2% / 0.9% |
| EDDF dark | 47,904 | 100.0 | 2.3% | 89.6% / 6.4% |

## The rows

200 rows: the largest v68 squared error among blind ordinary rows after removing labels > 3 h, LIRF fills (block within 6 s of the
schedule), no-NM-clock rows, extreme published days and the 830 rows read in tunnels 17 / 19 / 20 / 22: 100 LIRF (0.62% of
ordinary SSE), 50 EDDF (0.31%), 50 LFPG (0.49%). Every row was read with supplied data only (the 2026 condition): schedule, NM clocks
(AOBT_3, LOBT, EOBT_1), stand, runway, operator, the last supplied arrival at the stand and its type, arrivals and departures at the stand
over 3-4 h, departures within 30 min, the published day's implied shift, OPDI first-seen / runway-entry times. The remaining error is
diffuse: after the earlier rounds the worst blind LIRF rows miss by about 20 minutes either way.

| group | class (by the hidden block against the NM clock) | rows | share of the group's SSE | under-predicted | block within 5 min of LOBT / schedule |
|---|---|---:|---:|---:|---|
| LIRF | early record | 54 | 54% | 69% | 28% / 17% |
| LIRF | agree (block near AOBT_3; the model hedges toward early) | 39 | 39% | 0% | 23% / 0% |
| LIRF | late record | 7 | 7% | 0% | 29% / 0% |
| EDDF | early record | 38 | 75% | 87% | 40% / 24% |
| EDDF | late record | 10 | 20% | 0% | 0% / 0% |
| LFPG | early record | 49 | 98% | 96% | 31% / 12% |

The early records often sit on a plan time (LOBT or the schedule) while AOBT_3 is 10-25 min before take-off: the recording mode of the
off-block, the same unobservable class as tunnels 4 / 14 / 22 (`docs/clean_line_screens/lirf_mechanism.md`). Nothing in the supplied
fields separates the classes row by row; the chain predicts the mixture mean, so blind LIRF rows miss by about +-20 min on both sides.

## Mechanisms tested and rejected

- **OPDI event times as hard lower bounds** (off-block precedes the stand exit / apron exit / first taxiway entry / first seen, so the label
  is at least take-off minus the event): where v68 sits below such a bound the bound is usually wrong (validity among violations 24-89%);
  v68 -> max(v68, bound) worsens squared error at every airport and event type except LFPG first-taxiway (246 rows, -0.002%). Same result as
  tunnel 15 for adsb.lol: where the bound holds v68 already respects it.
- **OPDI peers as an hour-state signal at the blind airports:** OPDI sees almost no surface at LFPG / LIRF / EGLL / LTFM (median first-seen
  10-60 s after take-off; stand exit 0%, first taxiway 0-1%), so it cannot supply the missing neighbour activity there.
- **Supplied-data turnaround** (take-off minus the last supplied arrival in-block at the same stand, same type; usable when no ADS-B
  identity exists): the v68 residual by turnaround band is within a few seconds and changes sign between months at LIRF, LFPG, EGLL, LTFM,
  EDDF, LEBL, LEMD (largest consistent cell: LFPG turnarounds under 30 min, +6 to +28 s on 1,723 rows; worth well under 0.01%).

## Verdict

**No GO, closed.** The 2026 blind condition removes evidence the chain cannot replace from any open or supplied source checked: the
remaining blind-row error is the off-block recording mode (early / late against the NM clock, often on a plan time) and its hedging, which
no label-free input identifies. No configuration was declared and no dry run exists. Compute: light reads on the workstation only; Azure
arcw6-arcw7 were not started (no step needed them).
