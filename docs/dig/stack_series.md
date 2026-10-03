# Dig tunnel 13: published daily series and identity features on top of stack_resid_v2 — no GO, closed (3 October 2026)

2025 development months only; nothing declared, built or uploaded; no official score or leaderboard read. Base: stack_resid_v2's
out-of-month development predictions as reproduced by tunnel 8 (`runs/dig/stack3/base_v2.parquet`, `X46.parquet`; −0.0445 / −0.1249 /
−0.1801 / −0.3287%). Scratch: `runs/dig/stack_series/` (`build.py` features, `screen.py`, `screen_s2_ser_s2_id_s2_all_rep_all.json`).

## Features (label-free, present on the ranking rows)

- **EUROCONTROL daily series** (airport-day; the workbooks have no airline-day sheet): ATC pre-departure delay per IFR departure
  (airport-operator data) and per reporting flight (airline data), airline-report coverage, regulated share, early / late share of
  regulated departures (slot adherence), NM-versus-airport count gap, ATC share of the all-causes total. Coverage 48–100% of dev rows
  (the airport-operator ATC figure is missing at several airports), same on the ranking.
- **Identity** (EDDF / EGLL / LFPG, tunnel 10's ICAO table): ground time since our own aircraft's inbound in-block, the predicted
  turnaround (that minus the prediction) and its difference to the scheduled turnaround, inbound arrival delay and taxi-in, same stand as
  the inbound, gap to the same aircraft's previous departure. 28.6% of dev rows (29.4% of ranking rows). Sanity: no identified row has a
  ground time below its label.
- New features carry their own support guard (0.1–99.9% per airport of the fitting months).

## Screens (change against stack_resid_v2, complete-month RMSE; paired-day 95% in %)

| Screen | January | July | November | December | Equal-footing mean |
|---|---|---|---|---|---|
| second learner: 46 + day series | +0.0092 [−0.005, +0.039] | −0.0005 | −0.0094 | −0.0310 [−0.106, +0.044] | −0.0074% |
| second learner: 46 + identity | +0.0121 | **+0.0447** [+0.022, +0.069] | +0.0134 | −0.0569 [−0.126, +0.016] | −0.0000% |
| second learner: 46 + series + identity | +0.0082 | +0.0007 | −0.0084 | −0.0389 [−0.119, +0.045] | −0.0100% |
| v2's learner refitted with 46 + series + identity | +0.0072 | **+0.1098** [+0.081, +0.143] | +0.0177 | +0.0474 | +0.0548% |

Bar: incremental equal-footing mean ≤ −0.05%, no month worse, two intervals below zero. Nothing comes near it; January is worse in every
variant, and identity features cost July. **Closed.** stack_resid_v2 already holds what the day level says; the airport-day series add
nothing at row level (as at day level in `docs/clean_line_screens/README.md` section 6), and the aircraft's own turnaround is either
carried by the members' schedule and arrival-context inputs or not predictive of the residual.
