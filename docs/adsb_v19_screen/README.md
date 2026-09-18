# ADS-B v19 screen: new label-free surface features for dark and unmatched rows (declared 18 September 2026)

Declared before any feature or model output exists. This is a **screen**, not a candidate:
nothing here can be exported or uploaded. Its only product is a go / no-go for a declared
VM round (`adsb_v19_v1`) that would refit the combination CatBoosts and the LightGBM group
with the new features under the usual rule (baseline = v16's development predictions,
0.15% mean bar, not worse on any month).

## Why these rows

On the incumbent's development months (Jan/Jul/Nov 2025) the squared error sits at LFPG
unmatched 36%, LIRF 29%, LTFM 10%, **dark segments 11%, unmatched rows at covered airports
10%**, gate-covered segments 2.7% (`docs/export_audit/`, session diagnostics of 18 Sep).
The first three have no data lever (adsb.lol, OPDI and OpenSky all lack the aprons at
LFPG/LTFM; LIRF is a label mechanism). The last 21% is the only place a feature can act.

## Features (all label-free; from the departing aircraft's own trace, other aircraft's
traces, the supplied stand and runway names)

A. Segment timing, matched rows (`src/prc2026_adsb/features_v19.py::segment_timing`):
- `adsb_appear_s`: take-off minus the first point of the last run of ground points before
  the motion start that has no silence of 120 s or more (a transponder-on / continuous
  coverage proxy; for many gate rows the aircraft is visible at the stand long before it
  moves — at EGLL the airport block time precedes the motion start by a median 6 min).
- `adsb_last_stand_s`: take-off minus the last point before the motion start (gate rows).
- `adsb_pre_motion_gap_s`: the coverage hole between that point and the motion start.
- `adsb_first_motion_s`: take-off minus the first time the aircraft was more than 40 m
  from its anchor (differs from `adsb_motion_s` only when it returned to the stand).

B. Stand imputation (`impute_stands`): for a departure whose (airport, stand) is not in
the learned catalogue (`matches_v11/stands.parquet`, 1,111 stands), the position of the
closest stand code by name at the same airport (same alphabetic prefix, nearest number
within 3), else the median position of the stands sharing the prefix. Gives
`adsb_first_dist_stand_imp_m` and `adsb_stand_imputed` (0/1). LEMD and LSZH lack an
anchor on about a third of their matched rows.

C. Surface congestion, every departure at a covered airport, matched or not
(`surface_congestion`): distinct other aircraft on the ground in the aerodrome box in the
window before take-off — `sfc_ground_10m`, `sfc_moving_10m` (gs ≥ 5 kt), `sfc_moving_20m` —
and near the departure runway's take-off roll start (median per airport-runway-month of
matched departures' first point at 30 kt or more in their last three minutes on the
ground, radius 1,200 m): `sfc_rwy_queue_5m`,
`sfc_rwy_queue_10m`. The departure's own aircraft is excluded where matched.

## Screen

LightGBM on the incumbent's residual (label − v16 development prediction), rows outside
LIRF, leave-one-month-out over Jan/Jul/Nov 2025 (fit on two months, predict the third;
fixed 400 rounds, learning rate 0.03, 63 leaves, min_data_in_leaf 200, feature_fraction
0.8, λ₂ 10, bagging 0.9, seed 20260918; fitting rows restricted to labels ≤ 43,200 s so the
garbage one-day rows do not dominate the L2 fit, every row evaluated; amendment written
before any screen output existed). Two feature sets: **base** = the 27 recorded ADS-B/neighbour
features plus airport, hour, weekday, schedule interval, NM proxy, clock flag; **new** =
base plus A–C. Reported per month: RMSE of the incumbent, incumbent + base screen,
incumbent + new screen; by row kind (gate / dark / unmatched) and by airport.

**Go** for the VM round only if incumbent + new beats incumbent + base by at least 0.3% on
the whole-month RMSE of every one of the three months, with the gain located on dark or
unmatched rows. Otherwise the features are recorded as null and no round is declared.
The base screen also reports how much residual structure the incumbent leaves for a
second-stage model at all; a large base gain would itself be a finding to investigate
before any feature work.

## Result (18 September 2026, [`screen_results.json`](screen_results.json)): no-go

Features built for the three development months by `tools/build_adsb_v19_features.py`
(`data/external/adsblol/features_v19/`, git-ignored; coverage on the 468,895 non-LIRF
rows: timing 58%, stand distance 55%, congestion 81%, runway queue 76%; 26–36 roll-start
points per month; 495–512 stand codes imputed per month). Screen by
`tools/screen_adsb_v19.py`:

| month | incumbent | incumbent + base second stage | incumbent + new features | new vs base |
| --- | ---: | ---: | ---: | ---: |
| 2025-01 | 323.511 | 324.020 | 324.014 | −0.002% |
| 2025-07 | 208.377 | 210.766 | 210.453 | −0.148% |
| 2025-11 | 156.519 | 157.951 | 157.721 | −0.146% |

By row kind (base → new, all months): gate 99.9 → 99.5, dark 159.0 → 158.6, unmatched
327.4 → 327.4. Gain share of the new features in the all-months fit: every one below 2.2%
(`adsb_first_dist_stand_imp_m` 2.1%, `sfc_moving_20m` 1.8%, `adsb_first_motion_s` 1.8%,
`sfc_rwy_queue_10m` 1.5%, the rest ≤ 1.5%).

Readings. (1) A second-stage model on the incumbent's residual is worse than the
incumbent on every month (+0.16% / +1.15% / +0.91%): the committee leaves no structure
that these inputs can recover out of month, which is what a well-fitted ensemble should
look like. (2) The eleven new features move that second stage by 0.15% on July and
November and by nothing on January — the coverage-thin month that most resembles the
ranking's EDDF regime — with the gain on gate and dark rows only; the congestion counts
add nothing on unmatched rows beyond the timeline features. The 0.3%-on-every-month bar
is not met; **no `adsb_v19_v1` round is declared, nothing is fitted on the VMs.** The
feature code stays in the repository as a record; the lever "surface features for dark
and unmatched rows" is closed on this evidence. v16 (241.75) remains the incumbent.
