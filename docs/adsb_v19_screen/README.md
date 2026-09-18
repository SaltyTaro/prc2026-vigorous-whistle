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
and near the departure runway's lift-off point (median of matched departures' last
ground points per airport-runway-month, radius 1,500 m): `sfc_rwy_queue_5m`,
`sfc_rwy_queue_10m`. The departure's own aircraft is excluded where matched.

## Screen

LightGBM on the incumbent's residual (label − v16 development prediction), rows outside
LIRF, leave-one-month-out over Jan/Jul/Nov 2025 (fit on two months, predict the third;
fixed 400 rounds, learning rate 0.03, 63 leaves, min_data_in_leaf 200, feature_fraction
0.8, λ₂ 10, seed 20260918). Two feature sets: **base** = the 27 recorded ADS-B/neighbour
features plus airport, hour, weekday, schedule interval, NM proxy, clock flag; **new** =
base plus A–C. Reported per month: RMSE of the incumbent, incumbent + base screen,
incumbent + new screen; by row kind (gate / dark / unmatched) and by airport.

**Go** for the VM round only if incumbent + new beats incumbent + base by at least 0.3% on
the whole-month RMSE of every one of the three months, with the gain located on dark or
unmatched rows. Otherwise the features are recorded as null and no round is declared.
The base screen also reports how much residual structure the incumbent leaves for a
second-stage model at all; a large base gain would itself be a finding to investigate
before any feature work.
