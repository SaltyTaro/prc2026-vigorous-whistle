# Dig tunnel 15: own-aircraft ADS-B hard bounds — no GO, closed (3 October 2026)

Question: like the intruder stand bound (v58), do the departure's own surface points give a hard bound that the v61 chain breaks?
2025 labels only; the ranking is read only for sizing. Nothing declared, built or uploaded. Scratch `runs/dig/own_bounds/`
(`validity.py`, `violations.py`; tables `val12.parquet` = the twelve 2025 `matches2_*` segments with labels, `dev_bounds.parquet` = the
v61 development proxy `runs/dig/census61/dev_v61.parquet` with its segment). LIRF population and guard rows excluded throughout.

## What the shipped matcher already gives

`src/prc2026_adsb/features.py::segment` returns, for every matched departure, the first point of its departure segment (`segment_start`)
and the motion start (`motion_start`: the point after which the aircraft leaves its stand for good; equal to the first point for "dark"
segments, which start already moving). Both enter the models (`adsb_first_point_s`, `adsb_motion_s`, `adsb_first_gs_kt`,
`adsb_first_dist_stand_m`). The question is only whether a hard bound on them is valid and broken.

## Validity on 2025 (label against take-off − point time, tolerance 30 s)

| bound | rows | valid at | fails at |
|---|---:|---|---|
| lower, gate rows: block <= motion start | 426,819 | EDDF 99.6, EDDM 99.5, LSZH 99.2, LIRF 99.0, EGLL 98.7, LEMD 98.8, LEBL 97.8, EHAM 95.9% | LFPG 89% (408 rows) |
| lower, dark rows: block <= first point | 762,028 | EDDF 96.5, LEBL 97.0, LEMD 96.7, LSZH 94.0% | EHAM 89.9, EDDM 90.8, EGLL 84.0, LIRF 84.2, LFPG 58.2%; several airport-months below 90% everywhere but LEBL / LSZH |
| upper, gate rows: block >= first (parked) point | 426,819 | none | EHAM 87, LSZH 76, LEBL 75, EDDF 71, EDDM 68, LEMD 36, LIRF 22, EGLL 15% |
| upper, gate rows: block >= last parked point (motion start) | 426,819 | none | EHAM 64%, every other airport below 33% (the early record class) |

Upper bounds fail as expected: the airports record off-block before the aircraft is seen to leave (the early class of tunnel 14).

## Violations by the v61 development proxy (January / July / November / December 2025)

- **Dark rows:** v61 sits below take-off − first point on 27,158 rows, but only 14% of them satisfy the bound: the large violations
  (median 4,400 s) are segments whose first point is far older than the off-block (tow or reposition with the transponder on, arrival
  points). Moving them to the bound adds 7.7e11 squared error. Restricting to plausible bounds (first speed >= 5 kt, bound <= 2,400 s)
  leaves 1,423 rows with median violation 24 s and validity 64% (labels straddle the bound); enforcing it still worsens (+1.4e8).
- **Gate rows:** v61 below the motion-start taxi on 17,750 rows (16,944 at EHAM, where label = motion taxi + 7 s at the median), mean
  shortfall 61 s, validity 91%. Raising to the motion taxi plus a month-blocked offset (quantile 0.5 / 0.25 / 0.1 of label − bound on the
  other eleven months, violation cap 300 / 600 / 1,200 s) is worse or null in every setting: best case January +0.005%, July −0.015%,
  November +0.046%, December −0.084% of month squared error (positive = better), i.e. below the bar and mixed in sign.
- **Upper bound (seen parked):** v61 above take-off − first point + 30 s on 50,484 gate rows; only 20% satisfy it; capping worsens every
  month by 3-19% of squared error.

## Ranking size (v61, read only)

Gate rows 82,927, of which 9,560 sit below their motion taxi (median shortfall 9.6 s); dark rows 142,351, of which 755 sit below a
plausible first-point bound (median 25 s, squared change 1.0e8). The same picture as 2025: small shortfalls inside the measurement noise
of the bound, large ones where the bound is invalid.

**Verdict: closed.** Wherever the own-aircraft bounds are valid the chain already respects them (it learnt the same points as features);
where it breaks them the bound is wrong. Unlike the intruder bound, there is no hard information here the models have not seen.
