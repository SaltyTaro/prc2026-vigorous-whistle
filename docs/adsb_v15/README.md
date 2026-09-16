# ADS-B v15: learner strength and diversity on top of v14 (16 September 2026)

After v14 (official **243.5805**, 1st of 148 at 00:51 UTC; second 245.0207) the user
asked for a much larger margin. This record holds the diagnostics run on the recorded
v14 development predictions before anything new was declared, the declaration
`configs/adsb_v15_v1.json` (01:41 UTC, amended 01:47 UTC before any scoring), the
learner fits, the selection and the export.

## Diagnostics before the declaration (all on the three scoring months, no new upload)

Where the v14 development squared error sits (Nov 2025, the fold with the highest
ADS-B match rate): LIRF 37.7%, unmatched rows outside LIRF 34.8%, matched rows
outside LIRF 27.5%. In the ranking, LIRF is 80% matched in January 2026 and 0% in
July; LTFM has no surface data at all; LFPG is 21% matched.

1. **LIRF cannot be fixed with ADS-B.** Even on gate-covered LIRF segments (the
   aircraft seen stationary at the stand and then leaving) the airport block time is a
   median 373–495 s *before* the ADS-B motion start with an interquartile spread of
   about 400 s; at EDDF the same quantity is within 60 s for a third of gate rows. The
   LIRF label is therefore not the observable pushback. In addition 13–17% of LIRF rows
   *with* an NM record or aircraft type still carry `BLOCK_TIME = SCHED_TIME` fills. A
   LIRF-only residual (matched non-population rows, v12 features, leave-one-month-out
   Jul↔Nov; January has no LIRF coverage) was worse than v14 on both months: +0.17% /
   +2.01% at weight 1.0, +0.04% / +0.47% at 0.5. A within-month day-grouped fit gains
   only 440.6 → 426.0 (Jul) and 292.7 → 291.6 (Nov) on the in-scope rows, and a fit
   that excludes the fills sharpens the normal rows (378.8 → 351.9) but loses the same
   amount on the fill rows (726 → 802). LIRF stays excluded, as in v11–v14.
2. **First-point geometry** (first-point and stand latitude/longitude, bearing, 250 m
   grid cell) in the v12 residual: null (+0.05% / −0.04% / −0.14% versus v12 on Jan /
   Jul / Nov); the grid cell hurts.
3. **Blend structure**: per-airport or per-match-status weights and a three-way
   least-squares stack (v12, LightGBM, CatBoost), fitted leave-one-month-out on the
   recorded predictions, are null or worse than the fixed w = 0.6 mean of the two
   learners (best mean ratio +0.02%). The v14 blend is at its optimum.
4. **Unmatched rows are a coverage problem, not a callsign problem.** On sample days
   the unmatched departures at EDDF/LFPG/LEMD/EGLL are almost all "airborne only"
   (the callsign is in the archive but no ground position was received); a
   time-based fallback finds nothing. LTFM has no surface positions on any day.
5. **Incomplete matches.** Between 20% and 66% of the "matched" rows at EDDF (ranking
   months), LEMD (July), EGLL and LFPG have their last ground point more than 300 s
   before takeoff: the track is the *arrival's* landing rollout or a departure segment
   cut short by coverage. The learners saw thousands of such rows in 2025 (LEMD July
   2025: 87% of its matches) and carry `adsb_last_ground_s`, so they have learned the
   pattern. Restricting the residual scope to complete matches is clearly worse
   (incomplete rows 196.6 → 215.5 s on Jan, 232.0 → 243.5 on Jul, 196.8 → 205.9 on Nov;
   whole months +0.20% / +0.24% / +0.28%): the arrival rollout and the partial
   segment carry turnaround information that the models already use. No change.
6. **LIRF fill mixture with ADS-B** (a second LIRF attempt, cross-month): a fill
   classifier on the non-population matched rows (operator, type, hour, schedule
   interval, ADS-B-observed delay, NM clocks, stand, runway; AUC 0.79 Jul→Nov, 0.82
   Nov→Jul) with the exact fill value `MVT − SCHED` and a clean normal component is
   much worse than v9's implicit hedge (+2.75% / +1.98% on the month; +0.60% / +0.37%
   even at half weight). Closed.
7. **Seed variance of the LightGBM learner is small** (forward fold 389.054 vs 389.047,
   matched rows 205.29 vs 204.89), so seed bagging alone cannot buy much; the longer
   CatBoost (the v14 CatBoost hit its 6,000-round cap on both folds while still
   improving) and diversity are where the remaining learner gain is.

## Declaration `configs/adsb_v15_v1.json`

Nine learner variants, every one fitted without LIRF on the timeline bundle's 276
model features plus the 27 ADS-B features (the exact tables the v14 learners used),
seasonal and forward folds out of sample, refit on the eleven 2025 months at the
median best iteration; December never read:

| variant | kind | seed | change versus the v14 learner | worker |
| --- | --- | --- | --- | --- |
| lgb_s16 / lgb_s17 / lgb_s18 | LightGBM | 20260916–18 | seed only | arcw4 |
| lgb_deep | LightGBM | 20260915 | lr 0.02, 255 leaves, min_data 50, feature_fraction 0.7, λ₂ 30, cap 8,000 | arcw4 |
| lgb_clean | LightGBM | 20260915 | fitting/early-stopping rows with label > 43,200 s left out | arcw4 |
| cat_long | CatBoost | 20260915 | 20,000-round cap | arcw1 |
| cat_s16 | CatBoost | 20260916 | seed, 20,000-round cap | arcw2 |
| lgb_adsbres / cat_adsbres | LightGBM / CatBoost | 20260915 | target = label − ADS-B taxi on matched rows (amendment) | arcw3 |

Ensembles are plain per-row means (`tools/ensemble_adsb_full.py`): `ens_v15_lgb` (six
LightGBMs), `ens_v15_cat` (three CatBoosts), `ens_v15_bag` = mean of those two group
means, `ens_v15_all` (nine learners flat), `ens_v15_seeds` (four same-parameter
LightGBM seeds and the three CatBoosts, two-level), `ens_v15_res` (the two
ADS-B-residual learners) and `ens_v15_mix` (bag and res). Fifteen candidates: each
ensemble at w = 0.6 / 0.7 / 0.8 on every row outside LIRF, on top of v9 + the v12
residual; the residual model is exactly v12's. Baseline: the **v14 development
predictions** (331.31 / 273.17 / 194.21 on Jan / Jul / Nov 2025). Rule: not worse than
v14 on any month and mean ratio ≤ 0.997; lowest mean ratio wins; otherwise nothing.

## Learner fits (01:41–04:30 UTC; every worker deallocated afterwards)

Fold scores below are on the rows the learners predict; "matched" / "unmatched" are
the non-LIRF rows with and without a surface segment (the whole-set RMSE includes
the untouched LIRF rows and is not informative). Reference: the v14 learners.

| variant | seasonal best iter | seasonal matched / unmatched | forward best iter | forward matched / unmatched | refit rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| v14 LightGBM (`adsb_full_nolirf`) | 1,419 | 415.92 / 700.70 | 1,194 | 204.89 / 645.17 | 1,306 |
| v14 CatBoost (`adsb_full_cat`) | 6,000 (cap) | 417.25 / 697.96 | 5,996 (cap) | 205.12 / 647.84 | 5,998 |
| lgb_s16 / s17 / s18 | 1,710 / 1,738 / 1,600 | 415.84 / 415.84 / 416.00 | 1,217 / 1,207 / 1,019 | 205.29 / 204.83 / 205.32 | 1,464 / 1,472 / 1,310 |
| lgb_deep | 2,728 | 415.80 / 699.95 | 2,018 | 203.7 / 644.6 | 2,373 |
| lgb_clean | 1,665 | 415.03 / 700.04 | 1,546 | 204.31 / 644.20 | 1,606 |
| cat_long | 9,742 | 416.68 / 697.42 | 12,694 | 203.74 / 647.17 | 11,218 |
| cat_s16 | 13,870 | 416.34 / 696.64 | 15,284 | 202.23 / 645.88 | 14,577 |
| lgb_adsbres | 1,696 | 424.63 / 701.29 | 1,464 | 216.10 / 645.91 | 1,580 |
| cat_adsbres | 11,997 | 419.17 / 698.42 | 12,000 (cap) | 209.73 / 647.58 | 11,998 |

Single-variant gains are 0.1–0.7% on matched rows; the ADS-B-residual target is worse
(the offset is garbage on the incomplete matches, which the label target simply ignores).
Learner means alone on the non-LIRF rows of the two folds (seasonal / forward): v14
pair 269.91 / 160.94; `ens_v15_lgb` 272.65 / 162.91; `ens_v15_cat` 270.14 / 162.10;
**`ens_v15_bag` 268.81 / 159.18**; `ens_v15_seeds` 269.07 / 159.47; `ens_v15_all`
269.52 / 159.69; `ens_v15_mix` 269.29 / 160.09; `ens_v15_res` 272.50 / 165.66. The
nested means were built with the unchanged ensemble tool by listing members with
multiplicities; each equals its two-level mean to 4e-12 s.

## Development results and selection ([`development_results.json`](development_results.json))

`tools/run_adsb_v11.py --config configs/adsb_v15_v1.json`: the residual variant
reproduces v12 exactly and the baseline reproduces v14 (331.309 / 273.169 / 194.212).
Every label-target ensemble improves on v14 at w = 0.6; the ADS-B-residual ensemble
loses at every weight. Best candidates versus v14 (Jan / Jul / Nov; paired-day 95%
intervals in seconds):

| candidate | Jan | Jul | Nov | mean | intervals |
| --- | ---: | ---: | ---: | ---: | --- |
| ens_v15_bag_60 | −0.121% | −0.177% | −0.346% | **−0.214%** | [−0.68, −0.27] / [−0.57, −0.40] / [−0.80, −0.55]; 91 of 92 days lower |
| ens_v15_bag_70 | −0.070% | −0.119% | −0.395% | −0.195% | [−0.45, −0.12] / [−0.45, −0.19] / [−0.96, −0.62] |
| ens_v15_seeds_60 | −0.096% | −0.132% | −0.295% | −0.174% | all below zero |
| ens_v15_mix_60 | −0.136% | −0.097% | −0.263% | −0.165% | all below zero |
| ens_v15_res_60 | +0.068% | +0.298% | +0.480% | +0.282% | — |

`ens_v15_bag_60` improves every airport outside LIRF on every month (EDDM −0.9 / −0.4 /
−1.1%, EHAM −0.7 / −0.5 / −0.8%, LSZH −0.4 / −1.0 / −0.7%, LTFM −0.3 / −0.3 / −0.5%),
but its mean ratio 0.99786 is above the declared bar of 0.997. **The declared rule
selects nothing**; no export was made under this declaration. The effect is real
(intervals strictly negative, 91 of 92 days) and small: roughly 0.5 s on the ranking
if it transfers like v12_v5 did.
