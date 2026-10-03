# Dig campaign (3 October 2026)

The user, 3 October 2026: "now we need to play this like minecraft, where we dig and dig for hours and hope to find diamond, in this
case reduce our RMSE, because at this rate we will be out of top 5 by the end of comp close". Official best v57 = 221.3289 (2 Oct).

Every tunnel validates on 2025 only (month-blocked; January / July / December equal footing, November supplementary; complete months
including extreme labels; paired-day bootstrap intervals). No official score, no leaderboard, no upload; LIRF population rows and the
two guard rows stay untouched. Control for post-processing ideas: the shipped chain on the shipped model
part, `runs/scout/refit/ctl_v55m.parquet` column `new` (pooled dev RMSE 239.33 s).

GO bar for a screen (before any declared round): equal-footing mean <= -0.05%, no equal-footing month worse than +0.01%, at least
two intervals below zero; or, for a label-free ranking anomaly, a sized potential of at least 0.1 s.

## Census of the shipped chain (development proxy, share of all squared error)

| cell | rows | RMSE (s) | share |
|---|---:|---:|---:|
| LFPG unmatched, incl. two unidentifiable January 2025 tails | 64,897 | | 36.0% |
| LIRF population (guarded) | 562 | | 16.6% |
| LTFM (all unmatched) | 91,338 | 205.8 | 10.0% |
| LIRF matched | 30,899 | 334.4 | 9.0% |
| EGLL unmatched / matched | 33,085 / 46,548 | 230.9 / 177.6 | 4.6% / 4.0% |
| LIRF unmatched | 18,463 | 286.8 | 3.9% |
| LEBL matched | 53,542 | 149.1 | 3.1% |

LIRF clock-present rows hold 12.9% (fill rows with an NM delay above 600 s about 4.4%; the class is unobservable, closed by
`docs/clean_line_screens/lirf_mechanism.md`). Rows with |error| > 600 s are 1.1% of ordinary non-population rows and hold 46% of their
squared error; 69% of them are under-predictions.

## Tunnels

| # | tunnel | owner | state | record |
|---|---|---|---|---|
| 1 | 2026 ranking-file regime audit (label-free) | fork | **find**: no-NM rows at EDDF/EHAM/LSZH matched to ADS-B by flight number, not callsign; declared round `configs/rematch_nonm_v1.json` running | `regime_audit.md` |
| 2 | residual ore test: everything label-free on the shipped chain's residual | fork | **find**: components + published day discrepancy, equal-footing -0.258% (3-seed bag, w 0.5); being prepared for a declared round (`stack_resid_v1`) | `residual_ore.md` |
| 5 | cross-row hard constraints (turnaround minima, stand families, ordering) | fork | closed (only MARS stand families valid, -0.02%) | `constraints.md` |
| 6 | published-gap regime hunt on v59 | fork | closed (nothing >= 0.1 s; small extreme-day count / day-boundary idea) | `published_gaps.md` |
| 7 | micro-layer bundle (MARS families, count-aware extreme days, day-boundary pairs) as one declared round on v60 | fork | not accepted (the day-boundary piece could move a row past a midnight its take-off precedes) | `../micro_bundle/` |
| 8 | stronger stacking learner (seed spreads, finer splits, airport interactions) on top of stack_resid_v2 | fork | closed (best -0.020%, December only) | `stack3.md` |
| 9 | plumbing audit of the shipped ranking path (key/unit/join/fallback bugs) | fork | closed (no bug worth 0.1 s; remaining gaps are archive coverage) | `plumbing.md` |
| 11 | position-based intruders at EDDM/EHAM/LSZH/LEMD/LEBL (ADS-B parking fixes) | fork | closed (validity <= 14%; fails at the EDDF/EGLL control too: towing and coverage edges) | `position.md` |
| 7b | MARS stand families alone (`configs/micro_bundle_v2.json`) | fork | not accepted (-0.0173% vs -0.02% bar) | `../micro_bundle/` |
| 14 | v61 error census + five new hypotheses | fork | census done (v61 dev proxy pooled 237.59 vs v46 240.48); exact-identity bound and aircraft-level history closed; the rest expected below the bar | `census61.md` |
| 13 | EUROCONTROL day series + aircraft-identity features on top of stack_resid_v2 | fork | closed (best -0.010%, January worse in every variant) | `stack_series.md` |
| 12 | hex-based ground rematch (stale/blank ground callsigns) | fork | closed (the shipped matcher already uses every ground point of the identified aircraft; unmatched rows lack departure coverage) | `hexmatch.md` |
| 10 | same-type intruders by ICAO identity (adsb.lol low airborne points) | fork | **accepted** (`configs/intruder_identity_v1.json`; dev -0.0945%) but only 5 ranking rows (S 4.06e6); rides with the next file | `identity.md` |
| 3 | LTFM deep dive (runway configuration, queues, clock structure, banks) | fork | **closed, no GO** (shared temporal state: neighbouring take-offs' residuals correlate 0.25-0.27, but every label-free signal reads it at <= 0.076; new groups help the older twelve-month predictor -0.13% at w 0.5 but not the shipped chain, whose context networks hold them; stand x runway bias -0.010 / -0.013%, all months better, a tenth of the bar) | `ltfm.md` |
| 4 | LFPG / EGLL unmatched deep dive | fork | **closed, no GO** (configuration, queues, stand x runway: +0.001 to +0.225% equal-footing; early class AUC +0.002 at most; no partial traces to recover) | `lfpg_egll.md` |
| 15 | own-aircraft ADS-B hard bounds (first seen moving -> label lower bound; last seen parked -> upper bound) on partially traced rows | fork | **closed, no GO** (the matcher already feeds first point and motion start to the models; where a bound holds v61 respects it, where v61 breaks one the bound is wrong: dark-row first point valid 58-90%, upper bounds fail everywhere (early class); gate-row motion floor at EHAM mixed, best +0.005 / -0.015 / +0.046 / -0.084%) | `own_bounds.md` |
| 16 | adsb.lol replica union (prod-0 + staging-0 of the same day) for surface coverage | fork | **modest GO**: 52 of 62 ranking days have a staging-0 copy; the union matches 2,021 more departures (0.68%; 412 gate-traced, 50 upgraded), mostly on prod-0 outage days 23 January and 11 July; > 1% new at LFPG Jan/Jul, EDDF Jul, EGLL Jan, LEMD Jul; sized at S ~ 9e6 (-0.05..-0.15 s); declared layer `configs/replica_union_v1.json` (0bb9eb5) **NOT accepted** on the v62 and v63 controls (Jul +0.329% [+0.088, +0.649], eq mean +0.117%, twelve 3/12 better): the union's new gate traces are mostly sparse 3-7 point tracks with meaningless implied taxis (gain about -1 x S); closed, nothing built | `replica_union.md`, `../replica_union/` |
| 17 | case files: the worst v61 development rows read by hand for a new label-free mechanism | fork | **find**: own-motion bound at EDDF / EGLL / LFPG (own aircraft seen moving at >= 3 kt after its inbound blocked in at the same stand, B <= 2,700 s, not gate-traced; valid 93-100%; dev w 1.0 -0.038 / -0.111 / -0.082 / -0.163%, twelve 12/12; ranking 155 rows, S 4.25e7); declared round `configs/own_motion_v1.json` (bb21fb4) **ACCEPTED** (w 1.0: -0.0380 / -0.1103 / -0.0814 / -0.1630%, eq mean -0.1037%, every interval < 0, twelve 12/12); v62 dry run on v61 `runs/own_motion/dry_v62/` cba143f8..., 155 rows up, S 4.08e7 | `case_files.md`, `../own_motion/` |
| 18 | pre-gap own-aircraft ADS-B points (dropped by the matcher) as soft features in the guarded stacker | fork | **GO** (identity extended to all ten airports; replace-learner w 0.5: -0.087 / -0.078 / -0.354 / -0.394%, eq mean -0.186%, every interval < 0); declared round `configs/stack_resid_v3.json` on v62's chain **ACCEPTED** (14a3bdf; -0.0717 / -0.0808 / -0.2501 / -0.3269% in-chain vs the v62 proxy, eq -0.1598%, every interval < 0; LIRF kept at v2) -> **v63 BUILT** at the user's word ("build and keep it ready"), `4cff0945...`, 233,886 rows vs v62, S 2.27e7, audit clean; not uploaded | `pregap.md`, `../stack_resid/` |
| 19 | case files round two (next ~200 worst rows outside known mechanisms, ADS-B airports) | fork | **find**: own-motion bound at EHAM / EDDM / LSZH / LEMD / LEBL (v1 trigger + no return to the stand, w 0.5, after v1; validity >= 95% every month; dev vs v62 proxy -0.037 / -0.048 / -0.088 / -0.081%, every interval < 0, twelve 12/12; ranking 2,523 rows, S 2.14e7, EHAM 2,002 of them); declared round `configs/own_motion_v2.json` (47c7406) **ACCEPTED** on the v62 proxy (-0.0370 / -0.0479 / -0.0877 / -0.0812%, eq mean -0.0554%, every interval < 0, twelve 12/12); sizing dry run on v62 be684544 (2,442 rows, S 2.14e7); re-check on the v63 chain **failed** (December interval +0.014%; eq -0.0325%, twelve 12/12): not built; override dry run on v63 319b6340 (2,399 rows, S 1.76e7) held for the user. Closed: own motion beyond 45-60 min, different-stand motion, early-record classes (78% of the listed error) | `case_files2.md` |
| 20 | case files on the worst over-predictions of the v63 chain (late class, layer overshoot) | fork | **closed, no GO** (over-predictions are 37.1% of ordinary dev squared error, none above 3,600 s; LIRF clock rows hedged toward the interval 34%, delayed flights anchored near the schedule 31% (bias changes sign by month, perfect fix <= 0.03%), late records unmarked; every shipped layer is a net gain by airport x direction, worst cell -0.016%; bound misfires inseparable) | `case_files3.md` |
| 21 | own-point turnaround features inside the context network (design batch, CPU on arcw0-7) | fork | running | `ctx_ownpoints.md` |

## Official results of the campaign

v58 = 220.9212 (intruder bound), v59 = 218.8483 (+ no-NM rematch, user override), v60 = 218.6404 (+ guarded stacking residual),
v61 = 218.6142 (+ same-type intruders by ICAO identity), v62 = 218.4174 (+ own-motion bound, tunnel 17): -2.91 s against v57 in one day.

## FIND -> v58 = 220.9212 official (-0.408 s vs v57): the intruder stand bound

Declared and accepted (`docs/intruder_bound/README.md`); built and uploaded at the user's word. Extensions closed: per-stand validity at
the other airports (no stand reaches 90% with >= 10 candidates), gaps 1-2 h (below the validity bar), no inbound rule (-0.006%).

If our inbound aircraft (same type) blocks in at our stand, then a different-type aircraft blocks in at that stand before our
predicted off-block (gap <= G), we must have left before the intruder: label > B = take-off - intruder in-block. Valid in 2025 at
EDDF 100%, EGLL 97%, LFPG 97%, LTFM 99.7% of candidates (all twelve months); invalid elsewhere (shared stand codes). Value B + o(airport,
gap band), o fitted on the other months (about 400-1,250 s). Scratch `runs/dig/micro/intruder_screen.py` (results
`intruder_screen.json`), `intruder_ranking.py`.

| variant | twelve months (older predictor) | shipped chain dev Jan / Jul / Nov / Dec | equal-footing mean |
|---|---|---|---|
| G 1,800 s, w 0.5 | -0.171%, 12/12 better, 12/12 intervals < 0 | -0.096 / -0.240 / -0.087 / -0.120% | -0.152% |
| G 3,600 s, w 1.0 | -0.376%, 12/12 better, 11/12 intervals < 0 | -0.169 / -1.159 / -0.229 / -0.204% | -0.511% |

Ranking dry run on v57 (read only): G 3,600 w 1.0 moves 52 rows (EDDF 9, EGLL 12, LFPG 17, LTFM 14), squared change 7.1e7, largest
move +3,050 s; G 1,800 w 0.5 moves 50 rows, 1.4e7. Arrival block, stand and type are supplied for every ranking arrival. Open before a
declaration: leakage review (July above 1%), the concentration of the change in ~50 rows, the order against the published-data layers.

## Small tunnels closed on the way (this session)

- **Long schedule intervals outside LIRF** (take-off more than 3 h after schedule): 2025 has 10,692 / 1,716 / 595 non-LIRF rows with
  intervals of 3-6 h / 6-12 h / above 12 h and only 8 / 0 / 0 labels above 3 h (median labels 1,017 / 964 / 1,064 s). No one-day
  mechanism outside LIRF; nothing to hedge.
- **The reconciliation's heavy-row cap** (64 per airport-day): reached on 2 of 619 ranking airport-days (LIRF July 2026), never in the
  development months. Negligible.
- **Daily fill share at LIRF** (clock-present rows, block = schedule within 5 s): 16.8% +- 3.0% per day; among rows with an NM delay
  above 600 s 7.9% +- 2.7%, about binomial. No day-level fill regime for the published total to reveal beyond one degree of freedom
  per day; a fill-aware posterior has the same ceiling as v48 / v57.
- **NM clocks after take-off** (proxy <= 0): 396 development rows, 0.35% of all squared error, bias -18 / +61 s. Even a perfect fix is
  under 0.2% of RMSE; the pipeline already treats them without bias. Closed.
- **Estimated NM clocks as the source of the "early" class** (block more than 300 s before AOBT_3): an estimated clock repeats one
  take-off-minus-clock minute per stand x runway pair. At LFPG and EGLL almost no row sits on such a modal value (4.9% / 0.0%) and the
  early share does not depend on it (LFPG 14.1% vs 14.9%). The signature exists at EDDM (82.5% modal rows; early 6% on modal rows vs
  53% on the others) and LTFM (40.5%), where the error is already small or the clock is known to be an estimate. Closed for the cells
  that matter.

- **ADS-B anchor for long measured taxis** (after v59's transfer): on gate-traced matched rows the chain follows the measured taxi in every
  band (bias -4 to +5 s up to 2,400 s; label and prediction sit the same ~45-60 s above the motion start). No extrapolation gap. Closed.
- **Gap-cut ADS-B segments** (the segmenter starts a departure after the last coverage gap > 600 s): almost all such gaps exceed an hour
  (ordinary parked periods); 600-1,800 s gaps are 6,600 dev rows with biases -5 to +45 s. 2026 state shares sit inside 2025's range
  apart from EDDF's known coverage drop (gate-traced 0.04-0.08 vs 0.14-0.41). Closed.
