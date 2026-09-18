# Deep audit of the v18 recipe (18 September 2026, evening)

Read-only review of everything that feeds `vigorous-whistle_v18` (official 241.4077), asked for
by the user ("maybe we missed something, even if it's small we can test it because we have the
luxury of time and compute"). Development tables only (Jan / Jul / Nov 2025 with the incumbent's
out-of-sample predictions, `tools/audit_export.py::development_reference` with
`runs/adsb_v20/ens_v20_bag_q_all` and `runs/adsb_v17_sel`); no December row, no ranking label.
Every number below is descriptive (reused development data); nothing here selects anything.

## 1. Where the development squared error actually sits

All 506,751 development rows, incumbent predictions (330.302 / 271.528 / 192.187):

| class | rows Jan / Jul / Nov | share of SSE Jan / Jul / Nov | RMSE |
| --- | ---: | --- | ---: |
| LIRF population (the fill mixture) | 60 / 337 / 52 | 5.8 / 25.4 / 22.4% | 2,817 |
| LIRF non-population (ordinary LIRF rows, all with an NM clock) | 11,257 / 14,874 / 11,276 | 5.4 / 20.6 / 16.1% | 357 |
| unmatched (no surface segment) | 66k / 88k / 44k | 80.8 / 38.3 / 34.5% | 233 |
| dark (segment starts moving) | 50k / 54k / 65k | 6.6 / 12.5 / 22.1% | 157 |
| gate (segment starts at the stand) | 26k / 33k / 42k | 1.3 / 3.2 / 4.9% | 98 |

January's "80.8% unmatched" is two LFPG rows (labels 84,240 s and 58,206 s, predictions ~800 s):
28 rows with labels above six hours carry 34.7% of the three months' SSE and are not a modelling
target (v8's hedge lesson). On the **ordinary rows (label <= 6 h)** the picture is:

| class | share of ordinary SSE | rows | RMSE incumbent | RMSE v9 | RMSE committee alone |
| --- | ---: | ---: | ---: | ---: | ---: |
| unmatched (non-LIRF) | 44.7% | 197,778 | 233.2 | 238.1 | 234.1 |
| **LIRF non-population** | **19.8%** | 37,406 | 357.2 | 357.2 (= incumbent) | 507.2 (never fitted on LIRF) |
| dark | 17.4% | 170,326 | 156.8 | 200.8 | 158.6 |
| LIRF population | 14.0% | 424 | 2,817 | = incumbent | - |
| gate | 4.0% | 100,789 | 98.2 | 169.2 | 100.4 |

By airport (ordinary rows): LIRF 33.8%, LTFM 15.9%, LFPG 15.2%, EGLL 13.0%, LEMD 5.1%, LEBL 4.8%,
EHAM 3.8%, EDDF 3.5%, LSZH 2.7%, EDDM 2.2%.

The committee's gain over v9 is a matched-row gain: -8% (LFPG) to -48% (EDDM) on matched rows,
but only -0.5% (LSZH) to -5% (LEBL), mean about -2%, on unmatched rows.

**LIRF non-population rows are the largest bucket that no learner has ever been fitted on.**
They are ordinary departures with an NM clock (100% have `AOBT_3`), but at LIRF that clock is
weak: `label - proxy` has median +63 s, IQR -116 ... +305 s, 1-99% -544 ... +1,915 s; the proxy
alone scores 496 s where the incumbent (v7's timeline models) scores 317 s on the non-fill rows.
13-18% of these rows are BLOCK = SCHED fills (label = schedule interval exactly), which carry
34% of the class's SSE; the remaining 66% is ordinary error at 259 / 382 / 267 s (Jan / Jul / Nov).
Both learner attempts recorded for LIRF were *residual* models trained on one month of LIRF
matched rows (v12 residual: LIRF +1.0 / +1.2%; v15 item 6 fill classifier: +2.8 / +2.0%) and the
full-data learners were fitted without LIRF because the population rows' 80,000-second labels
break early stopping. A full-data learner with LIRF *non-population* rows in its fitting set has
not been tried. LIRF ADS-B coverage: 0% Jan-Apr 2025, 10% May, 42-94% Jun-Nov 2025,
**80% in January 2026 (gate 18%)**, 0.2% in July 2026 - the dev months cover both regimes
(Jan 2025 none, Jul 2025 94%, Nov 2025 66%).

## 2. Recipe constants (descriptive scans on the recorded development tables)

- Blend weight (learner share on non-LIRF rows): 0.65 +0.011%, **0.70 0**, 0.75 +0.007%,
  0.80 +0.032% mean over the three months. At the optimum.
- v12 residual stage: removing it costs +0.58 / +0.68 / +1.60% (mean +0.95%); scale 1.5
  +0.05%. It matters and is at its optimal scale. It is a single-seed LightGBM fitted on
  `matches_v11` (1,111 stands) - the learners use `matches` (1,332 stands). Seed bagging and the
  newer match table are untested for this stage (cheap, local, minutes).
- Global rescaling of the final prediction: optimal factor 1.006 / 1.010 / 1.019 per month -
  the models are unbiased to within noise; no calibration lever.
- Refit rounds: the median of the two folds' best iterations, fitted on 22-57% more rows than
  the folds (1.77M vs 1.13M / 1.45M). Standard practice scales rounds up with the fitting set;
  the curves' tails are flat to 0.01 s over the last five iterations, so the loss is small but
  unmeasured.

## 3. Data path

- **Day-boundary truncation** (`tools/build_adsb_features.py::one_day` loads only the departure
  day's surface file; `segment()` looks back four hours): departures at 00:00-00:10 UTC are
  matched 10.8% / gate 2.8% of matched, against 13-29% / 29-36% at 01:00-04:00; 23:50-24:00
  rows (airborne point after midnight) are matched 16%. Affected: ~0.15% of rows carrying
  0.36% of the non-LIRF SSE (534 + 370 ranking rows). Consistent between training and ranking,
  so the models never saw the correct features for these rows. Fix: load day +- 1; rebuild the
  13 match tables; re-predict with the existing models (or refit). Worth about 0.05%.
- **Match rate / callsign fallback / complete-match scope**: closed on 16 Sep (v15 items 4-5).
- **Alternative ADS-B archives** (probed tonight): airplanes.live serves daily `globe_history`
  traces for January 2026 (HTTP 200; adsb.fi, theairtraffic.com and ADS-B Exchange refuse or
  404). For 16 LFPG aircraft of 15 Jan 2026 it holds **0 ground points** (adsb.lol: 25) and
  fewer in-box points; for six Turkish-carrier aircraft it has Istanbul approach points down to
  200 ft but **no LTFM ground point** (adsb.lol has no LTFM point at all). No second network
  adds surface coverage at LFPG or LTFM. Closed.
- **December 2025** is labelled, in the raw files, and has never been used for fitting, early
  stopping, scoring or ADS-B extraction (`surface_202512*` absent; the tools refuse it). It is
  the month adjacent to January 2026 (44% of the ranking) and would add 9% to the fitting set
  (165,677 departures). Two uses, in order: (a) score December with the shipped refit models -
  a free, genuinely out-of-sample check of the v18 recipe that no dev fold gives; (b) add it to
  the learners' and residual's refit months. Both need the user to lift the reservation
 . Expected value of (b)
  is unmeasured; an analog (forward fold with October merged into the fitting set at fixed
  rounds) can estimate the value of one adjacent month before touching December.
- **Covariate shift of the ADS-B features between 2025 and the ranking months**: see section 5.

## 4. Levers ranked (expected value, cost)

| # | lever | evidence | expected | cost |
| --- | --- | --- | --- | --- |
| A | learners fitted **with LIRF non-population rows** (new `--exclude-population` flag; population = LIRF & no AOBT/LOBT & no type), blended on LIRF non-population rows only, incumbent elsewhere | section 1: 19.8% of ordinary SSE, no learner ever fitted; Jan-2026 LIRF 80% matched | 5% on the class = about -0.4% total; uncertain (two residual attempts lost) | LightGBM screen: 1 VM, ~1 h; CatBoost round: 4 VMs, ~6 h |
| B | December: (a) score it with the shipped models; (b) refit on 12 months | section 3 | (a) evidence only; (b) unmeasured, positive in expectation | (a) hours; (b) Dec ADS-B extraction + full refit, ~1 day |
| C | v12 residual stage: 5-seed bag, `matches` table | section 2 | 0.03-0.1% | local, minutes |
| D | day +- 1 surface loading | section 3 | about 0.05% | rebuild match tables + re-predict, ~half a day |
| E | refit rounds scaled to the fitting set (x1.2) | section 2 | <= 0.05% | measurable in the same analog run as B(b) |
| F | CatBoost variants (lr 0.03 / 40k cap; l2 30; rsm 0.5) | none; ctr2 seeds spent | 0.05-0.2% | 4 VMs, ~1 day |

Checked and closed tonight: blend weight, residual scale, calibration, alternative ADS-B
networks. Closed earlier and not reopened: OPDI, OpenSky, test-time adaptation, v19 surface
features, airport specialists, ADS-B-residual target, LIRF residual/mixture with ADS-B,
arrival context, match-rate recovery, more committee seeds.

## 5. Covariate shift of the ADS-B features (2025 fitting months vs the ranking months)

Count and density features stay inside the fitted range: for `adsb_points`, `nb_airport_10m_count`,
`nb_runway_10m_count` and `nb_runway_30m_count` the ranking p99 is at or below the training
p99.9 at every airport-month; the only excess is EGLL July 2026 `nb_runway_30m_count` (3.7% of
rows above the training p99.9 of 45 - the p99 is 47). No extrapolation risk from the trees.

Regime shifts (same feature ranges, different mix), all consistent with the export audit:

| airport | 2025 dev months (matched / gate-of-matched / median last-ground s) | 2026-01 | 2026-07 |
| --- | --- | --- | --- |
| EGLL | 0.71 / 0.24 / 229; 0.23 / 0.13 / 261; 0.69 / 0.27 / 199 | 0.88 / 0.27 / 19 | **0.97 / 0.46 / -5** (complete, dense) |
| EDDF | 0.92-0.96 / 0.15-0.43 / 48-60 | 0.79 / 0.10 / 96 | 0.76 / 0.05 / 64 (thin) |
| LFPG | 0.00 / -; 0.07 / 0.04 / 123; 0.36 / 0.01 / 103 | 0.21 / 0.01 / **2,735** | 0.21 / 0.01 / **4,327** (matches are mostly the arrival rollout) |
| LIRF | 0.00; 0.94 / 0.09 / 22; 0.66 / 0.15 / 220 | 0.80 / 0.18 / 37 | 0.00 |
| EDDM, EHAM, LEBL, LEMD, LSZH | dense in 2025 (except EDDM Jul, LEMD Jan) | dense | dense |

EGLL July 2026 is the one airport-month whose rows look unlike any EGLL training month (EHAM-like
complete gate segments); the models learned gate behaviour mostly at other airports and EGLL's
block-before-motion offset from thinner 2025 coverage. Nothing to act on without labels; noted
for the release documentation.
