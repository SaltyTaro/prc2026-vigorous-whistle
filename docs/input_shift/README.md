# Input-shift audit, 2025 → 2026 (27 September 2026). Label-free for the ranking; nothing fitted or selected

`tools/input_shift_audit.py` → [`input_shift.json`](input_shift.json), plus [`tables/`](tables/): clocks and arrivals per airport and
month, and the shift of every numeric feature.

The audit compares the ranking months (January and July 2026) with 2025 per airport. It never reads a ranking label; the ranking
file has 0 departure labels and 0 departure block times. The weights are feature importance: the mean normalized gain over the
twelve-month v7 nodes (globals and airports) and two committee LightGBMs.

The top gains are:

| feature | gain share |
|---|---:|
| `nmclock_aobt_movement_minus_s` | 17.1% |
| `nmclock_aobt_minus_eobt1_s` | 13.3% |
| `stand` | 8.5% |
| `nmclock_aobt_minus_iobt_s` | 6.0% |
| `surface_anchor_lookback_s` | 4.6% |
| `movement_minus_eobt1_s` | 3.4% |

Follow-up probes were run as scratch code. Their numbers are quoted below.

## Findings, ranked by likely impact

**1. LFPG, July 2026: EOBT_1 changed meaning. This looks like a format or process break.**

- The share of departures with EOBT_1 later than IOBT is 6–7% in every 2025 month and in January 2026. In July 2026 it is 35%.
- The change is uniform across all 31 days (48–57% of EOBT_1 still equal IOBT each day, against 71–81% before). It holds for every
  large carrier: AFR 4% → 33%, EZY 11% → 38%, EJU 8% → 24%.
- The new values sit at exactly +300 s (5.8%) and +600 s (13.2%) after IOBT. This looks as if EOBT_1 now holds an updated EOBT,
  where it used to hold the first-filed one.
- LOBT still equals IOBT in 99.9% of rows. AOBT_3 − MVT is stable (median 898 s → 959 s).
- Feature effect on all 21,743 LFPG July rows:
  - `nmclock_aobt_minus_eobt1_s` mean 752 s → 206 s (−0.62 SD); this is the largest importance-weighted shift in the whole bundle;
  - `movement_minus_eobt1_s` −0.55 SD;
  - `eobt_*` / `gd_*_eobt_*` also move.
  - Features built on EOBT_1 carry about 19% of gain.
- 2025 evidence: rows with EOBT_1 later than IOBT had labels +75 s above the rest (1,102 s vs 1,027 s), and v30 already under-predicts
  them by 34 s in development. In July 2026 the models see five times as many such rows.

**2. LTFM, July 2026: Turkish Airlines' 10-minute early IOBT disappeared. A format or practice break.**

- In every 2025 month and in January 2026, 14–21% of LTFM departures (24% of THY) have IOBT exactly 600 s before schedule. In July
  2026 it is 0.6% (THY 0.3%), uniform by day. IOBT equal to schedule rises from 56–66% to 82%.
- It is IOBT that changed, not the schedule: MVT − SCHED p25 is 905 s → 898 s, while MVT − IOBT p25 is 957 s → 840 s.
- At LTFM, EOBT_1 = LOBT = IOBT (97–100%) and AOBT_3 is an estimate. So `nmclock_aobt_minus_iobt_s` and `nmclock_aobt_minus_eobt1_s`
  move by about −600 s for roughly 4,500 July rows. The feature means drop by 0.22 SD over all 25,185 rows.
- 2025: those rows had shorter labels (1,047 s vs 1,083 s) and a development residual of +15 s, against +28 s on the rest.

**3. New stands: real infrastructure change, and unknown to the stand category and the frozen ADS-B stands table.**

- **EDDF:** Pier J opens (Terminal 3). J5, J10 and V333/V335/V337 carry about 590 movements, roughly 300 of them departures, in July
  2026. A26, A28, F225, V171A/B and V173A/B vanish.
- **EDDM:** stands 103–108 are new, with 2,354 movements (1,180 departures, 7.9% of EDDM July). Stands 110, 111, 183, 232 and
  254–256 vanish; 120/121 have been gone since August 2025.
- Stand carries 8.5% of gain. The novelty baseline, a stand unseen in the other 2025 months, is at most 0.01%.

**4. EHAM, January 2026: the NM record is missing for 5.0% of departures (1.6% in January 2025).**

- About 550 extra rows go to the no-clock route. This coincides with the disruption days, so it is a real-world change. The error map
  already shows v30 tracking those days.

**5. New operators and flight prefixes (real).**

- EDDF July: VL, i.e. Lufthansa City Airlines, 443 departures; 2.6% of rows have a new prefix and 2.5% a new operator.
- Elsewhere at most 0.9% of rows: AIC at LIRF, ITY at EGLL, WUK at LEBL, among others. The 2025 leave-one-month-out baseline is
  0.1–0.3%.

**6. Minor.**

- AOBT_3 carries non-zero seconds at LEBL (3.6%) and LFPG (2.8%) in January 2026. In 2025 the most was 1.5% at LEBL and 0.5% at
  LFPG. `nmclock_aobt_second_of_minute`
  is a feature.
- Arrival taxi-in medians move outside their 2025 range: LIRF January −52 s, EGLL January +47 s.
- LFPG uses the long runway more in July 2026 (`wx_runway_length_m` +0.44 SD).
- Winter-delay shifts in January 2026 (`dep_schedule_delay_mean_30m` +0.5 SD at five airports; AOBT → MVT p90 higher at EDDF, EDDM,
  EHAM and LSZH) look like real weather.

**Stable:**

- missingness of the NM fields, which move together as a unit;
- SCHED on a whole-minute grid (100%, and a 5-minute grid at LTFM);
- MVT and EOBT rounding;
- AOBT_3 − MVT at the airports with a real AOBT clock;
- arrival in-block and taxi-in presence (100%).

## Suggested levers (none started)

- **LTFM, a cheap and exact test:**
  1. On the development months, rewrite the 2025 rows with IOBT = SCHED − 600 into the 2026 convention: IOBT, EOBT_1 and LOBT set to
     SCHED where they equal IOBT.
  2. Rebuild the NM-clock and schedule features, re-predict with the saved fold models, and score. This measures what the shift costs.
  3. If it costs something, refit with 2025 converted. The inputs then match 2026 by construction.
- **LFPG:** the 2026 EOBT_1 cannot be reconstructed for 2025. Two options:
  - measure the sensitivity: re-predict July 2026 LFPG with EOBT_1 := IOBT on the changed rows, and see how far v30 moves;
  - run a development ablation that drops EOBT_1 features at LFPG. If it costs little, route LFPG July through that variant.
- **New stands:** learn positions for J5/J10/V33x and EDDM 103–108 from the 2026 adsb.lol traces. This is label-free, like the frozen
  1,332-stand table. It would improve ADS-B matching and stand imputation for about 1,500 departures.

## v31: LFPG EOBT_1 50% blend (27 September 2026). Official 229.5832, a new best

Built from the sensitivity study ([`lfpg_eobt_sensitivity.md`](lfpg_eobt_sensitivity.md)):

- **Recipe:** the recorded float v30 (`runs/adsb_ctx/export_v30/ranking_values.parquet`, which rounds exactly to the uploaded v30),
  plus 0.5 × (v30 with LFPG July-2026 EOBT_1 := IOBT − v30 unchanged). The unchanged v30 comes from the final-file runner's
  `final_lfpg_eobt` and `final_dryrun` runs. The effect is exactly zero outside LFPG July 2026. The floor is −12 s, then nearest
  integer.
- **Tool and record:** `tools/export_lfpg_eobt_blend.py --version 31` → [`export_v31.json`](export_v31.json).
- **What changed:** 19,661 of the 21,743 LFPG July rows change against v30 (mean +18.5 s, RMS 33.6 s, range −121 to +912 s); 0 rows
  outside that scope. Only 7 rows move by more than 300 s, all with announced delays of 1–2.7 h.
- **Cross-check:** the independent build is byte-identical,
  SHA-256 `ea1c17568e49b83cb1c8667212c2097ab8a8dec774b4db57e01b9b79bb531be2`.
- **Audit:** [`../export_audit/v31_audit.json`](../export_audit/v31_audit.json) is clean. Only LFPG differs from v30.
- **Upload:** approved by the user ("yeah upload it"). Uploaded at 08:29:23 UTC as the second upload of the UTC day and processed at
  08:30:00 UTC. The official score is **229.5832 on all 344,841 rows**, −0.0177 s (−0.0077%) against v30. The record is
  [`official_result_v31.json`](official_result_v31.json). Never re-upload v1–v31.

**What the score says.** The squared-error change is −2.80e6, which is −0.015% of the ranking SSE.

- The pre-upload bounds for the 50% blend were −0.71%, −0.23% and +0.20 to +0.28%. The result falls between the second and third
  readings, close to neutral.
- In aggregate it corresponds to about 28% of the reset effect being real bias (ΔSSE = D·(w² − 2wf), with D = 9.79e7 and w = 0.5).
  The best weight would then have been about 0.28, which is only about 0.03 s better than 0.5.
- This comes from one aggregate score, not a row-level signal. Tuning the LFPG weight on further uploads would be leaderboard
  fitting, so the lever is closed at v31.

## LTFM IOBT-convention sensitivity (27 September 2026). Descriptive; nothing selected or exported

[`ltfm_iobt_sensitivity.md`](ltfm_iobt_sensitivity.md) is the analogue of the LFPG work for finding 2.

- **The convention.** Only THY used the early IOBT, on about 29% of eligible rows. It is not a per-flight-number property, and 55% of
  the July-2026 THY callsigns are new. So the declared restoration covers all 16,618 July THY rows with IOBT = SCHED, each weighted by
  its label-free probability q (mean 0.296).
- **The runner.** The effect is exactly zero outside LTFM July. v30 moves +41 s on the restored rows and +21 s on the other LTFM July
  rows (context networks).
- **The 2025 anchor.** Early-type rows taxi 27–65 s longer at the same MVT − SCHED, and v30 learned that.
- **The bounds.** The soft hedge v31 + w·q·Δ gains 0.008–0.019% of SSE at w = 0.5 under a format change, or costs 0.005% if the 2026
  rows are genuinely IOBT = SCHED rows. Full unweighted restoration is wrong under any plausible reading.
- **Decision.** Whether to build a w-blend is the user's call.
