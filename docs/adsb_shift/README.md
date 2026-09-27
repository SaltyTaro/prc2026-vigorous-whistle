# ADS-B feature-shift audit, 2025 → 2026 (27 September 2026). Label-free for the ranking; nothing fitted or selected

`tools/adsb_shift_audit.py` → [`adsb_shift.json`](adsb_shift.json) and [`adsb_shift.csv`](adsb_shift.csv).

- **Scope:** per airport, gate and dark matched rows separately, January 2026 against January 2025 and July 2026 against July 2025.
  Every ADS-B, v19 and neighbour feature is compared, along with the gate/dark/unmatched mix.
- **Inputs:** the committee learners' ADS-B tables (`runs/adsb_ctx2/inputs/adsb_features_arcw0`) and the v19 extras.
- **Weights:** gain from two committee LightGBMs and the adj2 residual booster. `adsb_motion_s` carries 11.5% of gain, the first
  point 2.0%, slow start 2.2%, anchor/first stand distance 1.5% / 1.2%. Each `nb_*` carries ≤ 0.2%. In the x19 CatBoost
  (`cat_ctr2_x19_s27`), `adsb_motion_s` is 6.1% and all `sfc_*` together are 0.44%.
- **Anchor:** scratch code, quoted below. In 2025 development (four recipe months, labels ≤ 3 h), v30's prediction minus
  `adsb_motion_s` matches label minus `adsb_motion_s` within about 10 s in every airport × status × coverage bin. In 2026 the same
  statistic is computed for v31 and compared bin by bin. The bins are taxi-path length, points, and points per minute.

## Findings

**1. EGLL, July 2026: a receiver coverage jump. Real, but it breaks two feature meanings.**

- The status mix changes: unmatched 77% → 3%, gate 3% → 45%.
- Taxi path medians are 3.1–3.2 km, against 0.2 km in July 2025. January, November and December 2025 already had full paths, so this
  is not outside the training data as a whole.
- **Meaning break 1, the dark class:** July-2026 dark rows are first seen 122 m from the stand (1,376 m in 2025) and now show a slow
  start (median 340 s, against 7 s). They are gate-like observations the matcher does not label as gate.
- **Meaning break 2, the `sfc_*` counts:** they measure visible aircraft, so they jump 5–7× (ground aircraft within 10 min: 37 vs 7).
  This is coverage, not congestion. Their weight is small (≤ 0.44%).
- **Anchor** on full-path rows (> 2 km), by points per minute: v31 − motion sits 37–53 s *below* 2025's label − motion for gate rows
  (268 vs 304, 304 vs 357, 251 vs 299). The dark rows sit at the same level as the 2025 gate relation, which is where they belong now.
- A possible under-prediction of about 45 s on about 14,000 rows, ≤ 0.16% of SSE. There is no clean repair: nothing can be undone,
  unlike LFPG's EOBT_1.

**2. EDDF: coverage thinning, already known ([coverage regime](../coverage_regime/)).**

- Gate share falls from 14–17% to 4–8%.
- Dark rows with a spurious `adsb_motion_s` > 3,000 s rise from 3–6% to 12–17%.
- The first point sits 1.5 km from the stand, against 0.5 km.
- The spurious long segments existed in 2025 too and are handled: 2025 labels 691 s, predictions 699 s.
- **Anchor:**
  - July-2026 gate rows sit +60 to +115 s above the 2025 relation per points bin, but there are only 813 rows.
  - January-2026 dark rows sit +80 to +135 s above it on about 9,900 rows. Winter disruption days confound this.
  - July dark rows are within ±30 s, except the sparsest bins.

**3. LFPG dark rows (about 4,000 per month):** spurious long segments rise from 33% to 50–59%, with a median of 4 points per segment.
v30 already discounts LFPG dark motion (2025: label 1,023 s, prediction 1,018 s).

**4. Real coverage appearing or disappearing, already known:**

- LEMD January is 0% matched in 2025 and 96% in 2026.
- LIRF July 2026 is dark.
- LTFM has no ADS-B at all.

**Stable:** EDDM, EHAM, LEBL and LSZH. On gate and dark rows, prediction minus motion stays within about ±20 s of the 2025 relation
in both ranking months, apart from small winter offsets in January.

## Verdict

No format break comparable to LFPG's EOBT_1 has a clean repair. Every shift is a receiver-coverage change. The largest weighted item
is EGLL July 2026, a possible −45 s on about 14,000 rows. The only fix would be a coverage-aware model round, for example refitting
with EGLL 2025 months thinned or densified to the ranking regime. The earlier coverage-regime study found that avenue gives little
back.
