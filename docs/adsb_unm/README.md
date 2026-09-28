# Unmatched-row specialist design (configs/adsb_unm_design_v1.json) — no GO (28 September 2026)

Declared 27 September 12:34 UTC; arcw2's two runs completed that day, the other seven workers ran their chains on 28 September
(14:59–17:13 UTC, launched with `tools/launch_adsb_unm.py`; the launcher's final ssh calls timed out after starting the chains, which
was harmless). All 22 design runs mirrored to `runs/adsb_unm/design/`; report `tools/unm_design_report.py` →
[`design_report_1.json`](design_report_1.json). October only; no scoring month was read.

- The reproduction run equals `lgb_x19_s24`'s October column exactly (0.0 s); the unmatched flag agrees with the learners' pass-2 flag (1.0000).
- Control recipe proxy 175.912 s; committee on October's unmatched non-LIRF rows 215.879 s.
- Specialists fitted only on unmatched rows (`unm_*`) or with unmatched rows up-weighted ×2 / ×4 (`w2_*`, `w4_*`): standalone
  unmatched RMSE 215.3–216.7 s (CatBoost) and 228.1–229.2 s (LightGBM) against their seed-matched all-row members' 215.2 / 228.0 s.
  Every recipe reading is within ±0.04% of the member control; the three bags read −0.0025% … +0.0042% at c = 0.5 with intervals
  straddling zero. **GO list empty.**

Reading: the committee's all-row members already spend their capacity where the unmatched rows need it; re-weighting or restricting the
fit changes nothing. The unmatched rows' remaining error is not a fitting-emphasis problem. Lever closed; no refit, export or upload.
VMs deallocated after the mirror.
