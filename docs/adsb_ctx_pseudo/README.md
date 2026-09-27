# Pseudo-label adaptation test (27 September 2026) — [`configs/adsb_ctx_pseudo_v1.json`](../../configs/adsb_ctx_pseudo_v1.json), [`report.json`](report.json): harmful, closed

The idea ("swing big"): let the models learn the scored period's own conditions by adding its ADS-B-seen departures to the fit with
pseudo-labels from their own surface traces (the 2026 ranking months differ from 2025: docs/input_shift/). Simulated on December 2025
(true labels only score): ctx_e32 refit fold (January–November) plus December's gate rows (`pg`: adsb_motion_s + the per-airport
January–November median of label − adsb_motion_s; 46,478 rows) or gate + dark rows (`pgd`: dark rows from adsb_first_point_s; 116,713
rows), seed-matched to GPU development seeds 401/402/404. `tools/run_adsb_ctx_pseudo.py`, `tools/ctx_pseudo_report.py`.

| variant | pseudo-label RMSE vs true | December non-LIRF change | gate | dark | unmatched |
|---|---:|---:|---:|---:|---:|
| pg | 253.9 s | **+143%** | +176% | +241% | +62% |
| pgd | 1,644 s | **+763%** | +174% | +1,417% | +60% |

- Training itself looks normal (same target statistics and loss as the controls), but the networks learn that December rows carry the
  pseudo-labels' distribution (median prediction 813 s vs 916 s; the pseudo-labels miss the label tail) and apply it to every December
  row, including the unmatched rows that were never pseudo-labelled. Surface-trace pseudo-labels are far noisier (254 s RMSE on gate
  rows, correlation 0.78) than the models' own gate-row predictions (~87 s), so they cannot teach the period anything reliable.
- Closed. A period-adaptation idea would need pseudo-labels better than the models' own predictions, which the traces do not give.
