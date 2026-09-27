# Per-airport fine-tuning test (27 September 2026) — [`configs/adsb_ctx_ft_design_v1.json`](../../configs/adsb_ctx_ft_design_v1.json), [`design_report.json`](design_report.json): no GO, closed

The four GPU-sprint e32 design networks (seasonal fold's fit partition, October held out) were each fine-tuned on one airport's
fitting rows (6 epochs, lr 1e-4 `lo` / 3e-4 `hi`, from the base weights and preprocessor) and re-predicted that airport's October rows;
4-seed means against the same four base networks. `tools/run_adsb_ctx_ft.py`, `tools/ctx_ft_report.py`; 72 fine-tunes on four L40S.

| airport | Oct rows | base RMSE | lo | hi | hi, unmatched rows |
|---|---:|---:|---:|---:|---:|
| EDDM | 15,469 | 70.1 | −0.35% | −0.50% | −0.88% |
| EDDF | 21,239 | 96.0 | −0.08% | −0.40% | −0.43% |
| LSZH | 12,302 | 98.2 | −0.05% | −0.32% | +0.46% |
| LFPG | 20,809 | 225.0 | −0.14% | −0.20% | −0.07% |
| EHAM | 21,784 | 97.5 | 0.00% | −0.17% | +0.39% |
| LEMD | 18,892 | 135.5 | −0.11% | −0.14% | −0.15% |
| EGLL | 20,657 | 214.9 | +0.02% | +0.03% | +0.18% |
| LEBL | 16,300 | 141.7 | +0.16% | +0.29% | −0.28% |
| LTFM | 24,080 | 214.4 | +0.12% | +0.19% | +0.19% |
| **all** | 171,532 | 159.9 | −0.005% | **−0.026%** (interval [−0.16, +0.08] s) | |

No airport reaches the declared −1.0%; the two largest error pools (LTFM, EGLL) get worse. The global network already uses its
airport information; per-airport specialisation of the network is closed.
