# Round C — residual-stage context network (adsb_rctx), 26 September 2026: design NO-GO

- **Question:** does a context network (src/prc2026_ctx, attention over up to 128 neighbour movements) improve the residual stage
  when fitted as rnet is (rows matched_no_lirf, target label − cross-fit v9, the residual trees' nb columns, leave-one-month-out)?
  Requested with the user's "Ok proceed" (CPU only).
- **Tool:** [`tools/run_adsb_rctx.py`](../../tools/run_adsb_rctx.py) (rnet's data path from tools/run_adsb_rnn.py, imported unchanged,
  plus ContextNet and each row's context tokens; modes --design, --held-out M, --refit12), network config
  [`configs/adsb_rctx_network_v1.json`](../../configs/adsb_rctx_network_v1.json) (the committee context network's architecture,
  16-epoch budget and k = 128 window). Smoke tests of the design and refit12 modes passed on arcw2 before the declaration.
- **Design declaration** ([`configs/adsb_rctx_design_v1.json`](../../configs/adsb_rctx_design_v1.json), 04:58 UTC, commit `7128e87`,
  before any design fit): 4 seeds on October (fit on the other eleven months' in-scope rows, monitor October). GO only if the 4-seed
  mean at its best w in the stage 0.05 adj2 + 0.05 cat + 0.9 ((1 − w) rnet + w rctx) reads at least 0.02% below the control on v28's
  October recipe (all rows) with the paired-day interval below zero.
- **Result** ([`design_report.json`](design_report.json), `tools/rctx_design_report.py`): **NO-GO.**

  | Reading (October) | Scope RMSE (s) | Error corr. with rnet | Best w | Recipe change at best w | Interval (s) |
  |---|---|---|---|---|---|
  | control (rnet 16 seeds in the stage) | 112.550 | — | — | — | — |
  | rnet alone, 1 / 4 / 16 seeds | 115.876 / 113.624 / 112.762 | — | — | — | — |
  | rctx seeds 1–4 alone | 115.22 / 116.13 / 114.87 / 114.55 | 0.969–0.972 | 0.1–0.4 | +0.003% … −0.008% | all include 0 |
  | **rctx 4-seed mean** | 113.263 | 0.987 | 0.2 | **−0.0012%** (bar −0.02%) | [−0.014, +0.008] |

  Seed for seed the context version is slightly better than rnet (4 seeds: 113.26 against 113.62 s), but its errors are 0.987
  correlated with rnet's, so mixing it into the 16-seed rnet stage changes the recipe by about one thousandth of a percent. On the
  residual stage's rows (all ADS-B-matched) the neighbour tokens add little beyond the surface features rnet already has.
- **Decision:** Round C stops with this recorded null; no development round, export or upload. Compute: four 16-epoch fits on arcw2
  (about 45 minutes each, two lanes), 04:59–06:30 UTC.
