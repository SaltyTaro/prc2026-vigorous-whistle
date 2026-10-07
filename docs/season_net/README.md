# Seasonal-weight lever: season-weighted own-points context network (fork S, 7 October 2026)

Declaration [`configs/season_net_design_v1.json`](../../configs/season_net_design_v1.json), committed before any fit. Design batch on
October 2025 (lever 9 / 12's readout): own_all (dl32own on matched rows + m25dl32own on the others) refitted with a seasonal loss weight
w = 1 + a exp(-d^2 / (2 x 1.5^2)) (d = circular month distance to the target month, mean weight 1), a in {1, 3}, 8 seeds, mixed into the
shipped member at c in {0.5, 1.0}; GO bar +0.05 pts at b = 1.0, non-LIRF RMSE below the control, paired-day interval below zero.

- Runner `tools/run_season_net_ctx3.py` = `tools/run_adsb_ctx3.py` unchanged, training swapped for `tools/season_net_train.train_weighted`
  (line-for-line copy of `prc2026_ctx3.train.train` with the weight); `tools/season_net_selftest.py` checks bit-for-bit identity at w = 1
  (plain and masked), that weights reach the loss, the source diff, and the month weights. Passed locally before the declaration.
- Ops (copies of lever 12's): `tools/season_net_{nodes.sh,run1.sh,guard.sh,dispatch.py,check_features.py,jobs.py}`, clusters
  prc-ctx7a-l (`infra/aws/sky_ctx7_*.yaml`, `infra/aws/gpu_authorization_20261007c.json`); report `tools/season_net_design_report.py`.

Results: pending.
