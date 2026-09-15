# v12_v3: the full-data learner stacked into the ADS-B residual (15 September 2026)

Declared in [`../../configs/adsb_v12_v3.json`](../../configs/adsb_v12_v3.json)
after v12 (official 246.4831, 3rd of 144; 2nd 245.2932, 1st 245.0207). Same
residual recipe as v12 (LightGBM on label − v9, leave-one-month-out over
Jan/Jul/Nov 2025, LIRF excluded, neighbour features) with one extra input: the
out-of-sample prediction of the **full-data ADS-B learner**
([`../adsb_full/README.md`](../adsb_full/README.md); `tools/run_adsb_full.py`,
no-LIRF variant) as the features `adsb_full` and `adsb_full_minus_v9`. The
learner sees all eleven 2025 months with the ADS-B features, where the residual
model alone sees two (development) or three (export); its fold predictions are
out-of-sample for Jan/Jul (seasonal fold) and Nov (forward fold), and the refit
on the eleven months predicts the ranking. Candidates: scope `matched_no_lirf`
(as v12) or `all_no_lirf` (every non-LIRF row, since the learner predicts every
row), weights 1.0 and 0.75. Baseline for selection: the recorded v12 development
predictions (332.993 / 274.585 / 198.544); rule: not worse on any month and at
least 0.3% better on average. Inputs are hash-bound (v12 matches snapshot, the
learner's run records and predictions, the v12 residuals and export).
