# Context network GPU sprint (adsb_ctx3), 26 September 2026

**Question.** Does anything in the context network's training beat round A's `ctx_e24` settings on the October design fold: masking
the focal flight's own ADS-B inputs during training (so the network learns more from matched rows about how to predict unmatched
ones), auxiliary heads that predict the hidden ADS-B taxi path, stopped time, stops and slow start on those masked rows, a wider or
deeper tabular branch, wider numeric embeddings, 32 epochs, a doubled learning rate, or weight-averaged readouts (EMA, mean of the
last six epochs)?

**Why.** Unmatched non-LIRF rows carry 29% of v28's development SSE (LTFM 10.5%, LFPG 9.5%, EGLL 4.7%); matched rows 15%
([`error_budget_v28.json`](../adsb_ctx/error_budget_v28.json)). Round 2 ([`docs/adsb_ctx2/README.md`](../adsb_ctx2/README.md))
closed the neighbour-side architecture (window, token width, self-attention); this sprint tests the focal side and training.

**Declaration:** [`configs/adsb_ctx3_design_v1.json`](../../configs/adsb_ctx3_design_v1.json), before any fit.
- Ten variants × seeds 1–4 (40 runs): `ref` (= ctx_e24), `mask25`, `mask50`, `aux25`, `aux50`, `wide` (width 1024), `deep`
  (depth 4), `dnum16`, `e32`, `lr2`. Every run also saves and predicts its EMA and last-six-epoch-mean readouts.
- Readings: the incumbent network table on October (P = 0.5 ens_ctx + 0.5 ens_ctx_e24, CPU seasonal-fold predictions); recipe
  proxy with table = 0.25 committee + 0.75 [(1 − b) P + b X].
- **GO** per item: R2 at b = 0.5 at least 0.05 points below `ref`'s, 4-seed standalone RMSE below `ref`'s, and the paired-day
  interval below zero. GO items go to a separately declared development round; two or more GO items first get a declared
  combination batch.
- Code: `src/prc2026_ctx3/` (ContextNet2 plus an optional auxiliary head; masking and readouts in `train.py`),
  `tools/run_adsb_ctx3.py`, `tools/ctx3_design_chain.sh`, `tools/ctx3_nodes.sh`, `tools/ctx3_design_report.py`;
  tests `tests/test_ctx3.py` (with everything off, training equals round 2's exactly).

**Hardware.** Four AWS g6e.2xlarge (one L40S each) in us-east-2 via SkyPilot, two lanes per GPU
([`infra/aws/gpu_authorization_20260926.json`](../../infra/aws/gpu_authorization_20260926.json)).
