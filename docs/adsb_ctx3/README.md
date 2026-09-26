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

## Result (26 September, runs 14:01–14:46 UTC) — [`design_report_1.json`](design_report_1.json), **one GO: `e32`**

All 40 runs finished without a failure (about 7 min per fit, 9.7 min at 32 epochs, two lanes per L40S).

October non-LIRF RMSE of the incumbent network table P (32 CPU seeds): 154.016 s; recipe proxy at share 0.75: 176.262. Gains are
points of R2 at b = 0.5 against `ref` (4 seeds each, seeds 1–4); positive = better.

| item | 4-seed R1 (s) | gain (pts) | interval (s) | unmatched / matched vs ref | decision |
|---|---:|---:|---|---|---|
| ref (= ctx_e24) | 156.199 | — | — | — | — |
| **e32** | **156.072** | **+0.057** | **[−0.19, −0.01]** | −0.09% / −0.07% | **GO** |
| dnum16 | 156.153 | +0.033 | [−0.25, +0.11] | −0.07% / −0.01% | no |
| lr2 | 155.666 | +0.027 | [−0.17, +0.06] | −0.06% / −0.00% | no |
| mask25 | 155.709 | +0.025 | [−0.17, +0.08] | **−0.20%** / +0.25% | no |
| deep | 156.246 | −0.018 | [−0.07, +0.13] | −0.01% / +0.09% | no |
| mask50 | 156.405 | −0.026 | [−0.07, +0.16] | **−0.20%** / +0.45% | no |
| aux25 | 155.972 | −0.039 | [−0.04, +0.17] | +0.02% / +0.13% | no |
| aux50 | 156.546 | −0.089 | [+0.01, +0.31] | −0.01% / +0.37% | no |
| wide | 157.200 | −0.089 | [+0.05, +0.27] | +0.22% / −0.03% | no |
| ref EMA readout | 156.239 | −0.029 | [+0.03, +0.08] | | no |
| ref last-6-epoch mean | 156.199 | −0.018 | [+0.02, +0.05] | | no |

- **e32 is the only GO, narrowly** (0.057 against a 0.05 bar). As with 24 epochs in round 2, each single e32 network is worse
  than a single ref network (165.1 / 163.6 / 164.6 / 165.5 against 164.0 / 163.4 / 162.9 / 164.2 s), while the 4-seed mean is
  better: longer training makes the seeds more diverse, so the gain lives in the seed average.
- **Masking works where it was aimed, but costs more than it gives:** unmatched rows improve 0.2% (EGLL, LFPG, LTFM unmatched all
  better), while matched rows lose 0.25–0.45%. The auxiliary heads do not help on top of masking. Routing a masked network to
  unmatched rows only was not declared here and would need its own batch with fresh seeds.
- Capacity (wide, deep) and weight averaging (EMA, last-epoch mean) do not help; dnum16 and lr2 lean positive below the bar.
- Hardware note (descriptive): the sprint's ref seeds 1–2 score 164.02 / 163.44 against round 2's g_e24 seeds 1–2 at 162.95 /
  162.81 — the same settings on a different GPU session; single-seed differences of this size are within the seed spread.

Per the declaration, e32 goes to a separately declared development round (hardware named) against the incumbent.
