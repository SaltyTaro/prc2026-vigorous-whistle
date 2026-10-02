# LIRF dark diversity network, design (2 October 2026, after v55): no GO — closed

Declaration [`configs/lirf_dark_div_design_v1.json`](../../configs/lirf_dark_div_design_v1.json) and network
[`configs/lirf_dark_div_network_v1.json`](../../configs/lirf_dark_div_network_v1.json) (06:11:52 UTC, f5a167d, before any fit), scorer
`tools/lirf_dark_div_report.py` (same commit), report [`design_report.json`](design_report.json). Nothing is exported or uploaded.

## Question

v54 / v55 use M = lm25_32 (the masked LIRF context network, 16 seeds 801–816) at w 0.70 inside the LIRF network with network share
0.75 in dark LIRF months (July 2026). Outside LIRF the combination batch's dl settings (numeric embeddings 16 wide, learning rate 0.002)
gained only through diversity, and the 2 October matched-row member (mw40) gained the same way. Does a second masked LIRF network with
the dl settings, `lm25dl32`, averaged with M (M' = 0.5 M + 0.5 lm25dl32) beat M in v54's dark arithmetic (network' = 0.30 shipped +
0.70 M', blend 0.75 / 0.25 tree, gate u600)?

## Fits

- lm25dl32 = lm25_32 with d_num 16 and lr 0.002, every other resolved setting identical (checked on the resolved variant before the
  declaration). 16 seeds (1001–1016) × 2 folds (`seasonal` for thinned July, `lomo_2025-01` for January 2025) = 32 fits,
  `tools/run_adsb_ctx_lomo.py --device cuda` through `tools/combo_queue.sh` (ROUND=runs/lirf_dark_div).
- Hardware: AWS g6e.2xlarge (one L40S each), `infra/aws/gpu_authorization_20261002.json`. Four nodes were requested; prc-ctx3d failed
  to provision (InsufficientInstanceCapacity for g6e.2xlarge in us-east-2a/b/c; nothing was created), so the 32 fits ran on three nodes
  (prc-ctx3a/b/c, two lanes each; job lists 12 / 10 / 10). A first worker launch command failed on a quoting error before any
  provisioning. Queues 06:34–08:10 UTC, 32 of 32, none failed; the quality monitor flagged no fit (seasonal 11.5–12.5 min, January fold
  13.8–17.6 min per fit; last-epoch loss 0.099–0.105 / 0.113–0.120).
- Before the fits every node's timeline bundle (14 files) and token build matched the workstation's hashes and the context tests passed.
  Every mirrored file was SHA-256-checked against its node (65 + 65 + 78 files); each node was terminated right after its mirror (b and c
  07:55, a 08:11 UTC; the jobs controller untouched).
- Thinned July: the 16 seasonal models re-predicted on coverage-regime attempt a13 on the workstation CPU
  (`tools/ctx3_lirf_predict.py --declaration configs/lirf_dark_div_design_v1.json --fold seasonal`, 8 threads, 2.7 GB); against each
  run's own GPU October predictions 0.11–0.12 s RMS (TF32 noise, as in the masked-member and stronger-masking rounds).

## Checks

The 16-seed lm25_32 mean equals v54's member on both dark sets exactly (max 0.0 s); every natural month's `lirf_net` equals the shipped
network's fold mean (max 1.6e-13 s). A code-path test before the declaration (lm25_32 standing in for the candidate) read exactly 0.0.

## Result (primary: route rows with labels below 20,000 s, v54's dark arithmetic, against v54's 16-seed member)

| reading | January 2025 (11,254 rows) | thinned July (14,771 rows) | standalone RMSE Jan / thinned (M: 274.64 / 430.09 s) |
|---|---|---|---|
| **mix** (0.5 M + 0.5 lm25dl32) | −0.429% [−0.91, +0.08], 22/31 days | **+0.098%** [−0.27, +0.53] | 273.25 / 431.22 |
| lm25dl32 alone (reported) | −0.429% [−1.34, +0.53] | +0.550% [−0.23, +1.49] | 273.48 / 435.72 |
| 8 → 16 seeds of lm25_32 (reported) | −0.103% [−0.55, +0.31] | −0.616% [−0.97, −0.29] | |

**No GO.** The mix misses January's bar (−0.574%) and is worse on thinned July (rule: at most 0). lm25dl32 is a little better than lm25_32
on January 2025 standalone (−1.2 s) and clearly worse on thinned July (+5.6 s); its errors correlate 0.988 / 0.984 with M's, so the
diversity the dl settings gave outside LIRF is small here. More seeds of the same masked network read better on thinned July than the dl
mix does (descriptive). The dl diversity member is closed for LIRF; lm25_32 stays v54 / v55's member.
