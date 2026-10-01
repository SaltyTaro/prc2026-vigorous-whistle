# Stronger LIRF masking, design (1 October 2026, after v54): no GO — closed

Declaration [`configs/lirf_mask2_design_v1.json`](../../configs/lirf_mask2_design_v1.json) and network
[`configs/lirf_mask2_network_v1.json`](../../configs/lirf_mask2_network_v1.json) (18:43:33 UTC, 72f7874, before any fit), scorer
`tools/lirf_mask2_report.py` (same commit), report [`design_report.json`](design_report.json). Nothing is exported or uploaded.

## Question

v53/v54's LIRF member is the masked LIRF context network at masking probability 0.25 (`lm25_32`: a matched row is shown as unmatched,
its 21 focal ADS-B columns blanked, with probability 0.25 in training). July 2026 has no LIRF ADS-B at all. Does stronger masking, 0.5
(`lm50_32`) or 0.75 (`lm75_32`), give a better network for that month, read inside v54's dark-month arithmetic (network' = 0.30 shipped +
0.70 M, blend 0.75 / 0.25 tree, gate u600)?

## Fits

- 2 variants × 8 seeds (901–908) × 2 folds (`seasonal` for thinned July, `lomo_2025-01` for January 2025) = 32 fits on four AWS
  g6e.2xlarge (one L40S each, two lanes), `infra/aws/gpu_authorization_20261001d.json` (the user: "yes, launch item 3 on GPU").
  Queues 19:01–20:05 UTC, 32 of 32, none failed; the quality monitor flagged no fit (seasonal 12 min, January fold 17 min per fit).
- Before the launch every node's token build and context tests passed and all 12 timeline-bundle files matched the workstation's
  hashes. The seasonal fits were mirrored at 19:33 and the rest after each queue ended; every file was SHA-256-checked against its node
  (4 × 52); each node was terminated right after its mirror (a 20:05, b/c/d 20:07; the jobs controller untouched).
- Thinned July: the 16 seasonal models re-predicted on coverage-regime attempt a13 on the workstation CPU; against each run's own GPU
  October predictions 0.11–0.14 s RMS (TF32 noise, as in the masked-member round).

## Checks

The 16-seed lm25_32 mean reproduces v54's member on both dark sets exactly (max 0.0 s); every natural month's `lirf_net` equals the
shipped network's fold mean. Seed-split null (lm25_32 seeds 809–816 against 801–808, same code): January 2025 +0.287% [−0.62, +1.14],
thinned July −0.310% [−0.79, +0.17], so the declared bar was B = max(0.25%, 2 × 0.287%) = **0.573%** on January. (A code-path test with
lm25_32 stand-ins showed this null before any fit finished; it does not depend on the candidates.)

## Result (primary: route rows with labels below 20,000 s, v54's dark arithmetic)

| variant | January 2025 vs 8-seed control | thinned July vs 8-seed control | vs v54's 16-seed member (Jan / thinned) | standalone RMSE Jan / thinned (control 275.46 / 432.31) | GO |
|---|---|---|---|---|---|
| lm50_32 | −0.067% [−1.04, +0.90] | −0.563% [−1.33, +0.17] | +0.035% / +0.053% | 274.39 / 432.56 | no |
| lm75_32 | −0.213% [−1.13, +0.75] | +0.165% [−0.45, +0.74] | −0.111% / **+0.786% [+0.24, +1.31]** | 274.95 / 435.90 | no |

**Neither variant reaches January's bar; stronger masking is closed for LIRF.** Every reading is inside the seed-split null's range.
Stronger masking makes the network slightly better on January 2025 standalone (−1.1 / −0.5 s) and worse on thinned July (+0.3 / +3.6 s);
inside the stack it is flat at 0.5 and harmful at 0.75 on thinned July (interval above zero against v54's member). The masked LIRF network
at 0.25 stays the member; the remaining LIRF-side design idea is the combination member (dl32 on matched LIRF rows, m25dl32 on unmatched).
