# LIRF masked member, development round (1 October 2026): `w55` selected by the declared rule

Declaration [`configs/adsb_ctx3_lirf_mask_dev_v1.json`](../../configs/adsb_ctx3_lirf_mask_dev_v1.json) (13:15:57 UTC, 540bbf0, before any
fit; hardware amendment 404aadf before any fit), network [`configs/adsb_ctx3_lirf_mask_network_v1.json`](../../configs/adsb_ctx3_lirf_mask_network_v1.json)
(the LIRF design's `lm25_32`). Scorer `tools/run_lirf_member.py` (ababae1, written before any member fit finished). Report
[`results_v1.json`](results_v1.json). Nothing is exported or uploaded by this round.

## Question

v52's masked member is routed to unmatched rows of every airport but LIRF. LIRF's unmatched rows are predicted by v29's round-1 LIRF
network, and in July 2026 every LIRF row is unmatched (adsb.lol has no Fiumicino ground coverage). The LIRF design
([`docs/adsb_ctx3_lirf`](../adsb_ctx3_lirf/README.md)) found masking helpful for LIRF's unmatched rows, most where LIRF is dark. Does the
masked LIRF network, blended into the shipped one on LIRF's unmatched non-population rows, improve the shipped LIRF arithmetic?

## Fits

- 16 seeds (801–816) × six folds: `seasonal` (for the thinned-July reading), `lomo_2025-01`, `lomo_2025-07`, `refit` (December),
  `forward` (November) on four AWS g6e.2xlarge (one L40S each, two lanes) — the user chose "GPU, 16 seeds"; `refit12` (ranking) on the
  Azure CPUs arcw0–7 (hardware amendment: AWS had no g6e capacity in us-east-2 for three nodes at launch).
- GPU queues 13:32–16:28 UTC; 80 of 80 fits, mirrored (`tools/combo_nodes.sh mirror` plus per-node `rsync`) and SHA-256-checked
  file by file against every node (4 × 132 files); all four nodes terminated after their mirror (the jobs controller untouched).
- **Incident:** at about 13:41 UTC the head node's copy of one timeline-bundle file (`features/training_2025-01-01_2025-02-01.parquet`)
  was overwritten by a truncated copy (31.4 MB against 50.0 MB; probably a SkyPilot file-mount re-sync). The bundle loader's size check
  failed the next 18 job starts on that node within seconds (no fit trained on the bad file; the two fits that had finished had loaded
  the bundle at 13:32). The file was re-uploaded from the workstation, every bundle file on all four nodes was hash-checked against the
  workstation (13 of 13 identical on each), and the node's queue was restarted at 13:53 UTC; it skipped the finished fits. The 18
  failed starts' logs are kept in `runs/adsb_ctx3_lirf_mask/node_logs/prc-ctx3a_failed_1345/`.
- Thinned July: the 16 seasonal models re-predicted on coverage-regime attempt a13 on the workstation CPU
  (`tools/ctx3_lirf_predict.py --fold seasonal`). Against each run's own October predictions the CPU re-prediction differs by 0.11–0.14 s
  RMS (the predictor's 0.05 s check was set for CPU-trained models; these were trained on the L40S with TF32 matrix products, and the
  26 September GPU-to-CPU readings were 0.09–0.31 s per network). Natural July: 0.15 s RMS on predictions of about 1,260 s, at most 2.4 s.
  The shipped network's thinned July values were also CPU re-predictions, so both sides of the comparison share the footing.

## Control check

Before any candidate was read: in all four natural months the rows table's `lirf_net` equals the shipped network's 16-seed fold mean
(RMS 0.0 s), and the control (v34's unmatched blend with v45's gate) equals the v46 development stack on every route row (max 0.0 s).

## Result (declared rule; primary = squared error of route rows with labels below 20,000 s)

| set | rows | w25 | w40 | **w55** |
|---|---:|---|---|---|
| **thinned July (LIRF dark)** | 14,771 | −0.351% [−0.55, −0.16] | −0.510% [−0.83, −0.20] | **−0.627% [−1.07, −0.20]** |
| January 2025 (LIRF naturally dark) | 11,254 | −0.122% [−0.60, +0.38] | −0.222% [−0.79, +0.34] | −0.411% [−1.12, +0.25] |
| July 2025 (natural) | 864 | −0.141% | −0.150% | −0.104% [−1.06, +0.97] |
| November 2025 | 3,871 | +0.151% | +0.297% | +0.484% [−0.33, +1.40] |
| December 2025 | 2,474 | −0.422% | −0.510% | −0.657% [−1.98, +0.67] |
| eligible | | no (thinned July above −0.5%) | yes | **yes, selected** (lowest thinned July + January) |

Relative change of the route's squared error; paired-day 95% interval (10,000 day draws). Unfiltered readings are identical (no route
label above 20,000 s changes the sums). w55 moves route RMSE 416.9 → 415.6 s on thinned July and 271.9 → 271.4 s on January.

- Whole-month effect on the v46 development stack (every other row unchanged): January −0.011%, July −0.001%, November +0.013%,
  December −0.019%. The 2025 months have few unmatched LIRF rows; the July 2026 ranking month has all of its ~15,500 LIRF
  non-population rows on the route, the regime thinned July simulates.
- The gate (tree − network > 600 s) fires on about the same rows (thinned July 96 → 91).
- Expectation for the ranking file, transferred at face value: about −1.8e7 squared error (thinned July's change on July 2026's LIRF
  rows plus January's on January 2026's unmatched LIRF rows), about −0.1 s on 222 s. Ranking-regime LIRF layers have transferred at
  0.5–2× their development reading (v29, v45). An estimate, not a prediction of an official score.

## Next (per the declaration's export plan)

The ranking member needs the 16 `refit12` fits (Azure CPUs, running). Then: v52's model part with the route's ranking rows recomputed
(network' = 0.45 ens_ctx_lirf + 0.55 M inside v34's blend and v45's gate), v48's reconciliation with the published daily totals applied
last, an audit, and a check that every other row equals v52's model part. Export and upload only at the user's word.
