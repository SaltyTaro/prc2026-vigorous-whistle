# Masked member seed extension (30 September – 1 October 2026): nothing selected, v52's 16-seed member stays

Declaration [`configs/adsb_ctx3_mask_seeds_v1.json`](../../configs/adsb_ctx3_mask_seeds_v1.json) (22:00:51 UTC 30 September, b8c6a8b,
before any fit of these seeds). GPU record [`infra/aws/gpu_authorization_20261001.json`](../../infra/aws/gpu_authorization_20261001.json)
(the user: "use gpu"). Report: [`results_v1.json`](results_v1.json) (= `runs/adsb_ctx3_mask2/dev/selection.json`). Nothing exported or
uploaded.

## Question

v52's masked member is the mean of 16 seeds (701–716) of the m25_32 context network on rows without an ADS-B match, added at weight 0.40.
Does a 32-seed mean (16 new seeds, 717–732, same network, same five folds) beat it at the unchanged weights (combination 0.40, masked 0.40)
under the unchanged rule (no equal-footing month worse, mean ratio at least 0.02% below 1, every paired-day 95% interval below zero)?

## Fits

- 80 fits (16 seeds × five folds: lomo_2025-07, lomo_2025-01, forward, refit, refit12) on four AWS g6e.2xlarge (one L40S each), two lanes
  per GPU, `tools/run_adsb_ctx_lomo.py --device cuda` through `tools/combo_queue.sh`.
- Queues started at about 22:18 UTC; the four development folds were complete at 00:37 UTC and refit12 at 01:09 UTC. No fit failed,
  and none stood out on its training curve (loss 0.096–0.105 by fold, 14–18 minutes per fit).
- Mirrored to `runs/adsb_ctx3_mask2/nets/` (512 files, every SHA-256 equal to the node's copy); node logs in
  `runs/adsb_ctx3_mask2/node_logs/`. The jobs controller was not touched.
- A first fan-out of the head's data stopped partway on the three workers (9 of 14 token files, no error printed); the second run
  completed, and every node's token hashes and context tests passed before any fit.
- Member tables of the new seeds: `tools/build_combo_member.py` (route: m25_32 on unmatched rows, matched rows without a member value):
  development months in [`member_build_dev.json`](member_build_dev.json), ranking rows (344,841, built after the result, not exported)
  in [`member_build_ranking.json`](member_build_ranking.json).
- Before any candidate was read, the v46 stack reproduced the v46 re-read and the control (v52's model part, 0.40 / 0.40 with the
  16-seed member) reproduced the masked round's recorded w40 on all four months (1e-9 s).

## Result

Change against the control (v52's model part), complete development months; paired-day 95% interval of the RMSE difference in seconds.

| candidate | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean (Jan / Jul / Dec) |
|---|---|---|---|---|---:|
| **mask32_w40** (32 seeds) | −0.0023% [−0.047, +0.019] | −0.0139% [−0.081, +0.010] | +0.0150% [−0.003, +0.056] | +0.0163% [−0.051, +0.096] | +0.0001% |
| mask_new16_w40 (new 16 only, reported) | −0.0053% [−0.125, +0.057] | −0.0139% [−0.126, +0.056] | +0.0431% [+0.021, +0.138] | +0.0618% [−0.033, +0.237] | +0.0142% |

Control RMSE: 322.5887 / 257.4131 / 184.6605 / 167.1763 s.

- **Nothing is selected.** The 32-seed member reads the same as the 16-seed one (mean +0.0001%); December and November are slightly worse,
  and no interval lies below zero.
- **The masked member itself holds up.** Read against v51's model part (the masked round's control), the member built from the new
  seeds alone improves every month, as the original seeds did:

  | against v51's model part | 2025-01 | 2025-07 | 2025-11 | 2025-12 |
  |---|---:|---:|---:|---:|
  | seeds 701–716 (v52, masked round) | −0.017% | −0.048% | −0.088% | −0.082% |
  | seeds 717–732 | −0.023% | −0.061% | −0.045% | −0.020% |
  | 32 seeds | −0.020% | −0.061% | −0.073% | −0.066% |

  The month-to-month spread between the two seed sets (up to 0.06 points in December) is seed noise of the 16-seed mean; the original
  set happened to read a little better on November and December. Averaging 32 seeds does not move the expected gain: the member is
  saturated at 16 seeds.

## Decision

v52's weights and its 16-seed masked member stay. No export is proposed: the 32-seed file would differ from v52 by tiny amounts with no
development gain. The seed-count lever for the masked member is closed.
