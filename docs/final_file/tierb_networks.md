# Tier B networks: the 80 four-seed February / June 2025 LOMO fits (lane G, 10 October 2026)

Tools: `tools/febjun_net_jobs.py` (job lists), `tools/febjun_net_queue.sh` (per-node queue, 2 lanes per GPU),
`tools/febjun_net_nodes.sh` (up / fanout / check / sync / launch / status / mirror / down), `tools/febjun_net_mirror.py` (hash check).

## Result

- **80 of 80 fits complete, 0 failed**: ten ensembles x 4 seeds x {lomo_2025-02, lomo_2025-06} (ctx_e24 117-120 and 133-136, ctx_e32
  401-404, ctx_lirf 301-304, dl32 / m25dl32 601-604, m25_32 701-704, lm25_32 801-804, dl32own / m25dl32own 601-604 with
  `features_v19own`). No held-month RMSE was written (by construction for these folds).
- **Mirror:** `runs/febjun_nets/nets/`, 480 files SHA-256-checked against the nodes' own lists (`runs/febjun_nets/node_hashes/`), no file
  missing, no problem. The final mirror was taken at 13:36:57 UTC
  after all three queues had finished.

## Nodes and spend

| cluster | region | up (UTC) | queue | queue done (UTC) | down (autodown, 45 idle min) |
|---|---|---|---|---|---|
| prc-ctx8f | us-east-1a | 09:51 | 26 jobs | 13:20 | about 14:07 |
| prc-ctx8e | us-east-1b | 09:57 | 27 jobs | 13:25 | about 14:12 |
| prc-ctx8h | us-west-2b | 09:57 | 27 jobs | 13:28 | about 14:15 |

- All other launch attempts
  failed before an instance started and cost nothing.
- `sky status` at 15:02 UTC shows no prc-ctx8* cluster; the only cluster listed (sky-jobs-controller-ba899da3, stopped) belongs to another
  project and was not touched.
