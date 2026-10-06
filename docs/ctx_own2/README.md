# Lever 5: a stronger / more diverse own-points context-network member on top of v66 (6 October 2026, fork E)

Final-week campaign, lever 5. The user (6 October, ~09:30 UTC): "use the gpu now if you think it will help us, i have 4 aws gpu ready i think
not sure please check". No GPU instance existed; the G/VT quotas allowed on-demand L40S in us-east-1 / us-west-2 (us-east-2 is fork B's).
No official score, no leaderboard and no ranking label enters any step; nothing is uploaded.

## Question

v66's network table is the own-points member own_all alone (dl32own on ADS-B-matched rows, m25dl32own on the others, weight 1.00): the older
context pool carries no weight any more. Both networks use the "dl" recipe (numeric embeddings 16 wide, learning rate 0.002). The
combination design (`docs/adsb_ctx3_combo/README.md`, 29 September) found that recipe weaker than the reference recipe standalone and at
member weight b = 1 (route_m25 +0.190 points against combo_all +0.123; dl worse on matched and on masked unmatched rows); its gain at b = 0.5
came from being different from the old pool, which v66 no longer carries. So: does the reference recipe with the same 15 own columns
(r32own on matched rows, m25r32own on unmatched rows) replace own_all better, or add to it as a 50/50 recipe mixture?

## Design batch (declared before any fit)

[`configs/ctx_own2_design_v1.json`](../../configs/ctx_own2_design_v1.json) (64565c2, 09:32 UTC). October design fold (seasonal fit partition,
LIRF excluded, October monitored and predicted), 8 seeds (9-16) of each of dl32own, m25dl32own (control, refitted on the same GPUs), r32own,
m25r32own: 32 fits. Items: control (own_all), ref_own and mix_own (decision items), ref_matched / ref_unmatched (reported). GO per decision
item: recipe proxy at b = 1.0 at least 0.05 points below the control's, standalone below the control's, paired-day interval below zero.
Report: `tools/ctx_own2_design_report.py` -> `design_report.json`.

## Hardware and GPU record

- Authorisation [`infra/aws/gpu_authorization_20261006b.json`](../../infra/aws/gpu_authorization_20261006b.json) (43dc8c8) and amendment 1
  (a35e9a9, coordinator ~10:05 UTC at the user's request): on-demand g6e.2xlarge (L40S) in us-east-1 / us-west-2 (prc-ctx4a-d) and spot
  g6e.2xlarge in the same regions (prc-ctx4e-h); us-east-2 spot only after fork B's nodes are up or B reports NO GO.
- Tools: `tools/ctx_own2_nodes.sh` (launch with capacity retry, fanout, checks), `tools/ctx_own2_dispatch.py` (one fit per `sky exec`, two
  per GPU, every finished fit mirrored at once and hash-checked, fits of a pre-empted node re-run from scratch; partial fits never counted),
  `tools/ctx_own2_run1.sh`, `tools/ctx_own2_jobs.py`, `tools/ctx_own2_check_features.py` (own-points tables against tunnel 21's manifest).
- Capacity: on-demand g6e.2xlarge launches failed with InsufficientInstanceCapacity in every zone of us-east-1 and us-west-2 from 09:34 UTC;
  spot attempts from 10:01 UTC likewise (SkyPilot also tried g6e.4x/8x/16xlarge spot; the spot yamls were then pinned to g6e.2xlarge).

## Pre-emptions

(recorded from `runs/ctx_own2/dispatch_<stage>.log`)

## Results

(pending)
