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

Design batch: none. No spot node was ever provisioned (spot g6e.2xlarge unavailable in us-east-1 / us-west-2 throughout 10:01-11:30).
The 32 fits ran on on-demand prc-ctx4b (us-east-1, from 10:31) and prc-ctx4c (us-west-2, from 11:05); prc-ctx4i / prc-ctx4j (us-east-2,
on-demand, up 10:43) were never prepared by the dispatcher and autostopped idle (see the hand-off). Every fit was mirrored the moment its
run.json existed, with all six files' SHA-256 equal to the node's (`runs/ctx_own2/dispatch_design.log`); no fit failed or was re-run.

## Design result (13:02 UTC, [`design_report.json`](design_report.json)): **GO for `mix_own`**

October non-LIRF RMSE (8-seed means), recipe proxy R2 at v66's member weight b = 1.0 (share 0.75), gain in points against the control:

| item | standalone (s) | matched / unmatched (s) | vs control | gain at b = 1.0 (pts) | interval (s) | GO |
|---|---:|---|---:|---:|---|---|
| control (own_all as shipped, GPU refit) | 154.123 | 115.71 / 208.48 | | | | |
| ref_own (r32own / m25r32own) | 153.271 | 114.46 / 207.97 | -0.553% | +0.126 | [-0.50, +0.11] | no (interval) |
| **mix_own (0.5 control + 0.5 ref_own)** | **152.738** | 114.37 / 206.93 | **-0.898%** | **+0.184** | **[-0.45, -0.18]** | **yes** |
| ref_matched (reported) | 153.509 | 114.46 / 208.48 | -0.398% | +0.088 | [-0.25, -0.07] | |
| ref_unmatched (reported) | 153.885 | 115.71 / 207.97 | -0.154% | +0.038 | [-0.33, +0.28] | |

- The reference recipe is better than dl on both routes (r32own 114.46 vs dl32own 115.71 on matched rows; m25r32own 207.97 vs m25dl32own
  208.48 on unmatched rows), as the 29 September record suggested; the 50/50 recipe mixture is better still (errors correlate 0.975).
- GPU controls against tunnel 21's CPU fits of seeds 9-12: per-seed October RMSE within 1.5 s either way (not exact across devices); the
  4-seed means agree within the seed spread.
- Design fits: 32 on L40S, about 10-12 min each at two per GPU.

## Development round ([`configs/ctx_own2_dev_v1.json`](../../configs/ctx_own2_dev_v1.json), c843eeb, declared 13:04 UTC before any fit)

160 fits (r32own / m25r32own, seeds 621-636, folds lomo_2025-01, lomo_2025-07, forward, refit, refit12) on on-demand L40S prc-ctx4c
(us-west-2) and prc-ctx4i / j / k / l (us-east-2), 13:09-17:13 UTC; every fit mirrored with equal SHA-256; no pre-emption. The first two
attempts on prc-ctx4c failed before fitting because the new network config had not reached the node (fixed: the dispatcher syncs code and
configs before a node's first fit). ref_own tables: `tools/build_combo_member.py --declaration configs/ctx_own2_dev_v1.json`
([`member_build.json`](member_build.json)); mix_own: `tools/run_ctx_own2_dev.py --build-mix` (own_all and ref_own finite on the same rows).

### Stage 1 (member level, [`results_stage1.json`](results_stage1.json)): eligible

Control own_all at 1.00 reproduces docs/ctx_ownpoints_weight/results_stage1.json exactly (and v46 / v51's model part their records).

| candidate | January | July | November | December | mean (Jan/Jul/Dec) | eligible |
|---|---|---|---|---|---|---|
| mix_own_w100 | -0.056% [-0.449, -0.025] s | -0.108% [-0.449, -0.097] | -0.165% [-0.491, -0.145] | -0.233% [-0.660, -0.170] | -0.132% | yes |

### Stage 2 (governing; v66's chain, variance re-allocation pinned; [`results_v1.json`](results_v1.json)): **NOT accepted**

| month | pinned control (own_all at 1.00) | mix_own_w100 | change | paired-day 95% (%) |
|---|---|---|---|---|
| January 2025 | 320.1258 | 320.0675 | -0.018% | [-0.093, +0.009] |
| July 2025 | 251.0731 | 251.0525 | -0.008% | [-0.046, +0.029] |
| November 2025 | 182.1762 | 182.0592 | -0.064% | [-0.134, -0.008] |
| December 2025 | 163.7131 | 163.5360 | -0.108% | [-0.262, +0.016] |

v57's clauses: (1) no month worse by more than 0.005%: yes; (2) equal-footing mean -0.045% (bar -0.02%): yes; (3) every equal-footing
interval below zero: **no** (January, July and December cross zero). **Nothing is selected; no dry run is built; v66 stays.** The refitted
stacker (stack_resid_v3, which already reads the own-aircraft points) absorbs most of the member-level gain: -0.132% at stage 1 becomes
-0.045% in the chain. The pinned control differs from the stored unpinned v66 proxy by +0.0002 / -0.0018 / -0.0090 / +0.0241% (Jan / Jul /
Nov / Dec), the drift fork B measured; only pinned control and candidate are compared.

## GPU use

On-demand L40S only (spot never provisioned). Every prc-ctx4 cluster was torn
down at 17:13 UTC after the last hash-checked mirror; the billing guard exited at 17:25 (no cluster left).
