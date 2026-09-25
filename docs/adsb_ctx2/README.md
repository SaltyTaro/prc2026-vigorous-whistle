# Context network round 2 (adsb_ctx2) on the AWS L40S, 25 September 2026

**Question.** The round-1 context network ([`docs/adsb_ctx/README.md`](../adsb_ctx/README.md)) passed its design reading at
−1.19% on the recipe proxy, and its CPU development round runs unchanged on Azure. Round 2 asks whether a wider neighbour window,
self-attention among the neighbours, a wider token encoder or a longer budget beats the round-1 settings. It runs on the
user-authorized L40S ([`infra/aws/gpu_authorization_20260925.json`](../../infra/aws/gpu_authorization_20260925.json)), October
only, before any development score of a round-2 network.

## Declaration — [`configs/adsb_ctx2_design_v1.json`](../../configs/adsb_ctx2_design_v1.json), commit `2879e56`

- Declared 05:51 UTC. Declaration SHA-256 `3efba0515870946b02890c685dae3dc0b6c7f52c87ddf93c2f456c11d8b314cb`, checked by the
  report against every run.
- Package `src/prc2026_ctx2/`. The round-1 package `src/prc2026_ctx/` is untouched. What round 2 adds:
  - device-aware training, float32 with TF32 matrix products;
  - a configurable neighbour window (`tools/build_ctx2_tokens.py`), which reproduces the round-1 tokens exactly with the round-1
    window (test);
  - optional self-attention layers among the neighbour tokens (`sa_layers`; with 0 layers the network is identical to round 1,
    test).
- Fold: the round-1 design fold exactly. Seasonal fit partition 2025-02..06, 08, 09 without LIRF (1,130,307 rows), October
  monitored, October predictions only.
- Variants (all others use the round-1 settings: 16 epochs, batch 2048, 128 tokens, token width 96, 4 heads):

| variant | change | seeds |
|---|---|---:|
| g_ctx | none (reference) | 3 |
| g_w256 | 256 tokens from a [t0 − 120, t0 + 60] min window (`data/ctx/tokens_w256`) | 3 |
| g_sa2 | 2 self-attention layers among the tokens | 3 |
| g_sa2_w256 | both | 3 |
| g_big | token width 192, 8 heads | 2 |
| g_e24 | 24 epochs | 2 |

- **Decision rule (declared):** a variant replaces the round-1 settings if its seed mean beats g_ctx's at equal seed count by
  at least 0.10 percentage points of recipe proxy at b = 1.0, AND it is better standalone. If several qualify, the lowest R2
  wins. Any next development round is declared separately, after the CPU round's selection is known.

## Hardware and runs

- - Cluster file [`infra/aws/sky_ctx2.yaml`](../../infra/aws/sky_ctx2.yaml): explicit file mounts only, `.env` never
    uploaded.
  - Setup checks: pytest, both token builds, and content-hash assertions against the local token manifests.
- Setup issues, fixed before any declared run (commit `3afa4c9`): missing receipt packages (psutil, tzdata) and missing mounts
  (requirements files, `docs/`, IEM and OurAirports sources). Sky job 2 was submitted without `--gpus` and cancelled at the
  first CUDA check. No declared run is affected.
- **Job 3** ran `tools/ctx2_design_chain.sh` with 2 lanes, 06:12–07:39 UTC. The chain log is [`chain.log`](chain.log).
- **Deviation: g_sa2_w256 out of memory.** All three g_sa2_w256 seeds failed on the first epoch's monitor prediction.
  - Cause: that prediction runs in batches of 8,192 (training uses 2,048). With 256 tokens and self-attention, one attention
    score tensor is 8,192 × 4 × 256 × 256 × 4 bytes = 8 GiB, and the other lane held 17 GB of the 44 GB card. Training itself
    fit.
  - The failed logs are preserved in [`failed_2lanes/`](failed_2lanes/). No output of these attempts was written or read.
  - **Job 4** reran exactly those three runs alone on the GPU, 07:39–08:57 UTC, with
    [`failed_2lanes/rerun_1lane.sh`](failed_2lanes/rerun_1lane.sh): the same chain with 1 lane and
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (an allocator setting only).
  - Code, configuration and seeds are unchanged; no variant's settings changed.
- Fit times per run (two lanes sharing the GPU, except g_sa2_w256): g_ctx 330 s, g_e24 440 s, g_big 535 s, g_w256 410–580 s,
  g_sa2 1,360–1,520 s, g_sa2_w256 1,520 s alone. The same network on one Azure D16ds_v7 takes about an hour.

## Readings — [`design_report_1.json`](design_report_1.json) (`tools/ctx2_design_report.py`)

**Hardware check** (descriptive). GPU g_ctx against the round-1 CPU ctx seeds, October non-LIRF RMSE at equal seed counts:

| seeds | 1 | 2 | 3 |
|---|---:|---:|---:|
| GPU g_ctx | 161.860 | 158.425 | 156.870 |
| CPU round-1 ctx | 163.558 | 158.891 | 157.114 |

Single runs differ, since the GPU and CPU compute in a different order. The seed means agree within the seed noise, so the
L40S is a valid substitute for the CPU fits.

**R1 and R2**:
- R1: standalone October non-LIRF RMSE, s.
- R2: recipe proxy on 183,761 October rows, table = 0.5 committee + 0.5 [(1 − b) nnb32 + b variant]. b = 1.0 was the best b
  for every variant.

| variant | seeds | R1 seed mean | R2 b = 1.0 | vs g_ctx at equal seeds: R1 | R2 gain (pts) | qualifies |
|---|---:|---:|---:|---|---:|---|
| g_ctx | 3 | 156.870 | −1.1105% | — | — | — |
| g_w256 | 3 | 156.632 | −1.0880% | 156.632 vs 156.870 | −0.0225 | no |
| g_sa2 | 3 | 157.241 | −1.0074% | 157.241 vs 156.870 | −0.1031 | no |
| g_sa2_w256 | 3 | 156.777 | −1.0312% | 156.777 vs 156.870 | −0.0793 | no |
| g_big | 2 | 158.094 | −1.0312% | 158.094 vs 158.425 | +0.0228 | no |
| **g_e24** | 2 | 158.281 | −1.1621% | 158.281 vs 158.425 | **+0.1537** | **yes** |

- Paired-day intervals at b = 1.0 are all below zero:
  - g_ctx [−2.73, −1.39] s;
  - g_e24 [−2.83, −1.44] s;
  - g_w256 [−2.66, −1.37] s;
  - g_big [−2.66, −1.16] s;
  - g_sa2_w256 [−2.53, −1.31] s;
  - g_sa2 [−2.50, −1.24] s.
- Matched and unmatched non-LIRF rows at b = 1.0:
  - g_ctx −0.32% / −2.32%;
  - g_e24 −0.60% / −2.28%.
  g_e24's difference sits on the matched rows.

**Decision (declared rule): g_e24 is chosen.** It is the only variant that qualifies.

- **What the rule does not show:** g_e24 clears the 0.10-point bar by 0.05 points on two seeds. Each of its single runs is
  worse standalone than g_ctx's (162.95 / 162.81 against 161.86 / 162.26); only the two-seed mean is better. A GPU
  development round at 24 epochs would test it on complete months. This design reading does not establish that 24 epochs are
  better.
- **What round 2 does settle:**
  - The round-1 architecture is not improved by a wider window, a wider token encoder, or self-attention among the neighbours.
  - Self-attention costs about 0.08–0.10 points and 4–5× the fit time.
  - The network's gain comes from the individual neighbours the focal query attends to, not from interactions among them.

## Next

Per the declaration, any GPU development round, with g_e24 settings on seasonal/forward/refit and refit12 folds, is declared
separately after the CPU round's (`configs/adsb_ctx_v1.json`) selection is known. The CPU round and round-2 networks are never
mixed. Seed scaling is the other open question: the round-1 design gain was still rising at six seeds. The L40S fits a
round-1 network about 11× faster than a D16ds_v7.
