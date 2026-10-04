# Dig tunnel 21: own-aircraft turnaround features inside the context network (design batch, 3 October 2026)

Declaration [`configs/ctx_ownpoints_design_v1.json`](../../configs/ctx_ownpoints_design_v1.json) (b8be18f, 17:41 UTC, before any fit).
October design fold only: no scoring month, no ranking row and no official score is read. Nothing is exported or uploaded.

## Question

Tunnel 18 ([`pregap.md`](pregap.md)) fed the aircraft's own turnaround ADS-B points (including those before the matcher's last coverage
gap) to the guarded stacker: stack_resid_v3, equal-footing −0.16%, built as v63 (not uploaded). Does the strongest model member, the
context network (shipped member: dl32 on ADS-B-matched rows, m25dl32 on the others, `configs/adsb_ctx3_combo_dev_v1.json`, weight 0.40),
gain more when the same information is a network input?

## Features

`tools/build_ownpoints_features.py` builds tunnel 18's 14 raw columns plus `p_has_inb` for every 2025 month and the ranking file, at all
ten airports, into `data/external/adsblol/features_v19own/features_v19_<month>.parquet` (the 11 v19 columns unchanged, 15 own columns
appended; manifest with hashes). Departure labels and departure block times are never read (departure BLOCK_TIME is blanked on load).

## Design

16 CPU fits on arcw0-7 (dl32 and m25dl32, each with and without the 15 own columns, seeds 9-12, October design fold), finished
20:57-21:45 UTC on 3 October; mirrored and hash-checked (`tools/ctx_ownpoints_mirror.sh`), all eight VMs deallocated at 01:15 UTC on
4 October (they idled about 3.5 h after the fits because the agent's monitor did not resume it). Controls reproduce the saved ctx3 design
runs exactly (max difference 0.0). Report `tools/ctx_ownpoints_report.py` -> [`ctx_ownpoints_design_report.json`](ctx_ownpoints_design_report.json).

| member (routing as shipped: dl32 matched / m25dl32 unmatched) | October RMSE (matched / unmatched) | stack at weight 0.40 | gain vs control member | paired-day interval (s) |
|---|---|---|---|---|
| control (shipped settings) | 156.383 (118.34 / 210.55) | -0.0731% | | |
| own columns in both networks | 155.525 (116.41 / 210.75) | -0.2442% | +0.171 points (+0.332 at weight 1) | [-0.46, -0.16] |
| own columns on the matched network only | 155.430 (116.41 / 210.55) | -0.2077% | +0.135 points | [-0.30, -0.18] |
| own columns on the unmatched network only | 156.477 (118.34 / 210.75) | -0.1097% | +0.037 points | [-0.19, +0.05] |

Rows with an identified inbound gain most (-1.56% standalone). **Design verdict: GO** (bar -0.05%), on one design month and before
stacker v3, which already uses the same information; the overlap with v63's stacker is unknown until a development round re-fits the
stacker on the new member. A full round = the ctx3 development protocol (`configs/adsb_ctx3_combo_dev_v1.json`: 16 seeds x the
development folds, about 160 fits) with the own columns on the matched network, then the stacker refit and the v63 chain rebuild.
