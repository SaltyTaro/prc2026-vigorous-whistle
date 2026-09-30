# Masked context network on unmatched rows, on top of v51 (30 September 2026)

The design batch of the context-network combination (`docs/adsb_ctx3_combo/README.md`) passed two items. The full combination went to
the first development round and became v51 (222.2306 official). This round tests the other item: the masked network (`m25_32`, ADS-B
inputs of matched rows hidden with probability 0.25 during training) used only on rows without an ADS-B match, as a second member added
on top of v51's network table. The user: "go ahead with the masked network round".

## Declaration (05:28 UTC, before any fit; commit 22be51d)

- `configs/adsb_ctx3_mask_network_v1.json`: `m25_32` with the design batch's settings unchanged (32 epochs, numeric embeddings 8 wide,
  learning rate 0.001, masking 0.25).
- `configs/adsb_ctx3_mask_dev_v1.json`: member `mask_unmatched` = the seed mean of 16 new seeds (701–716) on rows with
  `adsb_matched = 0`; rows with a match carry no member value and keep v51's table; LIRF rows keep the shipped table. Folds as in the
  combination round (leave-one-month-out January and July 2025, the forward fold for November, the January–November refit for December,
  the twelve-month refit for the ranking rows): 80 fits.
- Base member: the combination at weight 0.40 (v51's model part). The control is the v46 stack with it; the candidates add the masked
  member at 0.15 / 0.25 / 0.40 on top. Unchanged rule (no equal-footing month worse, mean ratio at least 0.02% below 1, every
  equal-footing month's paired-day interval below zero; lowest mean ratio among the eligible).
- Hardware: AWS L40S nodes (`infra/aws/gpu_authorization_20260930.json`), for turnaround (about 3–4 hours against about 29 hours on the
  eight Azure CPU workers).
- Disclosed: the design batch's readings, the combination round's readings and v51's official score were known; the item, the grid and
  the rule come from the earlier records; nothing is derived from the official score.

## Tools (commit 77aa1c2)

- `tools/run_network_member.py` and `tools/export_network_member.py` take one base member. Checks run before any fit of this round:
  the control with the combination at w40 reproduces the combination round's recorded RMSEs on January, July, November and December to
  1e-9; on the ranking rows, the masked round's declaration with the combination member itself at weight 0 exports v51's member file
  byte for byte (`bd8f73aa…`).
- `tools/combo_jobs.py --declaration`, `ROUND=runs/adsb_ctx3_mask` for `tools/combo_queue.sh` / `tools/combo_nodes.sh`, and
  `tools/build_combo_member.py --declaration` (the member's route from the declaration; the combination's tables rebuild byte for byte).

## Development round (fits 06:09–09:05 UTC on four L40S nodes; governing run 08:35 UTC)

80 fits, none failed, all 32 epochs, 14–16 minutes each at two lanes per GPU. The monitors flagged no fit.

**Governing run (`docs/adsb_ctx3_mask/results_v1.json`): the declared rule selects nothing.** January's paired-day interval includes
zero at every weight; every other month and weight improves with its interval below zero.

| candidate | January 2025 | July 2025 | December 2025 | mean (rule months) | November 2025 (reported only) |
|---|---:|---:|---:|---:|---:|
| control (v46 stack + combination w40 = v51's model part) | 322.6450 | 257.5356 | 167.3137 | | 184.8231 |
| `w15` | −0.007% [−0.065, +0.033] | −0.030% [−0.121, −0.040] | −0.034% [−0.095, −0.019] | −0.023% | −0.037% [−0.113, −0.035] |
| `w25` | −0.014% [−0.103, +0.009] | −0.041% [−0.162, −0.053] | −0.053% [−0.145, −0.038] | −0.036% | −0.065% [−0.220, −0.054] |
| `w40` | −0.017% [−0.134, +0.022] | −0.048% [−0.198, −0.045] | −0.082% [−0.221, −0.065] | −0.049% | −0.088% [−0.301, −0.070] |

- The member only changes unmatched non-LIRF rows (29–50% of non-LIRF rows); matched rows are untouched by construction.
- Standalone on its own rows the masked network beats v51's table in every month: 453.96 / 235.65 / 209.17 / 204.12 s against
  454.58 / 236.10 / 210.75 / 205.33 s (January / July / November / December).
- The gain is small because v51's table already holds a masked network on these rows (the combination's m25dl32 at 0.40).
- At `w40` the alignment of the change is 1.1 / 1.5 / 2.0 / 1.6 (January / July / November / December): the largest declared weight is
  still below the optimum. No weight was added after the scores.

**Disclosed interim readings:** January and July at 07:30 UTC, January, July and December at 08:00 UTC
(`runs/adsb_ctx3_mask/dev/interim_*.json`, `interim3_*.json`), with the same numbers as the governing run. After the first, I
recommended stopping after December; the user said "let it finish".

**The user's override (30 September): "ship it anyway"**, for `w40`, after the plain-language summary of the interim readings. The
override is recorded in the export's provenance. The user chose to upload v52 only, without an open-data backup file for now.

## Official result: v52 = 222.0519 (30 September 2026)

v52 = v51's model part + the masked member at w40 (the user's override), then v48's daily-totals reconciliation applied last.
Member file (dry run on v51's member file, override recorded): SHA-256 `16e9f33a…`, 91,409 rows differ from v51's member file,
squared change 1.56e7, largest 422 s; the control reproduces v51's member file on every row; LIRF rows unchanged
(`docs/adsb_ctx3_mask/member_export_dry_v52.json`). Reconciled file `runs/submissions/vigorous-whistle_v52.parquet`, SHA-256
`9b17130a8cbce4e76e0093805009f3313f72048e9c25880a057e579dcca000d2`, byte-identical to its dry run
(`docs/daily_totals/export_v52.json`); audit clean (`docs/export_audit/v52_audit.json`; RMS difference to v51 6.8 s, LIRF identical,
no row over 600 s). Uploaded at 09:09 UTC, processed at 09:09:30 UTC: **222.0519 on all 344,841 rows, 0.1787 s (0.080%) better than
v51 (222.2306)**, the official best. Record: `docs/adsb_ctx3_mask/official_result_v52.json`.
Nothing is derived from the score.
