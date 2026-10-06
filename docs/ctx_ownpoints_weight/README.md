# Own-points member weight round (6 October 2026): nothing selected, v65 stays

Lever 1 of the final-week campaign (the user: "yes start on them now"). Declaration
[`configs/ctx_ownpoints_weight_v1.json`](../../configs/ctx_ownpoints_weight_v1.json) (bdd0087, 05:16:47 UTC, before any score of a
weight above 0.40). Tool `tools/run_ownpoints_weight.py`, which imports the v65 round's arithmetic unchanged
(`tools/run_ctx_ownpoints_dev.py`, `tools/run_ctx_ownpoints_chain.py`, `tools/export_ctx_ownpoints.py`). No official score and no
leaderboard enter; nothing is built or uploaded.

## Question

v65's member (own_all: dl32own on ADS-B-matched rows, m25dl32own elsewhere) replaced the combination member at weight 0.40, the top of
its declared grid. Its development record (`docs/ctx_ownpoints_dev/results_v1.json`, written before the upload) showed an alignment of
2.1-2.4, so the development months still gained from more weight. Candidates: weights 0.55 / 0.70 / 0.85 / 1.00 with the same 16 seeds and
fold tables (no new fit). Incumbent: own_all_w40.

## Stage 1: member level against own_all_w40 ([`results_stage1.json`](results_stage1.json))

Controls reproduced before any candidate was read: the v46 stack, v51's model part (combination w40) and own_all_w40 equal their recorded
RMSEs to 1e-9 s in all four months.

| candidate | January [95% s] | July | November | December | equal-footing mean | eligible |
|---|---|---|---|---|---|---|
| own_all_w55 | -0.047% [-0.35, -0.04] | -0.097% | -0.082% | -0.127% | -0.090% | yes |
| own_all_w70 | -0.087% [-0.59, -0.12] | -0.174% | -0.129% | -0.214% | -0.158% | yes |
| own_all_w85 | -0.107% [-0.74, -0.14] | -0.221% | -0.148% | -0.232% | -0.187% | yes |
| own_all_w100 | -0.113% [-0.80, -0.13] | -0.244% | -0.120% [-0.44, +0.00] | -0.231% | -0.196% | yes |

Alignment of the change falls from 2.3-3.8 at 0.55 to 0.7-1.1 at 1.00. Stage-2 order (by mean ratio): w100, w85, w70, w55.

## Stage 2: v65's chain against the v65 development proxy ([`results_stage2.json`](results_stage2.json))

Identities first (`runs/ctx_ownpoints_weight/stage2_identity.json`): the model-part step at 0.40 reproduces the v65 round's `dmp.parquet`,
and the scorer on the own_all_w40 chain files reproduces the v65 development proxy, both with max |diff| 0.0. The chain refits
stack_resid_v3 on the rebuilt components for every candidate.

| candidate | January [95% %] | July [95% %] | November [95% %] | December [95% %] | equal-footing mean | clause 1 |
|---|---|---|---|---|---|---|
| own_all_w100 | -0.068 [-0.24, -0.02] | -0.163 [-0.25, -0.08] | **+0.029** [-0.05, +0.11] | -0.133 [-0.24, -0.02] | -0.121% | fails |
| own_all_w85 | -0.061 [-0.22, -0.02] | -0.135 [-0.20, -0.07] | **+0.009** [-0.06, +0.08] | -0.082 [-0.14, -0.02] | -0.093% | fails |
| own_all_w70 | -0.047 [-0.17, -0.02] | -0.105 [-0.16, -0.06] | **+0.022** [-0.03, +0.08] | -0.071 [-0.11, -0.03] | -0.074% | fails |
| own_all_w55 | -0.024 [-0.09, -0.01] | -0.059 [-0.10, -0.03] | **+0.011** [-0.02, +0.04] | -0.037 [-0.06, -0.01] | -0.040% | fails |

Every candidate passes clauses 2 and 3 (equal-footing mean at most -0.02%, all three intervals below zero) but fails clause 1: November
(supplementary month, still covered by clause 1) is worse than the v65 proxy by more than 0.005% at every weight. **Nothing is selected;
v65 stays.** No ranking file, dry run or v66 was built.

## Reading

- About 40-60% of the member-level gain survives the chain in January, July and December; in November the refitted stacker turns a
  member-level gain of 0.08-0.15% into a small loss. The November change is not monotone in the weight (+0.011 / +0.022 / +0.009 /
  +0.029%), so part of it is refit noise of the stacker bag, but every weight reads above the 0.005% tolerance.
- The rule is not re-read and no weight outside the grid is tried; the lever is closed on this record.

## Files

- `runs/ctx_ownpoints_weight/`: `stage1.json`, `stage2_identity.json`, `selection.json`, `chain/<candidate>/` (model part, chain tables,
  `dev_v66_proxy.parquet`), logs.
- The imported chain step writes its daily-totals runs to `runs/daily_totals/ctx_ownpts_own_all_w{55,70,85,100}_{ctl,cand}/`, as the v65
  round did for w40.
