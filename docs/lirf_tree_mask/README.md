# Masked LIRF trees, design (2 October 2026, after v54)

Declaration [`configs/lirf_tree_mask_design_v1.json`](../../configs/lirf_tree_mask_design_v1.json) (03:24:47 UTC, e14664d, before any
fit), runner `tools/run_lirf_tree_mask.py`, report tool `tools/lirf_tree_mask_report.py` (same commit, written before any design prediction
was read), report [`design_report.json`](design_report.json). Nothing is
exported or uploaded.

## Question

July 2026 has no LIRF ADS-B at all (15,515 dark LIRF non-population rows). v54 gives the LIRF tree `ens_v22_lirf_q` (1/4 LightGBM group,
3/4 CatBoost group) share 0.25 there, and v45's u600 gate leans to it. On July 2025 thinned to July 2026's coverage (a13), the tree loses
much more than the network on the rows that go dark: 412.96 → 440.70 s RMSE (worse than v9's 434.7 s), against 429.92 → 435.70 s for the
shipped LIRF network. The trees never saw masked LIRF rows: the committee's masked-tree design (`docs/tree_mask`) left LIRF out, and LIRF's
own unmatched fitting rows are mostly January–April 2025. Does training-time masking of LIRF's matched rows make the tree better where
LIRF is dark?

## Design

- The seven learners of `ens_v22_lirf_q` refitted on their own seasonal fold (fit 2025-02..06, 08, 09; early stopping October; score
  January + July 2025) with identical settings and seeds; only LIRF non-population matched **fitting** rows are masked: weight 1 − p plus a
  masked copy with weight p.
- Two semantics. **focal** (`prc2026_tree_mask.augment`): the 15 focal ADS-B columns NaN, match flag 0. **dark**: focal plus the neighbour
  columns set to what LIRF rows carry in July 2026 (counts 0, the eight other neighbour columns NaN). Read before the declaration
  (label-free): July 2026 ranking LIRF rows carry neighbour counts 0 on 100% of rows and NaN medians on 99.3%, July 2025 thinned the same,
  January 2025 NaN everywhere; focal copies keep naturally tracked neighbour values, so they do not look like July 2026's rows. The brief
  asked for focal p 0.5 on all seven learners; this reading moved the CatBoost group to dark.
- Runs (8 threads each, arcw0–arcw3, two lanes per worker): LightGBM s31 / cap_s32 / nm_s33 × {focal 0.25, focal 0.5, dark 0.5}; CatBoost
  ctr2_s34 / ctr3_s35 / ctr2_nm_s36 / ctr3_nm_s37 × dark 0.5; unmasked controls of lgb_s31 and cat_ctr2_s34 (refit noise, thread count
  included).
- Candidates: `dark50_all` (all seven), `dark50_lgb`, `focal50_lgb`, `focal25_lgb` (LightGBM group only), each inside the saved ensemble's
  own leaf list. tree' = set tree + (candidate ensemble − saved ensemble) on the same inputs.
- Reading: v54's dark arithmetic (network' = 0.30 ens_ctx_lirf + 0.70 masked LIRF network, share 0.75, gate u600) on thinned July and
  January 2025 (`tools/run_lirf_dark.build_sets`), labels below 20,000 s, paired-day 95% interval.
- GO: thinned July ≤ −0.5% with interval below zero, January 2025 ≤ 0, and the thinned-July gain more than twice the control reading.

## Fits

15 seasonal-fold fits, none failed; arcw0–arcw3 started 03:20 UTC, each deallocated right after its SHA-256-checked mirror (arcw2 05:15,
arcw3 05:28, arcw1 05:33, arcw0 06:05 UTC). LightGBM 6–8 minutes per fit, CatBoost 1.6–2.5 hours (8 threads, two lanes on arcw0).
29,840 masked copies per run (LIRF non-population matched fitting rows). **Both controls reproduce the saved learners exactly** (lgb_s31
best iteration 1,659, cat_ctr2_s34 13,218; maximum prediction difference 0.0 s at 8 threads), so the refit-noise reading is 0 and only the
−0.5% bar binds. Masked best iterations: CatBoost 10,254–13,037, LightGBM 1,594–2,244. Before any candidate: the saved ensemble rebuilt from
its leaves equals the stored seasonal `ens_v22_lirf_q` (0.0 s) and, re-predicted on the a13 thinned frame, the measure table's thinned
column (0.0 s).

## Result: no GO (`design_report.json`)

v54's dark arithmetic; primary = route rows with labels below 20,000 s; tree RMSE = the tree component alone on the same rows.

| candidate | thinned July | January 2025 | natural July (864 rows, descriptive) | tree RMSE thinned July (saved → candidate) | GO |
|---|---|---|---|---|---|
| dark50_all | −0.102% [−0.43, +0.23] | +0.041% [−0.32, +0.38] | +0.021% | 441.77 → 438.03 s | no |
| dark50_lgb | −0.087% [−0.18, +0.01] | +0.075% | +0.253% [+0.07, +0.46] | 441.77 → 440.77 s | no |
| focal50_lgb | −0.046% [−0.13, +0.04] | +0.028% | +0.117% | 441.77 → 441.28 s | no |
| focal25_lgb | −0.041% [−0.14, +0.05] | +0.050% | +0.097% | 441.77 → 441.31 s | no |

- **Masking makes the LIRF tree only slightly better where LIRF is dark** (all seven learners, dark semantics: −3.7 s standalone on
  thinned July, −0.85%) and the stack barely notices it at the tree's share 0.25: −0.10% with an interval across zero, January 2025 a
  little worse. The dark semantics beats focal on the LightGBM group, but neither reaches the bar.
- On naturally matched July rows the masked ensemble is slightly worse standalone (413.57 → 414.66 s), the usual price of masking.
- Descriptive: the least-squares network share on thinned July stays at 0.875–0.9 for every candidate (as for the control), so a better
  tree does not move the dark-month share either.

Masked LIRF trees are closed. The 6.7% loss of the tree when LIRF goes dark is not recovered by training-time masking of its matched rows.
