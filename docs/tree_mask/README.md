# Masked trees (1 October 2026): no GO in either family, closed

Declaration [`configs/tree_mask_design_v1.json`](../../configs/tree_mask_design_v1.json) (08:47 UTC, d16a1c5, before any fit; one
amendment, below). October design fold only: no scoring month, ranking row, official score or published aggregate is read. Nothing is
exported or uploaded.

## Question

v52's masked member came from training the context network with a quarter of its matched rows shown as unmatched (`src/prc2026_ctx3/train.py`):
on October's unmatched rows it read −1.05% against the same network unmasked. Does the same trick help the tree committee on rows without an
ADS-B match? The trees never had it: the unmatched-row specialist ([`docs/adsb_unm`](../adsb_unm/README.md)) only re-weighted or
restricted the existing unmatched rows.

## Method

- `src/prc2026_tree_mask/` (tests `tests/test_tree_mask.py`): the exact static expectation of the network's per-step draw. Every matched
  fitting row keeps weight 1 − p and gains a copy with weight p whose 21 focal ADS-B columns (the network's declared masking columns) are
  NaN and whose match flag is 0; neighbour columns and every non-ADS-B column are untouched. p = 0 is the unweighted control call.
- `tools/run_tree_mask_design.py`: the learner-family data path unchanged (314 committee features, label target, LIRF out of fitting
  and early stopping). LightGBM = `configs/timeline_v1.json`'s block (control: the learner-family `ref_lgb` seeds 1–4, not refitted);
  CatBoost = the committee's `cat_ctr2_x19` recipe (control `cat_ref` refitted on the same seeds).
- Reading (`tools/tree_mask_report.py`): the learner-family recipe proxy with the network pool P1 = P0 plus the masked network m25_32 at
  0.4 on unmatched rows (standing in for v52's masked member). Standalone on October's 58,494 unmatched non-LIRF rows; a swap proxy
  (committee + w · (variant − control) on unmatched rows: what re-fitting the committee with masking would do); a member proxy.
- Hardware: arcw0–7 CPU; each worker was deallocated after its runs were mirrored and hash-checked.
- **Amendment** (09:08 UTC, before any CatBoost prediction was read): after the LightGBM family read null, the CatBoost family was cut to
  seeds 1–2; its seed-3/4 fits on arcw0–3 were stopped about five minutes in, and those workers went to the LIRF context-network design.

## Result: no GO ([`design_report_lgb.json`](design_report_lgb.json), [`design_report_cat.json`](design_report_cat.json))

October unmatched non-LIRF RMSE of seed means; committee 215.879 s, network pool P1 207.526 s, table proxy 176.177 s.

| variant | unmatched | vs control | interval (s) | matched (control) | swap w = 1 (share 0.75 / 0.85) | decision |
|---|---:|---:|---|---|---|---|
| ref_lgb (control, 4 seeds) | 227.732 | | | 123.96 | | |
| lgb_m25 | 227.494 | −0.105% | [−1.00, +0.57] | 126.66 | +0.0062% / +0.0016% | no |
| lgb_m50 | 227.171 | −0.246% | [−1.33, +0.20] | 130.08 | +0.0040% / −0.0001% | no |
| cat_ref (control, 2 seeds) | 216.132 | | | 119.86 | | |
| cat_m25 | 216.618 | +0.225% | [−0.93, +1.91] | 121.56 | +0.0195% / +0.0083% | no |

- **Masking does nothing useful for trees.** LightGBM gains a quarter of a percent on unmatched rows at most, inside seed noise;
  CatBoost (the committee's main family) gets worse. Matched rows lose 1.4–4.9% in every variant.
- **The committee would not move.** Transplanted into the committee on unmatched rows the change is +0.004% to +0.020%.
- Reading: a tree routes unmatched rows down their own branches; the masked copies only add matched rows' labels there, which the
  network can share through its representation and a tree cannot. **Lever closed on evidence.**
