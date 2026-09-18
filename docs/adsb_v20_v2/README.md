# ADS-B v20_v2: shipping the thirteen-CatBoost committee under a post-hoc 0.02% bar (18 September 2026)

`docs/adsb_v20/README.md` recorded that `adsb_v20_v1` selected nothing: its best candidate
`ens_v20_bag_q_all_70` (one quarter of the nine-LightGBM group mean, three quarters of the
thirteen-member combination-CatBoost group mean, blend 0.7 on every row outside LIRF over
v9 + the v12 residual) improved the v17 development predictions by −0.021 / −0.020 /
−0.046% (mean ratio 0.99971) with every paired-day interval below zero, 0.02 percentage
points short of the pre-registered 0.05% bar. The user, told that the gain is about 0.07 s
and that shipping it bends the pre-registered rule a second time in a day, directed
"ship it anyway".

## Declaration `configs/adsb_v20_v2.json` (post-hoc relaxation, recorded as such; commit 2db72ac)

Identical to `adsb_v20_v1` — same learners, ensembles, candidates and baseline (v17's
development predictions) — except the bar: mean ratio at most 0.9998 (0.02%); the other
two conditions kept (not worse than v17 on any month; every month's paired-day interval
below zero). Declared with the v20_v1 scores in hand (its declaration and selection are
hash-bound parents, as are all seven v20 learners' and eight ensembles' outputs).

## Selection ([`development_results.json`](development_results.json))

Reproduces the v20_v1 table exactly (reproduction check ratio 1.0) and selects
**`ens_v20_bag_q_all_70`**: 330.302 / 271.528 / 192.187 against v17's 330.372 / 271.583 /
192.276 (−0.021 / −0.020 / −0.046%; intervals [−0.16, −0.02] / [−0.08, −0.03] /
[−0.14, −0.04] s). `ens_v20_bag_q_ctr_70` ties to 1e-6 in mean ratio and loses on name.

## Export, verification, audit

`tools/export_adsb_v11.py --version 18` → **`runs/submissions/vigorous-whistle_v18.parquet`,
SHA-256 `663e9d38a9f89064b5692932f9fd9bd11a8b71e3c018d6fdaf4d5036177e0807`**, 344,841 rows,
min −12, max 120,549, refit rounds 1,173 ([`export.json`](export.json)).
`tools/verify_adsb_v11.py --version 18` passed: receipts, v9 reproduction from scratch,
residual recomputation (max difference 0.0), out-of-scope rows equal to v9, 300 of 300
independent segment re-derivations agree ([`verification.json`](verification.json)); no
hash-bound file was touched while it ran. `tools/audit_export.py --version 18`
([`export_audit_v18.json`](export_audit_v18.json)): RMS difference to v17 4.2 s per row
(no row moved by more than 600 s), to v16 7.4 s; extremes, negative-proxy rows and
disruption days unchanged.

## Upload

Prepared for the user's upload decision (would be the second of the five slots of 18
September UTC). Expected official effect ≈ −0.07 s.
