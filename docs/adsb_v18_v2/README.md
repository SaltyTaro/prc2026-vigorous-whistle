# ADS-B v18_v2: shipping the banked v18 committee under a post-hoc 0.05% bar (18 September 2026)

`docs/adsb_v18/README.md` recorded that `adsb_v18_v1` selected nothing: its best candidate
`ens_v18_bag_q_70` (one quarter of the six-LightGBM group mean, three quarters of the
nine-member combination-CatBoost group `ens_v18_ctr`, blend weight 0.7 on every row
outside LIRF over v9 + the v12 residual) improved the v16 development predictions by
−0.029 / −0.073 / −0.073% (mean ratio 0.99941) with every paired-day interval below zero,
0.09 percentage points short of the 0.15% bar. On 18 September the user directed that even
minor improvements are worth shipping at this stage of the challenge.

## Declaration `configs/adsb_v18_v2.json` (post-hoc relaxation, recorded as such)

Identical to `adsb_v18_v1` — same learners, ensembles, candidates and baseline (v16's
development predictions) — except the rule: eligible when not worse than v16 on any
month, mean ratio at most 0.9995 (a 0.05% mean improvement), **and** every month's
paired-day bootstrap interval of the RMSE difference to v16 lies below zero (a condition
the v18_v1 rule did not have; `tools/run_adsb_v11.py` gained the optional
`require_paired_day_intervals_below_zero` flag). Lowest mean ratio wins. Declared with the
v18_v1 scores in hand (its declaration and selection are hash-bound parents), so the
selection cannot be read as pre-registered at 0.05%. The v16 recipe candidate
`ens_v17_bag_ctr_c2_70` is the reproduction check (ratio 1.0).

## Selection ([`development_results.json`](development_results.json))

`tools/run_adsb_v11.py --config configs/adsb_v18_v2.json` reproduces v18_v1's table exactly
and selects **`ens_v18_bag_q_70`**: 330.372 / 271.583 / 192.276 against v16's 330.469 /
271.782 / 192.417 (−0.03 / −0.07 / −0.07%; intervals [−0.23, −0.02] / [−0.26, −0.14] /
[−0.19, −0.09] s). The next-best `ens_v17_bag_ctr_q_70` (re-weighting alone, −0.033%) fails
the interval condition on January.

## Export and verification

`tools/export_adsb_v11.py --version 17` → **`runs/submissions/vigorous-whistle_v17.parquet`,
SHA-256 `fe0f08e4f6aa235b9550a8e4690662cfc8d20f7591b2f9e826c401fd20089649`**, 344,841 rows,
min −12, max 120,549, refit rounds 1,173 ([`export.json`](export.json)).
`tools/verify_adsb_v11.py --version 17` passed: receipts, v9 reproduction from scratch,
residual recomputation (max difference 0.0), out-of-scope rows equal to v9, 300 of 300
independent segment re-derivations agree ([`verification.json`](verification.json)).
`tools/audit_export.py --version 17` ([`export_audit_v17.json`](export_audit_v17.json)):
same shape as v16 — RMS difference to v16 6.3 s per row (EHAM 8.3, LFPG 8.7, LIRF 0), no row
moved by more than 600 s, extremes and disruption days unchanged.

Procedural note: two tool edits made for the next round (an `--extra-features` option in
`tools/run_adsb_learner.py`, extra-feature sync in `tools/launch_adsb_learners.py`) were
started while the export ran; both files are hash-bound parents of this declaration, so
the edits were set aside (git stash / checkout) and the files restored byte-for-byte to
their declared hashes before the verification that is recorded here. The export itself
ran with every parent intact (its own `check_parents` passed at start).

## Upload and official result

v17 (241.5068, SHA `fe0f08e4…9649`) is
the incumbent. Never re-upload v1–v17.
