# After v12 (246.4831, 3rd of 144; 2nd 245.2932, 1st 245.0207): plan updated 15 September 2026 ~12:10 UTC

## Ready to upload (needs the user's approval; allowance reopens 00:00 UTC 16 Sep, `--calendar-day`)

**v14** (`configs/adsb_v12_v5.json`, `ens_all_60`): v12's residual, then 60% of the
mean of two full-data learners (LightGBM and CatBoost on the timeline bundle's
276 model features + 27 ADS-B features, fitted without LIRF) on every non-LIRF
row. Development −0.51 / −0.52 / −2.18% vs v12 (mean −1.07%); expected ≈243.8
(first is 245.02). SHA-256 80903141…0e8f, verified. v13 (`adsb_v12_v4`,
LightGBM alone at 30%, −0.67% mean) is the superseded fallback, never uploaded.

## Done today

- v11 248.4368 (4th) → v12 246.4831 (3rd): ADS-B residual, LIRF excluded, neighbour features.
- v12_v1 (neighbour scopes) and v12_v3 (learner as residual feature) were null results;
  fixed post-hoc blends (v12_v4, v12_v5) work. Learner tooling: `tools/run_adsb_full.py`,
  `run_adsb_full_nm.py`, `run_adsb_learner.py` (--kind lightgbm|catboost, --target
  label|nm_residual, --exclude-airports), `ensemble_adsb_full.py`.
- All 382 surface days on the workstation (`data/external/adsblol/surface/`, 3.6 GB);
  pass-2 matches for every month with 1,332 stands; v11/v12 inputs frozen in `matches_v11/`.
- All six Azure VMs deallocated (arcw0 cleared of the user's finished poker project).

## Where the remaining error is (Nov-2025 error levels × ranking rows)

LIRF 40%, LFPG 19%, LTFM 15% of the squared error — no ADS-B coverage or a label
that is not a taxi time; the seven covered airports hold 27%. Practical floor
with this data ≈ 238–241; hard floor ≈ 211.

## Next levers (in order of expected value)

1. Learner diversity/strength: seed bagging of the LightGBM learner (4 min per fit),
   a CatBoost NM-residual variant, learners on the matched subset only.
2. Pushback failure modes (docs/adsb_v12/README.md): transponder-on late and
   "ready but waiting" gate rows; dark-segment geometry (name-based stand
   imputation for LEMD/LSZH/LEBL unknown stands).
3. Coverage for LFPG/LTFM/LIRF needs another data source (not adsb.lol).

Rules: declare before running; baseline = the incumbent's development
predictions; never re-upload v1–v14; the user decides on organiser contact.
