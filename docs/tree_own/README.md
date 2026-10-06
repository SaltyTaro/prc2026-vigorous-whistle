# Own-aircraft turnaround columns in the tree members: screen, NO GO (6 October 2026)

Final-week campaign, lever 3 (fork C). Declaration [`configs/tree_own_screen_v1.json`](../../configs/tree_own_screen_v1.json) (eb59cc6,
05:25 UTC, before any fit), amendment 1 [`..._amend1.json`](../../configs/tree_own_screen_v1_amend1.json) (e26bf96: control = v66,
pinned reading) and amendment 2 [`..._amend2.json`](../../configs/tree_own_screen_v1_amend2.json) (edbf51d: LIRF population rows held
at the control), both before any in-chain candidate score. 2025 development months only; no official score, no leaderboard, nothing
built under `runs/submissions`, nothing uploaded. Results: [`screen_results.json`](screen_results.json).

## Question

The guarded stacker v3 and the context-network member (v65, then v66 at weight 1.00) carry the aircraft's own turnaround ADS-B points;
the tree members never saw them. Does a tree member refitted with the 15 own columns improve v66's development chain?

## Fits

Six LightGBM leaves: the three LightGBM leaves of the LIRF table `ens_v22_lirf_q` (`lgb_lirf_s31`, `lgb_lirf_cap_s32`, `lgb_lirf_nm_s33`;
group share 12/48; the LIRF tree carries 0.45 of LIRF matched non-population rows) and the three x19 LightGBM leaves of the committee
(`lgb_x19_s24..s26`; share 39/468). Each in two variants (ctl = the leaf's own recipe; own = + the 15 columns) on the four folds the
development tables use (LOMO January / July, forward fold for November, January-November for December): 48 fits, `tools/run_tree_own.py`,
arcw4-arcw7 (`tools/launch_tree_own.sh`, `tools/mirror_tree_own.sh`), about 5 minutes each. CatBoost leaves were not screened (1.7-3 h
per fit).

- **Controls reproduce the saved leaves exactly** (24 of 24: LOMO and forward predictions 0.0 s; January-November models byte-identical
  to each leaf's refit, December differing only on the documented 19-20 ADS-B neighbour tie-order rows).
- **Deviation:** arcw5 / arcw6 sat on hosts with a looping Azure MANA NIC fault (1-2 CPUs of kworker load; 16-thread fits 3x slower) and
  ran at 12 threads; four 12-thread controls failed the literal check by num_threads text / 1e-13 s and were refitted at 16 threads on
  clean hosts before any candidate ([`thread_deviation.json`](thread_deviation.json)).
- Standalone (descriptive, development labels): the own columns improve the x19 committee leaves by 0.4-1.6% RMSE on non-LIRF rows in every
  month, the LIRF leaves by 0.2-0.6% on LIRF non-population rows except January (two of three slightly worse).

## Reading (v66's development chain)

Candidates: `lirf_exact` / `com_exact` (the three leaves replaced inside their frozen member lists), `lirf_fam100` / `com_fam100` (the
LightGBM family's own-minus-ctl change applied to the whole table), `both_fam100`. Each through v66's chain (`tools/run_tree_own_chain.py`):
model part with the candidate tables (v46 recomputed, own-points member at 1.00, LIRF layers recomputed), no-NM rematch, v48, extreme-day,
variance re-allocation, stack_resid_v3 refitted on the rebuilt components, intruder bound, identity extension, own-motion v1 and v2.
Reading = v66 proxy + (pinned candidate - pinned identity), the variance re-allocation's LightGBM pinned (deterministic, column-wise) in
both; LIRF population rows held at the v66 proxy. Identity: the unpinned zero-delta chain reproduces v66's model part and the v66 proxy
exactly (0.0; a first attempt differed by 0.028 s from the variance re-allocation's timing choice in December, preserved).

| candidate | January | July | November | December | equal-footing mean | intervals below zero | GO |
|---|---:|---:|---:|---:|---:|---:|---|
| lirf_fam100 | -0.0121% | **+0.0113%** | -0.0168% | -0.0229% | -0.0079% | 0 | no |
| lirf_exact | +0.0015% | -0.0043% | -0.0095% | -0.0178% | -0.0069% | 0 | no |
| com_fam100 | -0.0180% | -0.0175% | +0.0056% | -0.0396% | -0.0250% | 1 | no |
| com_exact | +0.0011% | +0.0012% | +0.0096% | -0.0228% | -0.0068% | 1 | no |
| both_fam100 | -0.0364% | -0.0085% | -0.0274% | -0.0682% | -0.0377% | 1 | no |

GO bar: equal-footing mean <= -0.05%, no month worse than +0.01%, two of three equal-footing intervals below zero. **No candidate passes.**
Without the population hold the readings move by at most 0.0003 points (descriptive, per score.json).

## Verdict

**NO GO, lever closed.** The leaves improve standalone, but in the chain the own information is already carried by the v66 network member
(weight 1.00) and the stacker; the committee enters the mix at 0.15, and the LIRF LightGBM group is a quarter of the LIRF table. Even
extrapolating the LightGBM family's change to the whole tables (`both_fam100`) reaches only -0.038% with one interval below zero, so
refitting the CatBoost leaves (about 50 VM-hours for the LIRF four alone) is not justified. No development round, no ranking refit, no
dry run.

Machines: arcw4-arcw7 used 05:30-08:15 UTC, each deallocated after its SHA-256-checked mirror (re-checked deallocated 13:00 UTC).
