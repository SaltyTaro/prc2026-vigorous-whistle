# Stacker weight / ten-seed bag (`configs/stack_weight_v1.json`) — ACCEPTED: `w65_bag10` (7 October 2026)

Declared 03:34 UTC (6abd13e) before any reading; governing tool `tools/run_stack_weight.py` (a01c5cf); run on arcw4 03:42-04:00 UTC,
mirrored hash-checked (`runs/stack_weight/mirror/out.tar`, fe158447...), arcw4 deallocated 04:03 UTC. Results
[`results_v1.json`](results_v1.json). No official score, no leaderboard, nothing under `runs/submissions`, nothing uploaded.

## Identity checks

- Development: `w50_bag3` (the shipped weight and seeds) through the same code equals v68's development proxy
  (`runs/opdi_stack_lszh/dev_v68_proxy.parquet`, 938a91cd...) on every row, max |diff| 0.
- Ranking: `tools/run_opdi_stack_decl.step_stack_opdi(..., ['LSZH'])` unchanged, then the chain, reproduces the uploaded v68
  (d41d8705...) byte for byte; the per-seed ranking predictions recombined at `w50_bag3` equal that step's values exactly (max |diff| 0).

## Development (against v68's development proxy; paired-day 95% intervals in %, seed 20261004)

| candidate | January | July | November | December | eq. mean | clauses 1-3 |
|---|---|---|---|---|---:|---|
| w65_bag3 | −0.0088 [−0.041, +0.000] | −0.0175 [−0.028, −0.009] | −0.0621 | −0.1048 [−0.136, −0.074] | −0.0437% | no (Jan interval) |
| w80_bag3 | −0.0066 [−0.054, +0.010] | −0.0096 [−0.028, +0.006] | −0.0917 | −0.1724 [−0.233, −0.113] | −0.0628% | no |
| w100_bag3 | +0.0124 [−0.034, +0.061] | **+0.0385** [+0.012, +0.066] | −0.0859 | −0.2080 [−0.305, −0.112] | −0.0524% | no (July worse) |
| w50_bag10 | +0.0019 [−0.001, +0.010] | +0.0041 [−0.001, +0.009] | +0.0042 | −0.0140 [−0.026, −0.001] | −0.0026% | no |
| **w65_bag10** | **−0.0079 [−0.035, −0.001]** | **−0.0116 [−0.023, −0.002]** | **−0.0604** | **−0.1198 [−0.146, −0.093]** | **−0.0465%** | **yes** |
| w80_bag10 | −0.0079 [−0.051, +0.005] | −0.0014 [−0.021, +0.016] | −0.0918 | −0.1873 [−0.239, −0.135] | −0.0655% | no |

Month RMSE of `w65_bag10`: 320.0147 / 250.8358 / 181.8835 / 163.3537 s (v68 proxy 320.0400 / 250.8650 / 181.9933 / 163.5496).
The gain grows with weight in November / December and shrinks or turns in January / July; ten seeds narrow the intervals enough for
w 0.65 only.

## Clause 4 (transfer guard, ten-seed bags; ranking / development RMS of the guarded bag prediction)

v3: EDDF 1.40, EDDM 1.02, EGLL 0.85, EHAM 1.27, LEBL 1.15, LEMD 1.08, LFPG 1.13, LSZH 1.27, LTFM 1.15; v2 at LIRF 1.07; OPDI bag on
LSZH used rows 1.04. All inside [0.5, 2.0]. **Selected: `w65_bag10`** (the only candidate accepted).

## Ranking dry run (label-free; version number provisional)

`runs/stack_weight/rank/dry_v69/vigorous-whistle_v69.parquet`, **SHA-256 `192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665`**
(v68's chain with the stacking value at weight 0.65 from ten-seed bags). Against v68: 252,691 rows differ, squared change **5.04e6**,
RMS 3.82 s, mean −0.42 s, max +141 / −272 s; by airport (squared change): LIRF 1.15e6 (21,856 non-population rows), LSZH 0.84e6,
LFPG 0.78e6, EGLL 0.63e6, LTFM 0.50e6, EDDF 0.40e6, EHAM 0.26e6, LEMD 0.19e6, EDDM 0.16e6, LEBL 0.13e6. LIRF population rows and
guard rows unchanged; minimum −12; int32; template order. Checked again on the workstation against
`runs/submissions/vigorous-whistle_v68.parquet`.

Build (copy the dry-run bytes, `validate_submission`, receipt) and upload only at the user's word.
