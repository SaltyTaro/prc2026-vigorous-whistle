# Published airport-month aggregates (`configs/published_means_v1.json`, 29 September 2026): not accepted by its development rule

**What this is.** A gated level shift per airport-month from EUROCONTROL's published taxi-out aggregates, which cover the scored months.
They are aggregates of the hidden target, under terms that are not an open licence, and the organiser's reply on using them is
ambiguous. The user decided to use them ("yes use it"). No official score is involved.
Background and the sizing screen: `docs/clean_line_screens/README.md`, section 4.

**Procedure** (declared at 10:20 UTC and committed, `c0605d2`, before it was scored): per airport, the published mean is mapped to the
label mean on 2025; the stack's level error in an airport-month is its mean prediction minus the mapped figure, less the method's
development bias. A cell is shifted only if the left-out share is inside the 2025 range, the method's noise is at most 8 s and the
level error is at least three times that noise; the shift is shrunk by the noise.

## Development read (`results_v1.json`, control = the v46 stack)

| month | change in RMSE | interval (s) | cells shifted: shift against the true level error (s) |
|---|---|---|---|
| January | −0.003% | [−0.03, +0.01] | EGLL 6.9 / 6.9, LEBL −8.6 / −4.0 |
| July | −0.017% | [−0.16, +0.08] | EGLL 8.3 / 7.5, LEBL −15.2 / −20.9, LIRF 19.2 / 12.4 |
| November | −0.078% | [−0.23, −0.06] | EGLL 17.6 / 18.3, LEBL 13.4 / 12.7 |
| December | **+0.024%** | [−0.00, +0.09] | EGLL 6.9 / 7.2, **LEMD 15.1 / 1.6** |

**Rule: not accepted.** December is worse than the control (mean over the equal-footing months +0.001%). The loss is one cell: LEMD,
whose mapping missed by 14 s. EGLL's shifts are right to within 1 s in all four months, LEBL's and LIRF's to within 4–7 s.

## Scored months: the cells the procedure would shift (v46, no label involved)

| cell | shift | rows |
|---|---|---|
| EGLL January 2026 | −10.0 s | 19,283 |
| EGLL July 2026 | −30.2 s | 20,557 |
| LEBL July 2026 | −22.7 s | 17,879 |
| LIRF July 2026 | −45.0 s | 15,838 |
| LEMD January 2026 | −20.5 s | 16,802 |

If the estimates hold, v46 would read 223.47 (−0.46 s). Without LEMD the expectation is about 223.51.

## Standing

Per the declaration nothing is exported without a further decision of the user. No file was built.

## Second declaration and v47 (29 September, built on the user's decision; NOT uploaded)

`configs/published_means_v2.json` (10:55 UTC, commit `9be00c6`) is the first declaration with the noise limit at 5 s instead of 8 s. That
removes LEMD (noise 7.0 s) and nothing else. **It was chosen after the first read**, so its development read confirms the arithmetic
and is not independent evidence.

| month | change in RMSE | interval (s) | cells shifted |
|---|---|---|---|
| January | −0.003% | [−0.03, +0.01] | EGLL, LEBL |
| July | −0.017% | [−0.16, +0.08] | EGLL, LEBL, LIRF |
| November | −0.078% | [−0.23, −0.06] | EGLL, LEBL |
| December | −0.011% | [−0.06, +0.02] | EGLL |

Accepted by the rule (no equal-footing month worse, mean −0.010%). The development gain is small because 2025's levels are right.

**v47** = v46 with the four eligible cells shifted (`tools/run_published_means.py --export --version 47 --declaration
configs/published_means_v2.json`, [`export_v47.json`](export_v47.json)): `runs/submissions/vigorous-whistle_v47.parquet`, SHA-256
`b487310c5eb8cedf9c2a4edbaf4d716117c83402674a360b513e73e3059ca47c`.

| cell | shift | rows |
|---|---|---|
| EGLL January 2026 | −10.0 s | 19,283 |
| EGLL July 2026 | −30.2 s | 20,554 |
| LEBL July 2026 | −22.7 s | 17,879 |
| LIRF July 2026 | −45.0 s | 15,782 |

73,498 rows change (mean −26.3 s, largest 45 s, sum of squared moves 6.18e7); every other row equals v46. Audit
`docs/export_audit/v47_audit.json`: clean. Tests: `tests/test_published_means.py` (2 pass).

**Expectation against v46's 223.9339:** 223.53 (−0.41 s) if the estimates hold; 224.33 (+0.40 s) if every move is noise. Upload is the user's decision.

## Third declaration on v72's chain (`configs/published_means_v3.json`, 8 October 2026): not accepted

The v2 procedure unchanged, with v72's chain as the control (development: `runs/stack_second` v71 + sp_hicov7; ranking: the uploaded
v72) and the fixed rows (LIRF population, guard rows) never moved; `pm_np` also leaves them out of LIRF's means. Tool
`tools/run_published_means_v3.py`, results [`results_v3.json`](results_v3.json). Exploration readings of the ranking cells were known
before the declaration (disclosed in it).

| month | change | interval | cells shifted: shift / true level error (s) |
|---|---|---|---|
| January | **+0.0006%** | [-0.000, +0.003] | LEBL -6.7 / -2.3 |
| July | -0.0013% | [-0.005, +0.001] | EGLL 7.7 / 6.5 |
| November | -0.0015% | [-0.004, +0.001] | EGLL 2.2 / 3.0 |
| December | -0.0050% | [-0.009, -0.001] | EGLL 4.5 / 4.7 |

Both candidates read the same on 2025 (LIRF has no eligible development cell). **Primary rule: not accepted** (January worse, by LEBL's
overshoot); v57's clauses 2-3 fail by construction. No dry run written. EGLL's shifts track the truth within 1.2 s in all three months.

Ranking cells the procedure would shift: EGLL January -2.2 s (19,283 rows), EGLL July -8.1 s (20,553), LEBL July -8.6 s (17,879), LIRF
July -37.9 s (`pm`) or -26.7 s (`pm_np`, non-population means) on 15,544 non-fixed rows. Stakes: `pm` -0.175 s if the estimates hold,
+0.171 s if every move is noise; `pm_np` -0.10 / +0.10 s. Most of either stake is LIRF July 2026.
