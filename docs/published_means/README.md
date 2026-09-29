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
