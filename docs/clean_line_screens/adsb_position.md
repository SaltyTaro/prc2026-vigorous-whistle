# Stand-observed rows that the matcher labels dark (29 September 2026) — closed

Nothing is declared, selected,
exported or uploaded. Report: [`adsb_position.json`](adsb_position.json) (scratch code, quoted there).

## Where the lead came from

- `docs/adsb_shift/README.md` (27 September, label-free): at EGLL in July 2026 the receiver coverage jumped, and the rows
  the matcher labels dark changed meaning. They are first seen 122 m from the stand (1,376 m in July 2025) with a slow
  start of 340 s (7 s): observed from the stand, but not classed as gate rows.
- A participant's public remark: inputs that encode where an aircraft was heard did not carry over from
  2025 to 2026, inputs that describe its movement did.
- **Disclosure.** This reading uses none
  of those figures: the rows are defined without labels, and direction and size come from the 2025 development labels
  only. Had it found a bias, that origin would have had to be stated in any declaration.

## Reading

Rows: ADS-B matched, not classed as gate, first point within 150 m of the stand, slow start of at least 120 s. Control: the
v46 development stack; labels below 10,800 s; LIRF excluded. Bias = label minus prediction, with a paired-day interval.

| state | rows | stack bias (s) | stack RMSE (s) |
|---|---:|---:|---:|
| gate | 148,554 | −0.6 | 85.8 |
| dark, at the stand, slow start | 36,478 | **−3.3 [−4.6, −2.0]** | 90.6 |
| dark, at the stand, already rolling | 25,875 | −0.6 | 107.1 |
| dark, first seen away from the stand | 167,361 | +0.5 | 154.8 |
| unmatched | 244,229 | −0.3 | 214.4 |

By airport and month the bias stays within ±13 s on the equal-footing months. EGLL: January −5.7 [−15.0, +1.9], July −3.4
[−16.8, +13.5], December −10.6 [−16.2, −4.3]. The larger November values (EGLL −19.5, LSZH −22.7, LEBL −17.0) are on the
nine-month fold side.

**Exposure, label-free:** 4,943 such rows at EGLL in July 2026 (2,504 in January 2026), against 1,925 and 2,034 in November
and December 2025. The twelve-month refits have seen this kind of row at EGLL in its winter months.

**Closed.** The stack treats these rows like gate rows already: its error on them is that of gate rows and its bias is a few
seconds. At the development bias, the EGLL July rows are worth less than 0.01% of the ranking's squared error.
