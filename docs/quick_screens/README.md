# Quick screens, 7 October 2026 (coordinator, local, light; nothing built)

One-line hypotheses checked in minutes before spending a lane. Each is closed unless stated.

| Hypothesis | Check | Result |
|---|---|---|
| AOBT_3 of Air France at LFPG now equals the airline OUT (2N AOBT_3 - OUT median moved +120..180 s in 2025 -> +20 s in July 2026), so it could pin every AF departure | July 2026 2N rows: share within 60 s / 120 s of OUT; RMS of an AOBT-based pin vs the OUT pin, against v69's | 31% / 50% (2025: 22% / 32%); AOBT pin 257 s vs v69 227 s RMS against the OUT pin. Closed |
| v69 is biased on July 2026 LFPG rows because the NM clock moved | v69 - pin mean on pinned rows, 2025 vs 2026; calibration slope on unseen / dark rows | +7 (2N) / -10 s (8D) in July 2026, inside 2025's -32..+38; slope 1.06 unseen (2025 0.90-1.12). Closed |
| Pinned rows reveal a day-level bias of other LFPG unseen rows | correlation of day means, Jul / Nov / Dec 2025 | -0.44 / -0.37 / +0.23, slopes ~0. Closed (as the TTA screen for gate rows) |
| Other data-link tokens pin LTFM / LIRF departures | centred token statistics of the extension scan | good tokens only at EGLL / EHAM / LFPG; LTFM: Turkish QQ, 49 rows. Closed |
| MVT_ID or FLIGHT_ID order leaks the recorded off-block | within airport-day partial rank correlation with the recorded off-block given take-off, July 2025 | MVT_ID ~0 (|r| <= 0.06); FLIGHT_ID follows IOBT / LOBT (partial 0.98-0.99), i.e. the NM plan, and is null on the no-NM rows where it could add anything. Closed |
| High predictions are over-confident (1,059 ranking rows > 3,600 s) | v70 dev proxy by prediction band | calibrated: bias +22..+73 s for 2,400-5,400 s bands, -159 / -386 s for 5,400-14,400 s on 147 / 108 rows (RMSE 2,273 / 3,716 s); nothing to shrink. Closed |
| Monthly published taxi-out means vs v70's level | regime audit on v68 (`docs/regime2/`) | EGLL +9.8 / +4.5 s (about 0.01 s), LIRF July +38 s unreliable. Closed earlier |
| Block-time RECORDING convention changed in 2026 (`docs/rec_drift/`, 6a7d392, arcw1-3) | arrivals' recorded in-block - ADS-B stand stop (868k arrivals matched), stand fixed effects, 2025 range vs Jan / Jul 2026; AOBT_3 - motion start; v70's implied offset | every observable airport inside its 2025 range (EDDF -2.1, EGLL +1.0 s in July); AOBT_3 drifts (EGLL +52 s July) are input drift, not recording; v70 EGLL July +20 s matches 2025's AOBT_3 relation; largest cell 0.4-1.9e6 s². LFPG / LIRF / LTFM unobservable. Closed |
| New 2026 stands (EDDM 101-106: 1,182 July rows; EDDF J / V: 299) carry a bias | v70 - ADS-B-implied taxi on gate-traced rows, new vs old stands | EDDM new +57 / old +24 s mean (same 117 s RMS): ~1.4e6 s². EDDF 2026 gate rows too sparse to read. Closed |
| January 2026 disruption days (EHAM 2-7, LFPG 7): gate rows reveal a day error of the others | v70 - implied on gate rows by day | EHAM stable ~+70 s every day, worst days -35..+33 (at most ~2e6 s² if read as bias); LFPG has 0-5 gate rows a day. Closed |
| Renamed runways / unseen categories in 2026 | 2026 values absent in 2025 | runways none; aircraft types LIRF 107 / 286 rows; operators 1-5% per airport-month (fallback checked by the regime audit). Closed |
| Dark rows' first points show the pushback (track reversal) | EGLL 15 July 2025 | 1 of 163 dark rows; dark rows are first seen ~16 min after the recorded off-block (mid-taxi). Closed |
| **Residuals of consecutive take-offs on a runway are correlated** | v70 dev proxy, lag-1 within 180 s | **real**: 0.11-0.13 every month; LTFM 0.26, LFPG 0.20, LEMD 0.15. The shared state is on the rows nobody observes |
| ... and a gate-traced neighbour's label-free residual carries it | pseudo-residual of the nearest gate row (180 s) vs dark / unmatched focal residual | -0.04..+0.05 (even the gate neighbour's TRUE residual: 0.00-0.04). Closed |
| ... and a data-link-pinned LFPG neighbour carries it | nearest pinned row's (pin - value) vs LFPG focal residual | 180 s: Jul -0.01, Nov 0.17, Dec 0.17; 600 s: 0.03 / 0.10 / 0.13; at July 2026's pin density ~2e6 s². Closed (too small) |
| ... and arrivals' taxi-in excess in the same 10-20 min carries it | arrival taxi-in minus its (stand, runway) median, month-blocked | -0.03..+0.05 at LTFM / LFPG / LIRF / EGLL / LEMD. Closed (as the TTA screen at day level) |
