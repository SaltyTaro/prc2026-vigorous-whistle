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
