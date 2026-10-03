# Identity extension of the intruder stand bound — ACCEPTED by its declared rule; ranking stake tiny (3 October 2026)

Declaration [`configs/intruder_identity_v1.json`](../../configs/intruder_identity_v1.json) (after the disclosed exploration, before the
governing run); tool `tools/run_intruder_identity.py` (`--build-ids`, `--score`, `--dry-run`); results [`results_v1.json`](results_v1.json).
No official score and no leaderboard information is used.

**Mechanism.** The accepted layer (`configs/intruder_bound_v1.json`, v58) only uses intruders of a different aircraft type. Here arrivals and
departures at EDDF / EGLL / LFPG are identified by their ICAO address from the adsb.lol day files (nearest point of the same callsign at the
same airport within 600 s of the landing / take-off time; identity table `runs/intruder_identity/ids.parquet`, SHA-256 `3097dd06...`). A
same-type arrival Y != X that blocked in at our stand after our own inbound X (identified) and before our predicted off-block proves we
left earlier. Identification coverage: EDDF / EGLL 95-100% of movements in most months, LFPG 37-89% (2026: 77-84%), LTFM departures ~0
(excluded). Validity on the twelve 2025 months: EDDF 13 rows 92%, EGLL 112 rows 94%, LFPG 27 rows 100%.

## Governing run (applied after the accepted layer, month-blocked)

| month | change of complete-month RMSE | paired-day 95% | rows |
|---|---:|---|---:|
| January 2025 | -0.0601% | [-0.268, -0.010] | 11 |
| July 2025 | -0.0907% | [-0.205, +0.004] | 19 |
| November 2025 | -0.0086% | [-0.025, +0.000] | 3 |
| December 2025 | -0.1326% | [-0.274, -0.021] | 8 |

Equal-footing mean -0.0945%, no month worse, two equal-footing intervals below zero; twelve months 12/12 better (mean -0.048%), 11
intervals below zero. **Rule: accepted.**

## Ranking dry run (label-free; nothing under runs/submissions)

`tools/run_intruder_identity.py --dry-run --version 61` on v60's pre-intruder file (cff778e0...): the base plus the accepted layer
reproduces v60 (30021455...) exactly; the extension then moves **5 rows** (EGLL 1, LFPG 4; up 848-1,297 s), squared change 4.06e6 against
v60: `runs/intruder_identity/dry_v61/vigorous-whistle_v61.parquet`, SHA-256 `fe137f092fa68a3f838ed0cb5b3cc5d7e995707790accecef4fb3649dd6af4e8`.
At 2025's gain per unit of squared change (about 1) this is worth roughly 0.01-0.03 s.

Why so few: identification and inbound sequences are as frequent in 2026 as in 2025 (95 / 113 identified sequences in January / July
2026 against 133 / 168), but only 5 of them place the intruder within an hour before v60's predicted off-block, against 26 / 53 in the
development months. In 2025 these rows cluster on large under-prediction events (EGLL 30-31 July departure stop: labels 3,355-5,763 s
against 2,700-4,100 s predicted); the 2026 months have fewer of them and the current chain sits after fewer intruders. Not a defect.
