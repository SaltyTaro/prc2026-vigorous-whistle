# Case files, round three, on the v65 development proxy (6 October 2026) — closed, no GO

Lever #4 of the final-week plan. Diagnostic, 2025 development months only. Labels are read only to pick and classify rows and to
score screens. No official score, no leaderboard, no upload, nothing under `runs/submissions`, nothing declared. Scratch (git-ignored): `runs/dig/case4/`.

| script | output |
|---|---|
| `census.py` | `dev65.parquet`, the census below |
| `build.py` | the 260 rows, identities at all ten airports (tunnel 18's callsign / key-map identity) |
| `context.py` | per-row context (tunnel 19's script, paths changed) |
| `dump.py` | `dump.txt` |
| `classify0.py` | `cases_k0.parquet`, `cases_classified.parquet` |
| `lobt_flag.py`, `arrfill.py`, `screen_af.py`, `member_audit.py` | the screens below |

## Control

`runs/ctx_ownpoints_dev/chain/own_all_w40/dev_v65_proxy.parquet`, column `v65`: 320.3415 / 251.4886 / 182.1391 / 163.8910 s
(January / July / November / December 2025, 672,428 rows), exactly the recorded stage-2 result of `docs/ctx_ownpoints_dev`.

## Census of the v65 proxy (ordinary rows, label at most 3 h, outside the LIRF population)

Ordinary non-population rows hold 55.1% of all squared error; the LIRF population 17.0%.

| cell | rows | RMSE (s) | share of ordinary squared error |
|---|---:|---:|---:|
| LTFM (all unmatched) | 91,338 | 202.3 | 18.0% |
| LFPG unmatched | 64,894 | 232.0 | 16.8% |
| LIRF dark (matched, not gate-traced) | 26,700 | 340.7 | 14.9% (fill rows 5.7%) |
| EGLL unmatched | 32,852 | 220.1 | 7.7% |
| LIRF unmatched | 18,463 | 286.4 | 7.3% |
| EGLL dark | 35,605 | 178.1 | 5.4% |
| LEBL dark | 37,016 | 155.6 | 4.3% |

By label class (diagnostic): early record 36.6% (bias +141 s), agree 29.4%, late 14.7%, fill 12.4%, no NM clock 6.9%. Rows with
|error| > 600 s are 7,268 and hold 45.0%; 69% of them are under-predictions.

## The rows

260 rows, the largest squared error of the v65 proxy after removing the LIRF population, labels above 3 h, LIRF fill rows (block within
5 s of schedule), no-NM rows, extreme published days (|implied day shift| >= 80 s or |discrepancy| > 1e5 s) and the 570 rows already read
in tunnels 17, 19 and 20:

- 110 LIRF: 69 dark, 36 unmatched, 5 gate-traced;
- 80 EGLL / LEBL / LEMD dark or unmatched: EGLL 19 / 18, LEBL 26 / 2, LEMD 6 / 9;
- 25 LTFM;
- 45 others: LFPG 33, LEBL 7, EDDF 3, EGLL 1, EHAM 1.

Together 3.6% of ordinary squared error: the remaining error is diffuse. 88% of the LIRF rows and 97% of the EGLL / LEBL / LEMD rows have
an identified aircraft. Each was read with every supplied field and clock interval, the stand's arrivals and departures over 12 h, the
runway and airport take-off sequence, the published day discrepancy, the aircraft's own adsb.lol turnaround and other aircraft parked
within 60 m.

## Mechanisms

| mechanism | rows | share of the rows' squared error | label-free trigger? |
|---|---:|---:|---|
| early record, no own motion seen (block 25-120 min before the NM clock) | 104 | 43.3% | no (tunnel 4, `lirf_mechanism.md`) |
| early record, own ADS-B shows the real move late (push 2-15 min before take-off) | 77 | 28.8% | no: ADS-B confirms the physics, not the record |
| fill: block at the schedule (within ±6 s), flight delayed; LIRF and also LEBL / LEMD / EGLL | 54 | 22.0% | no (screened below) |
| LIRF over-prediction: the stack hedges toward the schedule interval and the row is not a fill | 21 | 4.7% | no (tunnel 20) |
| other | 4 | 1.3% | no |

Every large under-prediction is an early or fill record. In the early rows the block sits within 10 minutes of the filed EOBT / LOBT
55-81% of the time (LFPG 56%, LIRF 69-81%) while the aircraft moves with the NM clock, but within 2 minutes agree-class rows sit as
close to those clocks (16-31% against 12-26%): closeness to the plan does not mark the class. The 54 fill rows are LIRF 28, LEBL 18,
LEMD 5 and EGLL 3; in 15 of them the NM clock is a whole five minutes after the schedule (5 at exactly + 40 min).

**Side finding (no lever).** At LIRF, LEBL, LEMD, EGLL, LFPG and LTFM, block − schedule is uniform over −6..+5 s with a third-height
bin at +6 s, and the block's second-of-minute is 0 for 1/12 of rows (EDDF: 1/60). Block times there are minute records with about ±6 s
of jitter. The LIRF fill definition "within 5 s" therefore misses about 10% of fills; it changes no prediction.

## Screens (month-blocked, against the v65 proxy; change of complete-month RMSE)

**1. Own-motion bound at LIRF.** v1 (EDDF / EGLL / LFPG) and v2 (five more airports) never covered LIRF, whose identity tables
arrived with tunnel 18. Same stand, not gate-traced, 60 < B <= 2,700 s: 20,428 dev rows, valid 99.7%. But v65 already sits above B on
all but 34 of them (LIRF clock rows carry the fill hedge), and on those 34 the bound holds 73%. Best: −0.0024 / −0.0023% in July /
December, +0.0008% in November. **Closed.**

**2. LOBT equal to the schedule on delayed rows** (NM clock 30 min or more after the schedule). v65 bias by airport −24 to +90 s, sign
changing by month (LIRF +69 / +108 / +58 / −30 s). Nothing to correct. **Closed.**

**3. Arrival fills as a sign of departure fills.** The inbound arrival at the same stand recorded at its schedule (within 6 s), or the
operator's arrivals at the airport that day at least half at schedule. At LIRF, on delayed clock rows with the operator-day signal, the
departure fill rate is 49% against 6%, and v65's bias is +131 / +120 / +382 / +219 s (461 rows). At the other nine airports there is no
departure fill signal (rate 0-2% either way). A move toward the schedule interval, `v65 + w α (interval − v65)+`, with α fitted on the
other months:

| trigger | w | January | July [95% s] | November | December [95% s] |
|---|---|---:|---:|---:|---:|
| operator-day ≥ 0.5 | 0.5 | −0.0004% | **+0.0220%** [+0.00, +0.11] | −0.0117% | −0.0074% [−0.03, +0.00] |
| operator-day ≥ 0.5 | 1.0 | +0.0001% | **+0.1029%** [+0.10, +0.47] | −0.0231% | −0.0115% |
| inbound fill | 0.5 / 1.0 | −0.0004 / −0.0007% | −0.0009 / −0.0011% | 0.0000 / +0.0006% | −0.0011 / −0.0015% |
| either | 0.5 / 1.0 | −0.0007 / −0.0011% | +0.0004 / +0.0047% | −0.0034 / −0.0064% | −0.0019 / −0.0025% |

The bias is the mixture itself: a few fill rows with long intervals against many ordinary rows. Same result as the 29 September
arrival-side follow-up (`docs/clean_line_screens/lirf_mechanism.md`). These are LIRF fill-like rows, the class behind v10 / v43 / v50 /
v56. **Closed.**

**4. Audit of v65's member change** (v65 − v64, by airport × ADS-B state, own-point coverage, first-motion and gap bands, direction ×
size). Positive in every cell except LTFM (gain / squared change −0.10; −0.002 / +0.010 / −0.021 / −0.007 points by month) and LIRF,
which moved only through the stacker (dark −1.68, unmatched −2.88; no month beyond ±0.013 points). Reverting either is worth under
0.01%. **Closed.**

## Verdict

No GO; nothing declared, no dry run. The worst rows of the v65 chain are early and fill records, decided by when the airport writes
the off-block, which no supplied or ADS-B input records. The own-motion bound has nothing left to bound at LIRF, and the arrival-side
fill signal is real but is a mixture the chain already hedges. The member of v65 shows no label-free cell worth guarding.
