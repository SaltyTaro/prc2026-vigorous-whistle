# Heathrow and the UK CAA punctuality statistics (`configs/caa_heathrow_v1.json`, 30 September 2026) — not accepted, closed

**State: the declared rule selected nothing. No file was built.** The procedure reads the CAA's published average departure delay per
airline × destination × month at Heathrow (whole minutes; off-block against the planned gate time, early flights zero) and shifts
each group's taxi predictions on top of v48's procedure. January 2025 is worse than the control; the mean over the equal-footing
months is +0.004%. Results: [`results_v1.json`](results_v1.json). No official score is used.

## The source

`https://www.caa.co.uk/data-and-analysis/uk-aviation-market/flight-punctuality/uk-flight-punctuality-statistics/2026/` (and 2025):
per month, reporting airport, airline and origin/destination, the number of matched flights, the average delay in whole minutes and
the shares of flights in ten delay bands. July 2026 was published on 22 September 2026. Terms on the page: no resale, cite the CAA.
Files under `data/external/caa_punctuality/` (not in the repository; hashes in the declaration and in `verification.json`).

Mapping (built from 2025 monthly counts, `destinations.csv`, `airlines.csv`): 204 destination names to ICAO codes (95.6% of the
published departures) and 67 callsign prefixes to airline names; the counts per airline × destination × month then agree within 3%
for about 3,170 groups a year holding 92% of Heathrow's departures. Some flights are filed under their onward destination
(London–Singapore–Sydney and the like) and stay unmapped.

**Verification on 2025 labels** (`verification.json`): the published mean equals the label mean of max(off-block − schedule, 0) up to
the minute rounding, with an offset (published higher) of 0.10–0.44 minutes depending on the month and a robust sd across groups of
0.34–0.43 minutes (rounding alone: 0.29). About 3% of groups differ by more than a minute (a flight delayed by a day in one source
and a fill in the other).

## Procedure and read

Per group read (published count ≥ 30, our count within 3%): implied mean of max(delay, 0) under the row error, plus the definitional
offset from the other months, against the published mean; clipped to ±2 minutes; shrunk by the month's own dispersion of the level
errors beyond their noise (empirical Bayes, label-free); one taxi shift per group; rows predicted at 10,800 s or more keep their value.
Control = v48's procedure on the development stack. Rule: no equal-footing month worse, mean ≥ 0.01%.

| month | groups | rows moved | prior sd | mean shift | Heathrow | complete month | paired-day interval (s) | alignment | placebo (another month's figures) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| January 2025 | 182 | 16,398 | 27 s | 23 s | **+0.632%** | **+0.0275%** | [+0.045, +0.174] | 0.28 | +0.565% |
| July 2025 | 216 | 18,224 | 29 s | 20 s | −0.088% | −0.0106% | [−0.144, +0.060] | 0.60 | +0.603% |
| November 2025 (reported) | 186 | 16,455 | 15 s | 8 s | −0.056% | −0.0055% | [−0.025, +0.005] | 0.64 | +1.244% |
| December 2025 | 202 | 17,440 | 17 s | 9 s | −0.035% | −0.0046% | [−0.037, +0.018] | 0.59 | +1.098% |

**Not accepted:** January is worse (its interval lies above zero); the mean over the equal-footing months is +0.004%.

## Why it fails, and why it stays closed

The published figures carry real information: the placebo is worse than the read in every month by 0.6–1.3% of Heathrow's error, and
the moves in July, November and December point the right way (alignment 0.6). But the source's definitional offset moves from month
to month (0.10 to 0.44 minutes), by as much as the group biases the procedure looks for (sd about 0.3 minutes). In January 2025 the
offset was 0.44 against 0.25 in the other months, so every group read as 0.2 minutes too late and the month was pushed the wrong way.
That offset cannot be known label-free for a scored month, so a month can be hurt in the same way. The ceiling was 0.05–0.13% of a
month's squared error to begin with. Closed.

## Commands

```
.venv\Scripts\python.exe -X utf8 -B tools\run_caa_heathrow.py --verify
.venv\Scripts\python.exe -X utf8 -B tools\run_caa_heathrow.py --score --declaration configs/caa_heathrow_v1.json
```

The export path (`--export --version N [--base FILE]`) exists and refuses to run without the user's override, as the rule did not accept.
