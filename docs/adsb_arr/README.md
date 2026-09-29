# Arrival second task (29 September 2026)

Declaration: [`configs/adsb_arr_design_v1.json`](../../configs/adsb_arr_design_v1.json),
committed before any fit (651a2dc, 10:04 UTC). Report: [`design_report.json`](design_report.json). October only: no scoring
month, no ranking row and no official score is read. Nothing is exported or uploaded.

## The idea

The taxi-in time of every arrival is supplied, in the ranking file too. The plain committee network is trained on it as a
second task, with the arrivals of the month it has to predict among the training rows. Rows of the scored months then carry
a label the network may learn from, which the departure task never has.

## Design

- **Network.** The plain committee network (`nnb`, unchanged) plus one linear head for taxi-in
  (`src/prc2026_arr/model.py`). Base's parameters are created first, under the same seed.
- **Arrival rows** (`tools/build_arr_rows.py`). Each arrival enters the same 314 columns in the departure's role: the
  reporting airport is the arrival airport, the other end is the origin. 161 columns are populated by the frozen feature
  functions: the 15 categories, the calendar, record flags, the airport and runway traffic histories, weather and the
  takeoff-anchored surface state at the landing time. Every column that describes the departure's own clocks, its ADS-B trace
  or its stand history is missing. There is no task column: the missing-value flags identify the task.
- **Builder check.** The same code path on real departures reproduces the frozen builds: January, November and December
  exactly on all 161 columns; October on 160 columns exactly and one within 9e-16 (20 rows, platform rounding).
- **Preprocessing.** Numeric bins and the departure scaling come from the departure fitting rows only. The vocabulary is
  pooled over departure fitting rows and the variant's arrival rows.
- **Training** (`src/prc2026_arr/train.py`). Epochs, steps, learning-rate schedule and the departure batches are base's.
  Each step adds one arrival batch of 2,048 rows; loss = departure error + λ × arrival error, each in its own unit.
- **Arrival target.** Taxi-in clipped to [0, 3,600] s; LIRF arrivals are left out, as LIRF departures are.
- **Variants**, four seeds each: `mt_l025`, `mt_l05`, `mt_l1` (λ = 0.25, 0.5, 1; arrivals of the seven fitting months and of
  October) and `mt_l05_notarget` (λ = 0.5; arrivals of the fitting months only).

## Rule (declared)

GO for a development round if, for some variant and member weight b, the recipe proxy at network share 0.75 is at or below
−0.10% with its paired-day interval below zero, the variant beats the plain network read through the same grid, and its
standalone four-seed mean is below the plain network's. Otherwise the lever is closed.

## Result

**No-go. The lever is closed by its declared rule.** 16 fits (four variants × four seeds) and the nesting run on
arcw0–arcw7, 10:16–11:21 UTC; all eight workers deallocated afterwards; 136 run files mirrored and hash-checked.

**Nesting check.** The tool without arrival rows reproduces the plain network's seed-1 run exactly: the largest difference
over the 185,674 October predictions is 0.0 s. Whatever the variants show is the second task, not the code.

**Standalone (R1)**, October non-LIRF rows of the proxy table (169,734), four-seed means:

| member | RMSE (s) | against plain | paired seed difference (s), mean ± sd |
|---|---:|---:|---:|
| plain network (base, seeds 1–4) | 161.809 | | |
| `mt_l025` | 161.738 | −0.04% | −0.03 ± 0.54 |
| `mt_l05` | 161.922 | +0.07% | +0.12 ± 0.35 |
| `mt_l1` | 162.353 | +0.34% | +0.65 ± 0.53 |
| `mt_l05_notarget` | 162.158 | +0.22% | +0.51 ± 0.67 |

Single seeds of the plain network spread over 167.4–168.4 s. No variant differs from it by more than the seed noise; the
largest weight on the second task is the worst.

**Recipe proxy (R2)**, share 0.75, control 176.262 s, change against member weight b = 0:

| member | b = 0.125 | b = 0.25 | b = 0.5 |
|---|---:|---:|---:|
| plain network | +0.164% | +0.354% | +0.812% |
| `mt_l025` | +0.153% | +0.334% | +0.778% |
| `mt_l05` | +0.153% | +0.335% | +0.783% |
| `mt_l1` | +0.164% | +0.358% | +0.834% |
| `mt_l05_notarget` | +0.163% | +0.355% | +0.823% |

Every entry worsens the proxy, with its paired-day interval above zero (best entry: [+0.17, +0.40] s). The bar was −0.10%.
At share 0.85 the entries are +0.155% to +0.911%. A plain-type member of four seeds has nothing to add to the context
networks, with or without the second task.

**The decisive pair (R3).** `mt_l05` against `mt_l05_notarget`: paired seed differences −0.35, −0.12, +0.36, −1.48 s (mean
−0.39, sd 0.78); seed means −0.146% with a paired-day interval of [−0.99, +0.49] s. The month's own arrivals are not
distinguishable from none.

**What the network did learn.** The second head predicts October's taxi-in at 146–156 s RMSE with October's arrivals in
training and at 164 s without them (the fitting mean gives 291 s). The task is learned well; it does not carry over to the
departure head. This agrees with `docs/tta_screen`, `docs/arrivalctx` and `hour_effect.md`.

**One slice moves, and it is small.** On the 1,368 October departures whose stand, operator, aircraft type, runway or flight
prefix is outside the plain network's vocabulary, the variants with October's arrivals read 160.1–162.5 s against the plain
network's 169.6 s and 171.7 s without October's arrivals (−4% to −6%). Single seeds on this slice spread over ±8 s, and the
slice holds 0.9% of the squared error, so the whole effect is at most 0.05% of the network's RMSE. Label-free exposure in
the ranking file: 3,154 departures (0.91%) have such a category against the twelve 2025 months; January 0.28%, July 1.42%.

**New stands in July 2026 (label-free follow-up).** Counted by airport and stand, 3,271 ranking departures leave from a stand
with fewer than ten departures in 2025. Two cells hold nearly all of them. The EDDM group does not occur in 2025 or in
January 2026; the EDDF group holds 61 departures in all of 2025 and 26 in January 2026:

| cell | rows | v46 mean, new / other stands | what the inputs say |
|---|---:|---:|---|
| EDDF July, stands J1–J17, H9, H13, V333–V337 | 1,788 | 1,023 / 850 s | NM off-block clock (a real clock at EDDF): 1,012 / 921 s |
| EDDM July, stands 103–108 | 1,177 | 970 / 764 s | ADS-B taxi of gate-traced rows: 862 / 696 s; first-seen rows 947 / 641 s |

The file is higher on the new stands because the clock and the traces are: the longer taxi is observed, not guessed from the
unknown stand. Arrivals reach the new stands 106 s sooner than the others at EDDF and 69 s later at EDDM; landing and
take-off runways differ, so this does not contradict the file. One open point is too small to act on: at EDDF the file sits
72 s below the clock on known stands and 10 s above it on the new ones; if the new stands follow the known ones, that is
0.03% of the ranking's RMSE, and no development month can test it. **Closed.**

## Closed, and what is left of it

- The arrival second task is closed for the plain network. It is not repeated on the context network: the second task as
  such costs +0.2% here, and the context network already reads arrival taxi-in from its neighbours.
- Not built: the development runner. Arrival rows for 2025-01, 07, 11 and 12 exist locally (`runs/adsb_arr/rows`); no
  arrival row of a ranking month was built.
- The new stands were the next test this screen named; the label-free reading above answers it.
