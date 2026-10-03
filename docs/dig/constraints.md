# Dig tunnel 5: other cross-row hard constraints (3 October 2026): no GO

After the intruder stand bound (v58 = 220.9212 official), this tunnel looked for further logical constraints between rows of the supplied
data. 2025 only. Nothing declared, exported or uploaded; no official score or leaderboard read.

- **Base:** `runs/dig/constraints/base.py` applies the accepted intruder layer month-blocked (offsets fitted on the other months), so
  every reading below is incremental over it. It reproduces the governing run, for example dev January 322.059 → 321.513.
- **Validity:** the share of candidate rows whose 2025 label satisfies the bound within 60 s, over all twelve months.
- **Screens:** as `tools/run_intruder_bound.py`, on the twelve-month table and the dev chain proxy.
- **Protected rows:** LIRF_pop and the guard rows never move.

| # | constraint | validity / binding | verdict |
|---|---|---|---|
| 1 | minimum turnaround after our inbound (inbound = the last arrival at our stand of our type, or the one before a different-type intruder) | true turnaround is negative for 0.7–7.1% of the identified pairs (same-type arrivals after we left); the predictions place those rows as the truth does (predicted turnaround < 600 s on the same share); clamping rows predicted below the 1% turnaround quantile multiplies their squared error (e.g. EGLL 2.5e8 → 5.1e8) | closed (not a bound; already respected) |
| 2 | departure order at a stand (our off-block before the next different-type departure's off-block) | valid 99.9–100% everywhere, but consecutive departures from a stand are hours apart (median 3 h); predictions violate it on 160 rows in a year, and moving them is worse (3.2e6 → 7.8e6 at best) | closed (never binds) |
| 3 | runway take-off order as push-back order (FIFO) | true FIFO on consecutive same-runway departures is only 61–77% (66–87% from the same pier); the predictions reproduce the same rates | closed (not a constraint) |
| 4a | identity by operator: a same-type arrival of another operator is not our aircraft | validity 2.6–74% (EGLL 20%, LTFM 59%, LFPG 74%): codeshares, wet leases and regional subsidiaries carry other prefixes | closed |
| 4b | identity by count: two arrivals at our stand since the previous departure's take-off, so the second came after we left | validity 0.6–33% | closed |
| 4c | identity by flight-number rotation (A pairs with the next departure from our stand, not with us) | validity 0–28% | closed |
| 4d | same aircraft departing twice in a day | reduces to the inbound / turnaround bound (row 1), already respected | closed |
| 5 | MARS stand families (we at X+L / X+R and a different-type arrival at the centre X, or the reverse) | the failed airports (EDDM, EHAM, LSZH, LEMD, LEBL) have almost no suffix families; where the families are real: EGLL 21 / 21 and LTFM 45 / 45 valid, LEBL 6–9% (its suffixes are not MARS) | valid but small, below the bar |

## MARS screen (constraint 5)

`runs/dig/constraints/c5_screen.py`.

- **Airports:** chosen by fitting-month validity (≥ 90%, ≥ 10 rows), which gives EGLL and LTFM.
- **Value:** B plus the accepted intruder model's offset for the airport and gap band, fitted on the fitting months.
- **Increment:** over the accepted intruder layer.

| table | mean change | months better | rows | detail |
|---|---|---|---|---|
| twelve months (older predictor) | −0.0213% | 12 / 12 | 66 | February −0.075%, July −0.051%, June −0.045% |
| dev chain proxy | −0.0127% | 3 / 4 | 18 | January −0.0075, July −0.0379 [−0.079, −0.008], November +0.0009 (one row), December −0.0064% |

This misses the −0.05% bar. It is the same physics as the accepted layer and could ride along with it in a later version together with
other small riders, but it is not worth a declaration on its own.

**Reading.** The stand-occupancy information the shipped chain does not use is the intruder rule itself. The intruder rows are visible
only through a different aircraft type. Identity proxies (operator, rotation pairing, arrival counts) are too noisy to extend the rule to
same-type intruders. Turnaround, stand order and runway order are either respected already or not constraints.
