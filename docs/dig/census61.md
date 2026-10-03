# Dig tunnel 14: census of v61 and the next hypotheses (3 October 2026, 2025 only, nothing declared)

No official score, no leaderboard, no upload, nothing under runs/submissions. Scratch `runs/dig/census61/` (`build.py`, `census.py`,
`screen_h12.py`; `dev_v61.parquet`, `census_rows.parquet`, `steps.json`).

## The v61 development proxy

`ctl_v55m.new` (the shipped chain proxy) → no-NM rematch (tunnel 1's dev candidate, `runs/rematch_nonm/dev/cand.parquet`) → stack_resid_v2
delta (`runs/dig/stack3/base_v2.parquet`, base2 − new, added on the rematched chain) → accepted intruder layer (month-blocked,
`tools/run_intruder_bound.py`) → identity extension (month-blocked, `tools/run_intruder_identity.py`). The rematch and the stacking base share
`ctl_v55m.new` exactly (max difference 0.0). LIRF_pop and the two guard rows keep their value.

| step (change vs the previous step) | January | July | November | December | recorded reading |
|---|---:|---:|---:|---:|---|
| rematch | −0.0501% | −0.0194% | −0.0109% | −0.0213% | identical |
| stacking v2 | −0.0443% | −0.1249% | −0.1801% | −0.3283% | −0.0445 / −0.1249 / −0.1801 / −0.3287 (on the unrematched chain) |
| intruder bound | −0.1692% | −1.1601% | −0.2278% | −0.1918% | −0.1694 / −1.1591 / −0.2291 / −0.2036 (on the chain without stacking) |
| identity extension | −0.0603% | −0.0923% | −0.0115% | −0.1350% | −0.060 / −0.091 / −0.009 / −0.133 |
| **v61 proxy RMSE (s)** | **321.017** | **252.427** | **183.215** | **165.125** | ctl_v55m 322.059 / 255.995 / 184.005 / 166.247 |

The small step differences come only from applying each layer on the preceding one (the intruder candidate set depends on the prediction).

## Census (v61 proxy against the v46 development stack)

- Pooled development RMSE 237.59 s against 240.48 s (−1.20%); ordinary-label (<= 3 h) squared error −3.64%.
- **Tails:** the 78 labels above 3 h hold 35.8% of all squared error (v46: 35.0%); the LIRF population alone holds 16.8% of all squared
  error (13.7% of ordinary). Ordinary labels hold 64.2%.
- **Ordinary squared error by airport × ADS-B state** (v61 share; v46 share):

| cell | rows | RMSE v61 (s) | share v61 | share v46 | bias v61 (s) |
|---|---:|---:|---:|---:|---:|
| LIRF matched, not gate-traced ("dark") | 27,007 | 436.9 | 21.2% | 20.5% | −6.6 |
| LTFM (all unmatched) | 91,338 | 202.3 | 15.4% | 15.7% | +3.4 |
| LFPG unmatched | 64,894 | 233.2 | 14.5% | 14.5% | −1.1 |
| LIRF unmatched | 18,623 | 374.0 | 10.7% | 10.4% | −4.4 |
| EGLL unmatched | 32,852 | 223.0 | 6.7% | 7.1% | −13.7 |
| EGLL dark | 35,605 | 181.4 | 4.8% | 5.2% | −6.0 |
| LEBL dark | 37,016 | 156.3 | 3.7% | 3.7% | +6.4 |
| LEMD unmatched | 29,661 | 158.8 | 3.1% | 3.1% | −4.1 |
| every other cell | | | below 2.6% each | | |

  Shares are of a smaller total; in absolute terms every large cell fell (LTFM −4%, EGLL unmatched −9%, LFPG unmatched −3.6%).
- **Label classes** (diagnostic, by the hidden block time against the NM clock; ordinary, non-population): the **early** class (block more
  than 300 s before the NM clock) holds **31.7%** of ordinary squared error over 69,529 rows (bias +143 s): LFPG 7.8%, LIRF 7.2%, LTFM 5.5%,
  EGLL 3.6%, LEBL 2.2%, LEMD 2.1%. LIRF schedule fills 7.6% (bias +140 s). The **late** class (block more than 300 s after the clock) is
  over-predicted: LTFM 3.8% (bias −71 s), EGLL 2.9% (−139 s), LFPG 2.5% (−157 s). "Agree" rows carry the rest.
- **No NM record:** 1.2% of rows, 19.8% of ordinary squared error (v46 20.0%); outside the LIRF population 6.1%.
- **|error| > 600 s:** 1.15% of ordinary rows hold 52.5% of ordinary squared error; 67.7% are under-predictions. Largest cells: LIRF dark
  17.0%, LIRF unmatched 8.1%, LFPG unmatched 6.7%, LTFM 4.9%.
- **Hour structure:** the airport-day-hour mean residual explains 7.95% of ordinary non-population squared error against 3.91% after a
  within-airport-month shuffle (v46: 9.13% / 4.03%): the excess fell from 5.1 to 4.0 points; today's stacking/published layers took about a
  fifth of it.
- **Days:** the eleven largest airport-days are LIRF July 2025 days (13, 28, 4, 27, 2 July: 0.9-2.5% of ordinary squared error each) and
  LFPG 22 November; the hour-of-day profile is unchanged from v46.

**Reading.** Today's layers removed error where hard information existed (stands, ADS-B matching keys) and part of the day/hour structure.
What remains is dominated by three mechanisms no input observes: the early/late label classes against the NM clock (~38% of ordinary
squared error), the LIRF schedule-fill population and its tails (~17% of all), and the shared hour state (4 points).

## Five hypotheses for the remaining cells

| rank | hypothesis | mechanism and evidence | bound | why new | screen | result |
|---|---|---|---|---|---|---|
| 1 | **Exact-identity reactionary bound** (EDDF/EGLL/LFPG) | a departure leaves its stand after its own aircraft blocked in: label <= take-off − own inbound in-block (identity from tunnel 10's ICAO table, not stand/type proxies) | rows where v61 breaks it | tunnel 5 tested the turnaround bound with a type-based inbound (0.7-7.1% wrong) | validity and violations on the dev proxy, month-blocked offset | **closed**: valid on 100% of 422,680 identified departures, but v61 breaks it on 1 dev row (−0.003% of December) |
| 2 | **Aircraft-level residual history** | an individual aircraft (ICAO address) carries a persistent residual (handler / APU / door procedures) beyond operator and type | cell oracle on identified rows | flight-number history (29 Sep) and group biases used other keys | month-blocked shrunk mean residual of the same ICAO from the other dev months, k 20/50, w 0.5/1 | **closed**: worse in every month and setting (+0.007% to +0.060% equal-footing) |
| 3 | mixture-aware allocation of the day discrepancy by P(early) × expected early jump | the published total sees the early class as missing delay; allocating by the class jump instead of variance | <= the 1-dof-per-day ceiling | tunnel 4's leftover; v57 allocates by learned variance, v60's stacker sees the discrepancy | replace v57's weight by P(early) × lift, month-blocked | not run: v57's variance and v60's stacker already track the mixture; expected below 0.05% |
| 4 | late-class shrink toward the NM clock on high-lift rows at EGLL/LFPG/LTFM | over-predicted late rows (bias −71 to −157 s, 9.2% of ordinary squared error) | cell oracle 9.2% | the class itself was screened only at LIRF/LFPG for "early" | lift-band shrink, month-blocked | not run: the LIRF screen showed the stack is calibrated along its lift in every band; a shrink would trade early rows for late ones |
| 5 | cross-day composition regression on the 2026 published totals (stand-group early propensity per airport-month) | days with more early-prone stands should show more "missing" delay | 1 dof per airport-day, ~31 per month | group day slopes (scout §17) regressed on I_d within days, not across-day composition | ridge of day discrepancies on day composition, month-blocked | not run: composition varies little across days; identifiability ~nil |

**Verdict:** the two concrete new hypotheses close at once (the chain already satisfies the exact identity bound; aircraft history has no
persistent signal). No GO. The largest remaining error is the unobservable label class (early/late against the NM clock) and the LIRF
population; a further lever needs a new observation of when the airport records off-block, not a new model.
