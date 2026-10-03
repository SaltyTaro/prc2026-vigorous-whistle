# Dig tunnel 20: case files on the worst over-predictions of the v63 chain (3 October 2026) — closed, no GO

Diagnostic, 2025 only. Labels are read only to pick and classify rows and to score screens. No official score, no leaderboard, no
upload, nothing under `runs/submissions`, nothing declared. Scratch: `runs/dig/case_files3/` (`build.py`, `context.py`, `dump.py` →
`dump.txt`, `classify.py` → `mechanisms.json`, `layer_audit.py` → `layer_audit.csv`, `trig_a.py`).

Control: the v63 development proxy (`runs/stack_resid/dev_v63_proxy.parquet`, column `v63`), with each chain layer's delta
reconstructed from `runs/scout/refit/ctl_v55m.parquet` (model part → v48 → extreme-day → var-alloc), `runs/dig/census61/dev_v61.parquet`
(rematch → stacking v2 → intruder → identity) and the proxy's own `v62` / `v63` columns (`dev_layers.parquet`).

## Where the over-prediction error is

Ordinary rows (label ≤ 3 h, outside the LIRF population) hold 55.2% of all development squared error; over-predictions hold 37.1% of
that. Over-predictions beyond 600 s are 2,281 rows and 9.3% of ordinary squared error; none exceeds 3,600 s (under-predictions carry the
large tails).

## The 200 rows

170 from the four development months (largest over-prediction against v63; 2.6% of ordinary squared error) and 30 from the other
2025 months (against the older twelve-month predictor, `rows12.stack`). For each: every supplied field and clock interval, the
layer deltas, the stand's arrival and departure sequence, the runway's take-off sequence, the published day discrepancy and the
identified aircraft's own adsb.lol timeline (identity tables of tunnels 10/11; LIRF and LTFM identified on the fly by callsign).

| mechanism | rows | share of the rows' squared error | label-free trigger? |
|---|---:|---:|---|
| LIRF clock row: the stack hedges toward the schedule interval (fill reading) and the row is not a fill (ETH / UAL / DAL / ACA long-haul, WMT, ITY …) | 74 | 34.0% | no: the fill class is unobservable; `docs/clean_line_screens/lirf_mechanism.md` (stack calibrated along the class probability) |
| delayed flight: prediction near the schedule interval, the recorded block later than schedule (EGLL 33, LFPG 17; BA rows with an estimated NM clock at take-off − 600 s) | 67 | 31.0% | screened below: unbiased on its trigger, closed |
| unmatched, no signal (overshoot within the cell's noise) | 27 | 21.4% | no |
| late record: label ≤ 300 s, block written at take-off (LFPG 9, LEBL 5) | 18 | 7.2% | no: small LOBT intervals do not mark them (below) |
| ADS-B motion well before the recorded block (tow, reposition, de-icing pad) on dark rows | 9 | 4.3% | no (tunnel 15) |
| bound-layer misfire (own-motion / identity / intruder moved the row up past the label) | 5 | 2.0% | sized below, below the bar |

The over-predictions are the mirror image of the known label classes: where the class is unobservable the chain hedges, and the
rows of the other class pay the hedge. No new mechanism recurs.

## Screens

**A. Delayed flights anchored near the schedule interval** (trigger: schedule interval ≥ prediction − 300 s, interval > 1,800 s, and
the NM clock missing, more than 600 s below the prediction or an estimate at take-off − 600 ± 15 s; LIRF excluded). 21,916 development
rows (25.9% of squared error outside LIRF). The v63 bias on the trigger is near zero and changes sign between months (EGLL −1 / +13 /
−46 / −51 s; LFPG +72 / +14 / +35 / +7 s). An in-sample oracle bias fix per schedule band is worth 0.003% / 0.001% / 0.030% of
squared error, so no month-blocked correction can reach the −0.05% bar. The EGLL estimated-clock subset (1,165 rows) has bias −57 s:
0.01%. Ranking triggers on v63 (read only): 13,857 rows. **Closed.**

**Late records.** Rows whose LOBT lies within 700 s of take-off and whose prediction sits more than 600 s above it: the label is
within 300 s of the LOBT interval for 0-3% of them at every airport (LEBL 1,237 rows, 3%). A short LOBT interval does not mark a late
record. **Closed.**

**Layer overshoot audit** (`layer_audit.csv`): every chain layer split by airport × direction × size band of its move, realised gain
against squared change. The largest cost cell is v48's upward 300-1,000 s moves at LIRF (43 rows, −0.016% of development squared
error); by airport × direction no layer costs more than 0.002% (v48 EHAM up −0.0017%, LEMD up −0.0003%; var-alloc EDDF down
−0.0021%). Layer totals are all gains: intruder 1.22%, v48 0.60%, v63's stacker 0.31%, stacking v2 0.30%, var-alloc 0.20%, own-motion
0.20%, identity 0.18%, rematch 0.08%, extreme-day 0.07%. **No layer overshoots on a label-free subset.**

**Bound-layer misfires.** Rows each bound layer made worse: own-motion v1 99 of 327 (−0.033% of squared error, EDDF −0.030%),
identity 7 of 42 (−0.011%), intruder 6 of 120 (−0.019%). Most own-motion losses are valid bounds with the label between B and B +
offset (EDDF: 90 losing rows, 10 invalid). The 10 invalid EDDF rows (about −0.027%) are not separable: over all 2025 trigger-like
rows the bound is valid 98.8-100% in every band of the gap after the first move and of its speed (`B2gap`, `gs`), and tunnel 17
already screened the persistence filter. **Closed.**

## Verdict

No GO. The over-prediction side holds no new label-free mechanism: its large errors are honest hedges of the unobservable LIRF fill
and delayed-flight classes, late records written at take-off, and a few bound misfires that no label-free condition separates. No
shipped layer overshoots on a label-free subset.
