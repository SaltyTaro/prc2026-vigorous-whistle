# Context network without positional ADS-B inputs (30 September 2026): no GO; the drift worry is closed for falling coverage

Declaration [`configs/adsb_ctx3_pos_v1.json`](../../configs/adsb_ctx3_pos_v1.json), 10:17 UTC,
committed before any fit (4104b0f; amended before any fit, 213728d). Report: [`design_report.json`](design_report.json)
(`tools/ctx3_pos_report.py`; predictions `tools/ctx3_pos_predict.py`). Nothing is exported or uploaded; no official score is read.

## Question

Another team reported that adsb.lol features encoding *where* an aircraft
was heard did not transfer between years, while features describing its *movement* did. Our coverage fell into the scored months
(EDDF gate share 38% in December 2025, 8% in January 2026, 4% in July 2026; `docs/adsb_coverage/`). Does a context network without the
first-point / coverage-anchored inputs predict better when its ADS-B inputs come from the scored months' coverage regime?

## Design

- **Variants** on the 32-epoch base (`ref32`), seeds 13–16, October design fold (fit 2025-02..06, 08, 09; LIRF excluded):
  - `nopos`: 9 of the 27 ADS-B columns removed (gate class, first-point time, speed and distance to stand, stationary time from the
    first point, silence before the segment, stand-anchor distance, points per segment, the neighbours' first-point median). 5 of
    the 11 v19 columns also go (appearance time, last point at the stand, pre-motion gap, imputed-stand distance and flag). The
    tokens' `first_seen` and `adsb_gate` are blanked.
  - `nocount`: the coverage-dependent counts removed (3 neighbour counts, 5 surface counts).
  - `nopos_nocount`: both.
  - Every movement quantity stays: motion start, taxi path, stops, stopped time, slow start, last ground point, turnaround, the
    neighbours' taxi times, first motion.
- **Transfer reading (R3).** July and December 2025 lie outside the design fit. Each variant's four-seed mean predicted them twice:
  - with the tables and tokens the fits read;
  - with coverage-regime attempt a13 (`docs/coverage_regime/`), the scored months' coverage simulated by removal at EDDF (both
    months) and LSZH (July). This used the thinned committee and v19 tables, plus token files rebuilt from the thinned matches
    (`data/ctx/tokens_thin_a13/`). The natural July rebuild is byte-identical to `tokens_v1`.
- **Rule.** An item is GO when:
  - (a) its pooled thinned-row RMSE is below ref32's, with the paired-day interval below zero;
  - (b) both whole months are better on thinned inputs;
  - (c) October R2 at b = 0.5 is at most 0.02 points worse.
- **Hardware.** CPU, arcw0–7, two lanes per worker, 10:42–14:29 UTC, 3.2–3.7 h per fit. Each worker was mirrored, hash-checked and
  deallocated as soon as its two fits ended; all eight were deallocated by 14:31 UTC. Scoring ran on the workstation.
- **Scorer check.** Before any July or December label was read, every run's October re-prediction matched its own stop predictions:
  RMS 0.0001 s, at most 0.004 s.

## Result: no item is GO

October non-LIRF RMSE of the four-seed means. Thinned rows: EDDF July and December, LSZH July (non-LIRF). Positive changes are worse.

| variant | October | R2 at b = 0.5 | thinned rows, natural → thinned (penalty) | thinned rows vs ref32 (interval, s) | whole month on thinned inputs vs ref32 (Jul / Dec) | GO |
|---|---:|---:|---|---|---|---|
| ref32 | 155.31 | −0.131% | 120.86 → 144.55 (+19.6%) | | | reference |
| `nopos` | 158.87 (+2.29%) | +0.127% | 127.96 → 151.27 (+18.2%) | +4.65% [+5.18, +8.51] | +1.35% / +2.57% | no |
| `nocount` | 156.47 (+0.75%) | −0.003% | 122.60 → 146.05 (+19.1%) | +1.04% [+0.14, +2.95] | +0.57% / −0.15% | no |
| `nopos_nocount` | 158.31 (+1.93%) | +0.079% | 129.54 → 151.92 (+17.3%) | +5.10% [+5.63, +9.52] | +1.56% / +2.05% | no |

- **Positional inputs still help in the thinned regime.** EDDF on thinned inputs reads 128.1 s (ref32) against 138.0 s (`nopos`)
  in December, and 160.3 against 167.7 s in July. Removing them costs more on natural inputs (+13.7% at EDDF in December) than it
  saves in penalty.
- **The penalty is almost the same with and without them:** +19.6% for ref32, +17.3% to +18.2% without them. The loss from thinning
  is lost information, not features that change meaning. This matches the coverage study's finding for the residual stage.
- **Counts:** `nocount` is close to neutral in December (−0.15% on thinned inputs) but worse in July and in October.
- Natural tables against the coverage study's own natural tables: prediction RMS 0.02–0.15 s (float-level stand distances and a
  handful of neighbour values), negligible.

## What this closes, and what it does not

- Closed for falling coverage: dropping positional or count inputs does not make the context network transfer better to the scored
  months' EDDF / LSZH regime. No development round; the shipped networks keep their inputs.
- Not covered: airports whose coverage **rose** in July 2026 (EGLL, EDDM, LEMD). Removal cannot simulate that. The same
  mechanism argues against a problem there: a matched row in 2026 is read with features whose meaning did not change here.
- Labels read: October, July and December 2025, as declared. No ranking row, official score or published aggregate.

## Incidents (all disclosed; none touched a fitted model)

- The first smoke fit (10:27 UTC, 30,000 rows) failed while loading: the v19 tables load whole, so a shortened `extra_features`
  list was rejected. The fix routes v19 drops through `features.drop`. The declaration was amended before any fit; the inputs
  per variant are unchanged.
- At the user's word it was restarted one model per process with a free-memory check, and all 16 completed.
- Monitors: three pollers (progress, health, quality). The quality rule first compared fits across variants and flagged
  `nopos_nocount_s13` at epoch 2 (a slow-starting seed common to all variants). It was narrowed to the same variant from epoch 8.
  Nothing else was flagged. Compute: Azure CPU only.
