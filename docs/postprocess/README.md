# Extreme-row post-processing of v7 (15 September 2026, v8 scored 283.668; v7 retained)

Declared at **04:12 UTC on 15 September 2026** in
[`configs/postprocess_v1.json`](declaration.json), after a same-day diagnostic
review of the reconstructed v7 residuals and before any declared candidate was
scored. This campaign fits **no native model**: it applies three fixed rules to
the frozen v7 ranking predictions, with the frozen v5 no-clock pipeline as the
routing fallback.

**Selected: `floor_route_hedge`**, scoring **319.6600 seasonal / 215.5122 forward**
against the exactly reconstructed v7 development scores 322.8193 / 221.1382
(**−0.979% / −2.544%**). `runs/submissions/vigorous-whistle_v8.parquet` changes
**197 of 344,841** rows: 169 routed, 11 hedged, 17 floored; every other row
reproduces the uploaded v7 integer exactly.

**Official outcome: v8 was uploaded at 04:30:58 UTC on 15 September with the
user's approval, processed at 04:31:36 UTC, and scores 283.668 seconds — 8.611 s
(3.13%) worse than v7.** V7 remains the team's official best at 275.0570; the
leaderboard ranks each team by its best submission, so standing is unchanged.
The score change equals a 1.659e9 increase in total squared error. If every one
of the eleven hedged rows is a schedule fill (label equal to its schedule
interval), the hedge alone accounts for 1.248e9 of that; the remaining 0.41e9
is not attributable to a rule from the score alone. The dev-fold evidence for
the hedge rested on 11 + 2 scoring rows and was flagged as high-variance before
upload; the ranking set evidently does not follow the fitting-period one-day
share. **No row-level rule is derived from this score difference.** See
[`official_result.json`](official_result.json) and
[`submission_remote_verification.json`](submission_remote_verification.json).
The declared runner-up `floor_route` (318.5519 / 220.7813) contains no hedge
and remains available as a separately submittable candidate.

## What the diagnostics found

The complete-fold SSE is concentrated in a few hundred rows: 75 rows with
|residual| above 5,000 s hold 44% of the seasonal SSE and 7 rows hold 17% of
the forward SSE. Three label-generation mechanisms explain most of it:

1. **LIRF schedule fill.** For LIRF departures with no NM record and no
   `AIRCRAFT_TYPE_mvt`, `BLOCK_TIME_UTC_mvt` equals `SCHED_TIME_UTC_mvt` (±5 s)
   in about 46% of rows, so the label equals `MVT − SCHED`; the rest are
   ordinary taxi-outs. The modes are not separable from ranking-visible columns
   (AUC ≈ 0.87 using delay and local hour). V7's Phase-4 offset residual
   already carries most of the attainable value; a dedicated mixture model was
   worse on the seasonal fold and is not part of this campaign.
2. **One-day rows.** The block date is attributed to the scheduled date, so the
   label is 86,400 s plus an ordinary taxi-out. Inside the fill population,
   fitting rows with a schedule interval above 43,200 s are **never** ordinary:
   17 schedule fills and 10 one-day rows in January–November. Predicting one
   such row as ordinary costs about 7.5e9 SSE, roughly 29% of v7's entire
   ranking SSE.
3. **NM clocks that postdate takeoff.** Where `AOBT_3_flt` (else `LOBT_flt`)
   lies after the movement time, the residual model emits negative predictions
   (v7 ranking minimum −3,948 s). No departure label in January–November lies
   below −12 s.

`AOBT_3_flt` is a real off-block clock at EDDF, LIRF and LSZH (label slope ≈ 1)
but an estimate at EDDM (correlation 0.09) and LTFM (0.22). A per-airport
direct/residual re-weighting from the saved nmclock runs gave at most 0.2%
honest cross-fold gain and is not pursued. The ranking set also contains a
disruption absent from 2025 (EHAM 2–7 January 2026 and LFPG 7 January 2026:
arrival taxi-in mean 2,050 s on 5 January against ~600 s normally); no rule
here addresses it because the development folds cannot score one.

## Rules and parameters

Applied in declared order to every ranking departure; parameters come from the
fitting departures of the fold (refit months for the export) and never from
scoring rows.

| Rule | Population | Replacement | Refit parameters |
| --- | --- | --- | --- |
| `route_negative_proxy` | NM clock present, `MVT − clock < 0` | v5 no-clock pipeline prediction | none |
| `hedge_long_schedule` | LIRF, no NM record, no aircraft type, `s = MVT − SCHED > 43,200` | `(1−p)·s + p·(86,400 + m)` | p = 10/27 = 0.3704, m = 961 s |
| `floor` | prediction below the fitting minimum label | the minimum label | −12 s |

Fold parameters: seasonal p = 3/14, m = 995; forward p = 8/25, m = 847;
floor −12 on both. The long-schedule group contains zero ordinary rows in every
fitting period.

## Development results

| Candidate | Seasonal | Forward | Mean ratio | Paired-day 95% (seasonal / forward) |
| --- | ---: | ---: | ---: | --- |
| v7 (reconstructed) | 322.8193 | 221.1382 | 1.0000 | — |
| `floor` | 318.9008 | 220.8866 | 0.9934 | [−8.18, −0.93] / [−0.74, −0.00] |
| `floor_route` | 318.5519 | 220.7813 | 0.9926 | [−8.50, −1.21] / [−1.10, +0.39] |
| `floor_hedge` | 320.0077 | 215.6200 | 0.9832 | [−7.05, +0.49] / [−13.17, −0.07] |
| **`floor_route_hedge`** | **319.6600** | **215.5122** | **0.9824** | [−7.39, +0.14] / [−13.26, −0.06] |

The hedge gives back about 1.1 s on the seasonal fold, whose January/July
long-schedule rows are 6 fills to 5 one-day rows, and gains 5.3 s on the forward
fold from two November one-day rows. The declared rule selects the lowest mean
ratio among candidates not worse than v7 on either fold. These are reused
development folds, not an independent audit.

## Export and verification

- Export: [`export.json`](export.json); file SHA-256
  `2c3727b3ab90fa204bb4a16e0678043432f64d60f86b96d376aee34f973bc974`,
  344,841 rows, exact template order and types, range −12 to 102,657 s,
  18 rows below zero (all at or above −12). The eleven hedged rows are listed
  in the export report; the largest single change lowers the LIRF row with a
  111,654 s schedule interval from 120,549 to 102,657.
- Independent verification: [`verification.json`](verification.json) recomputes
  every rule from raw columns and the frozen parents without the rule package,
  confirms untouched rows equal v7, and reproduces the refit parameters from
  fitting labels.
- Selection receipt: [`development_results.json`](development_results.json),
  SHA-256 `15346f96e38bcb5713f232a9cd9ff8f8d397d788f936efbc2d2eaecef0384fc3`;
  model-code SHA-256 `d476d0c7d200b7be8f2195d770945b425457e7a9b2542cac413037ea258f9239`
  (`src/prc2026_postprocess/rules.py` and `tools/run_postprocess.py`).
- Six new tests in `tests/test_postprocess.py`; the full suite record is
  `tests_local.xml`.

Source is isolated in `src/prc2026_postprocess/`. Tools:
`tools/run_postprocess.py` (selection), `tools/export_postprocess.py --version N`
(export), `tools/verify_postprocess.py --version N` (independent check).
No Azure worker, external data, December label or leaderboard probe was used.
