# LTFM IOBT-convention sensitivity (27 September 2026). Descriptive; nothing selected, exported for upload, or uploaded

**Question.** In July 2026, LTFM's 10-minute early IOBT disappeared (see [`README.md`](README.md), finding 2). How much does that move
v30/v31, and which way is right? This is the analogue of the [LFPG EOBT_1 sensitivity](lfpg_eobt_sensitivity.md) behind v31.

## Step 1: the convention, label-free

`tools/ltfm_iobt_convention.py` → [`ltfm_iobt_convention.json`](ltfm_iobt_convention.json). It reads schedule, NM clock and identity
columns only.

- **Only THY uses it.** In every month from January 2025 to January 2026, 13–21% of LTFM departures have IOBT = SCHED − 600 s. Among
  THY rows with IOBT ∈ {SCHED − 600, SCHED}, the early share is 0.20–0.33 in every month (0.285 pooled). In July 2026 it is 0.001.
  Other carriers are at 0.2%.
- **It is not a property of the flight number.**
  - Of 2,787 LTFM flight numbers with ≥ 5 earlier departures, 828 never carry it, 1,911 carry it on 0–50% of occurrences, 47 on
    50–80% and 1 on > 80%.
  - July 2026 has 283 departures from numbers at ≥ 50% (n ≥ 5), and 21 (one number) at ≥ 80%. The 2025 rate would predict about
    4,900.
  - Callsigns give the same picture.
  - Per-number shares are stable: the correlation between the first and second half of 2025 is 0.80. A number with a 2025 share
    ≥ 0.5 carries it on 49% of its January 2026 departures; a number below 0.2 carries it on 6%.
  - So the row that would have carried the convention cannot be named. It can only be weighted.
- **THY's callsigns changed in 2026.** 9,183 of the 16,618 July rows below carry an alphanumeric callsign that never appears from
  January 2025 to January 2026; in July 2025 it was 131 of 19,289.

**Declared rule** ([`configs/ltfm_iobt_sensitivity_v1.json`](../../configs/ltfm_iobt_sensitivity_v1.json), declared 08:58:02 UTC,
SHA-256 `0b7f551e…`). It was fixed after the table above and before any prediction effect.

- **Restoration.** On the **16,618** July-2026 THY LTFM departures with IOBT = SCHED: IOBT := SCHED − 600 s. EOBT_1 and LOBT :=
  SCHED − 600 s where they equal the original IOBT (16,005 rows each; the other 613 have EOBT_1 ≠ IOBT and keep it). Departure rows
  only, as for LFPG.
- **Row weight.** q is the chance the row would have carried the old convention. Per flight number over January 2025 to January
  2026: q = (e + 20·p0) / (e + s + 20), with e the early rows, s the IOBT = SCHED rows and p0 = 0.2847. Unseen numbers get p0.
  - July 2026: mean q 0.296, about 4,920 expected old-convention rows.
  - Checked label-free on January 2026 with q fitted on 2025 only: Brier score 0.195, against 0.209 for the constant p0. Calibration
    by q bin (mean q → observed share): 0.16 → 0.13, 0.26 → 0.24, 0.34 → 0.34, 0.44 → 0.50, 0.61 → 0.56.
- **Hedge.** v31 + w · q · Δ, where Δ is the restored minus the original recipe v30 on each row. On LTFM rows v31 = v30.

## Step 2: the restored ranking copy through the final-file runner

**Input.** `tools/ltfm_iobt_input.py` → `runs/final_ltfm_iobt/input/final.parquet` (SHA-256 `05c1aa07…`) and `modification.json`.
Exactly 16,618 rows change, and every other column is checked to be identical.

**Run.** `tools/final_predict.py run --root runs/final_ltfm_iobt/run --threads 12 --workers 8 --version 902 --detail-ensembles
--vocab-cache runs/final_dryrun/run/leaves/vocabularies.json` ran from 08:58 to 10:37 UTC, 1 h 39 min.

- Stage times: features 189 s, adsb 604 s, v9 67 s, leaves 484 s, networks 4,233 s, residual 376 s.
- Free RAM was 0.7 GB at the features peak and 2.7–5.2 GB at the later checks.
- The export `runs/final_ltfm_iobt/run/export/vigorous-whistle_v902.parquet` is a sensitivity artifact only.

**Comparison.** `tools/ltfm_iobt_sensitivity.py` → [`ltfm_iobt_sensitivity.json`](ltfm_iobt_sensitivity.json), against
`runs/final_dryrun/run` (same code, models and machine). The per-row table is `runs/final_ltfm_iobt/delta.parquet`: q, Δ and q·Δ on
every LTFM July row.

- **Exactly zero outside LTFM July 2026.** Max |Δ| is 0.0 on all 319,656 other rows at every stage. That holds for all 39 changed
  feature columns and the export integers.
- **Features.** The IOBT/EOBT_1/LOBT columns (`nmclock_aobt_minus_iobt_s`, `nmclock_aobt_minus_eobt1_s`, `movement_minus_*`,
  `*_minus_schedule_s`, planned durations, `nmclock_clock_difference_s`) shift by exactly ±600 s on the edited rows.
  `nmclock_agree_60s` flips on 3,633 rows. The `gd_*_eobt_*` counts change on 21,100–22,400 LTFM July rows.

Mean per-row change (restored − original, s; RMS in brackets):

| stage | edited, IOBT + EOBT_1 + LOBT (16,005) | edited, IOBT only (613) | unedited LTFM July (8,567) |
|---|---:|---:|---:|
| v7 / v9 | +31.6 (46) | +14.8 (38) | 0 |
| committee trees `ens_v20_bag_q_all` / LIRF `ens_v22_lirf_q` | +27.9 / +27.7 | +13.7 / +14.2 | 0 |
| context nets e24 / e24b / e32 / lirf | +51 / +53 / +49 / +65 | +51 / +58 / +53 / +80 | **+41 / +42 / +39 / +55** |
| tabular nets nn / nnl / new | +23 / +23 / +24 | +9 / +15 / +8 | 0 |
| residual stage | 0.0 (0.1) | 0 | 0 |
| committee mix | +44.9 (55) | +44.0 (57) | +30.5 (43) |
| **v30** | **+40.9** (50), max 596 | **+35.2** (47) | **+21.3** (30) |

- **Direction.** Every stage raises the restored rows. The context networks also raise the unedited rows through their neighbour
  inputs. The full restoration makes 66% of LTFM's July departures "early", against 17–20% in 2025, so the context effect is an
  overshoot; scaled by the mean q it returns to the 2025 level.
- **Full effect.** Σ Δ² over the edited rows is 4.07e7, 0.22% of v31's ranking SSE.
- **The declared soft restoration q·Δ** on the 16,618 rows:
  - mean **+12.0 s**, RMS 15.2 s, p99 +43 s, max 170 s;
  - Σ (q·Δ)² = 3.82e6, 0.021% of ranking SSE;
  - rows with a seen callsign: +12.7 s (RMS 16.7); unseen callsigns (q = p0): +11.4 s (RMS 13.8).
- **Δ by the original AOBT_3 − IOBT:** +28 s below −600 s, rising to +61 s at 300–900 s, then +48 to +58 s beyond.

Largest |q·Δ| rows. Announced delay EOBT_1 − SCHED is 0 on all ten; they are late-evening THY departures from numbers with q
0.6–0.8, plus two delayed flights:

| flight | SCHED (UTC) | q | Δ v30 | q·Δ | v30 | AOBT_3 − IOBT | MVT − SCHED |
|---|---|---:|---:|---:|---:|---:|---:|

## Step 3: which way is right, 2025 bounds

`tools/ltfm_iobt_hypotheses.py` → [`ltfm_iobt_hypotheses.json`](ltfm_iobt_hypotheses.json).

**The 2025 anchor.** THY rows with IOBT ∈ {SCHED − 600, SCHED}, labels ≤ 3 h:

- Unconditionally the two types have the same labels: 1,047.7 s (early) vs 1,048.2 s (sched) in the recipe months.
- Early rows take off about 115 s earlier against schedule (median MVT − SCHED 965 s vs 1,080 s).
- **At the same MVT − SCHED, early rows taxi longer:** +27 to +65 s in every bin up to 45 min in the recipe months, +26 to +115 s
  over all of 2025, and +7 to +50 s in July 2025 alone. The mean over the edited rows' own timing is +49 / +35 / +61 s (recipe
  months / July 2025 / all 2025).
- v30 learned it. In development, at the same timing, it predicts early rows +26 to +74 s higher, and its residual is about 0 for
  both types (−3.5 s / +1.8 s).
- So the full-restoration Δ (+41 s) has the size of the 2025 gap. Row by row it tracks the gap only loosely: correlation 0.09–0.29
  across the three tables.

**Readings.**

- **sched:** the July-2026 rows are all IOBT = SCHED rows, because the practice ended or the convention carried nothing. Shipped
  v30/v31 is right.
- **mixture:** a pure format change. A share q of the rows is early-type, so the expected label is + q × the 2025 type gap at the
  same timing.
- **early:** every restored row is early-type. This is an upper bound only.

Anchor "relative": shipped v30 is right for the sched type, so a July-2026 level drift cancels. The table gives the bias-SSE change
against shipped, as a share of ranking SSE; 0.01% ≈ 0.0115 s RMSE, and negative is better.

| reading (table) | soft w = 0.5 | soft 0.75 | soft 1.0 | full Δ, w = 0.5 | full 1.0 |
|---|---:|---:|---:|---:|---:|
| sched (any table) | +0.005% | +0.012% | +0.021% | +0.056% | +0.224% |
| mixture, recipe months | −0.013% | −0.016% | −0.016% | −0.003% | +0.106% |
| mixture, July 2025 | −0.008% | −0.007% | −0.005% | +0.015% | +0.142% |
| mixture, all 2025 | −0.019% | −0.024% | −0.027% | −0.019% | +0.073% |
| early, recipe months (bound) | −0.054% | −0.077% | −0.097% | −0.146% | −0.180% |

- **Absolute anchor, LFPG-style** (expected = the 2025 sched-type table itself). On the edited rows the table sits +5 to +29 s above
  shipped v30, depending on the table. With that drift included, soft w = 0.5 gives +0.011 to +0.037% under "sched" and −0.008 to
  +0.025% under "mixture". The table has small bins and no calibration population, so the relative anchor is the primary one.
- **Context variant** (not declared). Adding mean-q × Δ on the 8,567 unedited LTFM July rows (+21.3 s × 0.296) would cost at most
  +0.001% (w = 0.5) to +0.004% (w = 1) under "sched" and gain as much under "mixture". It is negligible either way.
- **Label-free plausibility.**
  - Nothing suggests the early-pushback practice ended. Against other LTFM carriers, THY takes off earlier relative to schedule in
    July 2026 than in any 2025 month: mean difference −570 s, against −381 s in July 2025 and −256 s in January 2026. An ended
    practice would have made it later.
  - The per-number check (MVT − SCHED against q) is too noisy to discriminate: the slope ranges from −516 to +91 s per unit q
    across months.
  - The IOBT field itself changed while MVT − SCHED did not ([`README.md`](README.md), finding 2), which favours "mixture".

## Under v33 (added after v33 became the incumbent)

v33 sets the outer weight to 1.0 on LTFM rows, so its LTFM value is max(committee mix, −12). v9 and the residual stage drop out there, and
the hedge becomes v33 + w · q · Δ_mix. Both tools now record this as well (`delta_v33` and `q_delta_v33` in the per-row table;
`v33_ltfm_mix_only` and `readings_v33_mix_only` in the two JSON files).

- **Δ_mix:** +44.9 s on the restored rows (RMS 55, max 328) and +30.5 s (RMS 43) on the unedited LTFM July rows.
- **q·Δ_mix:** mean **+13.2 s**, RMS 16.8 s, p99 +48 s, max 122 s. Σ = 4.69e6 = 0.026% of v33's ranking SSE.
- **Relative-anchor bounds**, as a share of v33's ranking SSE:

| reading (table) | soft w = 0.5 | soft 0.75 | soft 1.0 |
|---|---:|---:|---:|
| sched (any table) | +0.006% | +0.015% | +0.026% |
| mixture, recipe months | −0.014% | −0.016% | −0.015% |
| mixture, July 2025 | −0.008% | −0.007% | −0.003% |
| mixture, all 2025 | −0.020% | −0.025% | −0.027% |
| early (bound only) | −0.059% | −0.084% | −0.105% |

The reading is unchanged: a small, roughly symmetric bet. It is parked (worth about 0.01–0.02 s, not an upload slot on its own).

## Reading

- **Size.**
  - The soft hedge's whole range is about ±0.02% of SSE, ≈ ±0.02–0.03 s RMSE.
  - At w = 0.5 it gains 0.008–0.019% if the break is a format change, and costs 0.005% if it isn't.
  - w = 1 gains 0.005–0.027% or costs 0.021%.
  - For scale: v31's LFPG fix was −0.015% officially, and v32's LIRF blend +0.017%.
- **Full, unweighted restoration is wrong** under every reading except the implausible "early" bound. The row weight q is what makes
  this usable.
- **The ceiling is low** because the old-convention row can't be named: 55% of the rows carry new 2026 callsigns and sit at p0. A
  perfect row-level q would reach about 0.05% of SSE.
- **No labels exist for this regime.** The development folds can't test it, since 2025 always has the convention. Whether to build a
  declared w-blend on v31 (v31 + w·q·Δ on the 16,618 rows; LTFM rows of v31 equal v30) is the user's call. Nothing was exported to
  `runs/submissions`, uploaded or selected.
