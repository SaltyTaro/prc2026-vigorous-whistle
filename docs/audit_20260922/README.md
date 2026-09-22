# Audit of the v22 recipe on December (22 September 2026)

Read-only review of the shipped recipe (`vigorous-whistle_v22`, official 239.1335) on its out-of-sample development predictions,
December first: `tools/audit_december_v22.py` → `audit.json`. Inputs are the adsb_v24_v1 development tables
(`runs/adsb_v24/dev/predictions_<month>.parquet`, candidate `v7x_r12x_lirf25` = the v22 recipe, `baseline` = v21), the twelve-month
scratch frames, the cached residual feature tables and the committee tables rebuilt exactly as the development tool built them
(January–November refit leaves on December, fold predictions on January / July / November). The recipe is rebuilt from its components
with the frozen blend arithmetic before anything is reported (maximum difference 0 on every month). Nothing was fitted or selected;
every number is descriptive on reused development data. OpenSky Trino access is still not granted, so LFPG/LTFM surface coverage
stays out of reach and this audit asks what is left with the data we have.

## 1. Where December's squared error sits (RMSE 176.54; v21 176.99)

| class (ordinary rows, label ≤ 6 h) | rows | share of SSE | RMSE final | v9 | committee alone | v9 + residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unmatched, non-LIRF | 46,454 | 44.5% | 218.6 | 223.7 | 219.5 | 223.7 |
| dark (segment starts moving) | 63,143 | 22.9% | 134.7 | 184.2 | 136.8 | 138.7 |
| LIRF non-population, matched | 9,506 | 13.7% | 268.1 | 283.9 | (never fitted) | — |
| gate (segment starts at the stand) | 44,012 | 6.8% | 87.5 | 169.7 | 90.8 | 88.7 |
| LIRF population (fill mixture) | 84 | 6.5% | 1,968 | = final | — | — |
| LIRF non-population, unmatched | 2,474 | 5.6% | 337.2 | 339.1 | (never fitted) | — |

Four rows above six hours carry 3.3% (all LIRF). By airport: LIRF 25.8%, LFPG 22.0%, LTFM 19.6%, EGLL 14.1%, the other six 15%.
The top 200 rows carry 26.6% of December's SSE and 174 of them are under-predicted (long delays). November / July / January give
the same shape (unmatched 42–56%, dark 13–27%, LIRF non-population 13–19%); January is dominated by two LFPG rows above six hours.

The ranking's match composition (label-free, `ranking_match_composition_pct`) is close to December's for January 2026 (LTFM 100%
unmatched, LFPG 79%, EGLL 12%, LIRF 20%) and differs for July 2026 only at LIRF (100% unmatched: no adsb.lol coverage) and LEMD (26%).

## 2. What each stage is worth (RMSE change if the stage is removed, same frozen arithmetic)

| removed | Jan | Jul | Nov | Dec |
| --- | ---: | ---: | ---: | ---: |
| residual stage (v9 + r on matched non-LIRF rows, weight 0.3 through the blend) | +0.67% | +0.84% | +2.22% | +2.62% |
| committee (29 learners, weight 0.7 on non-LIRF rows) | +0.17% | +0.50% | +1.20% | +1.77% |
| both (v9 alone) | +3.76% | +7.21% | +14.59% | +18.76% |

On matched rows the single residual LightGBM (27 ADS-B + neighbour features, `label − v9`) is as good as the 29-member committee:
December dark rows 138.7 (v9 + r) vs 136.8 (committee), gate rows 88.7 vs 90.8. Its marginal value is 1.5× the committee's on the
two well-covered months, and January 2026 is covered like December. The committee's advantage is on unmatched rows, where it is
worth about 2% over v9 (218.6 vs 223.7) and nothing else can act.

## 3. Things checked and found small

- **Additive bias.** Cells with the same residual sign in all four months: EGLL (−5…−15 s), LEMD (−4…−19 s), EGLL/LEMD unmatched,
  LSZH gate, 26 airport-hour cells (EGLL 12–13 h −20 s, LTFM 07–09 h +8…+40 s). A perfect additive correction of every consistent
  cell would remove 0.04–0.11% of a month's SSE. Not a lever.
- **Midnight rows (day-file boundary at the match stage).** 00:00–00:20 UTC holds 140–250 non-LIRF rows a month, matched 0–29% versus
  50–70% later in the day, and carries 0.2–1.0% of SSE (December's 2.9% in 00:10–00:20 is one LIRF population row). Day±1 surface
  loading is worth at most ~0.05%, as previously estimated.
- **LIRF non-population unmatched rows** (all of LIRF in July 2026): final 337.2 vs v9 339.1 in December, 283.1 vs 282.6 in November.
  The 0.25 blend on these rows is neutral; learners without ADS-B do not beat v7's timeline models here.
- **LIRF non-population matched rows**: the v20/v22 LIRF table is worth −5.6% in December (268.1 vs 283.9), +0.2% in November.

## 4. One identifiable block that is not a lever

LIRF population rows with a schedule interval above 43,200 s stay at v7 by the v9 declaration. On the four labelled out-of-sample
months there are 15 such rows: 7 schedule fills (label = interval) and 8 one-day rows (label = 86,400 s + an ordinary taxi); v7's
prediction sits 2–7 k s above the interval on the fold months and 9.4 / 9.6 k s above it on December's two rows (the January–November
refit hedges harder than the fold models). The ranking holds 11 such rows, predicted 2.8–9.1 k s above their intervals; if every one
were a fill they would carry 1.83% of the ranking's squared error (about 2.2 s of RMSE). The labelled evidence, however, says 38%
of long-schedule rows are one-day rows across 2025 (11 of 29 over twelve months, in seven different months) with no separating
feature (the turnaround signal is equally negative for both modes; the shipped classifier's 0.92 one-day probability on one ranking
row rests on 12 training rows), so the label-optimal move is the hedge that v8 already took and lost. Only the v8 leaderboard result
argues for predicting the fill, and a row-level rule may not be derived from it (`docs/postprocess/README.md`). Left at v7.

## 5. Reading for the next round

1. **The residual stage is the best-leveraged cheap item.** It carries more of the gain than the committee on covered months, it is
   one LightGBM that fits in minutes, and the v24 assets make a twelve-month leave-one-month-out possible for it for the first time.
   Untested on it: recency/season sample weights (the residual learning curve showed adjacent months dominate), rounds/learning-rate
   tuned on the twelve-fold LOMO, and a CatBoost residual. December is the transfer anchor as before.
2. **CatBoost kinds inside the committee** act on the same matched rows plus the 2% unmatched-row edge; the committee's marginal value
   is 0.2–1.8%, so a kind change of the ctr2 size (0.3% of the committee's own error) is worth 0.05–0.1% end to end. A VM round.
3. Day±1 loading, additive bias, per-hour corrections: ≤ 0.05% each, not worth a declaration.
4. LFPG / LTFM / LIRF-July unmatched rows (55% of ordinary SSE) remain out of reach without new surface data.
