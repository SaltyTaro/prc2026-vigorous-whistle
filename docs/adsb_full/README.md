# Full-data ADS-B learner (`tools/run_adsb_full.py`, 15 September 2026)

A single LightGBM (the `timeline_v1` LightGBM parameters: 127 leaves, learning
rate 0.04, early stopping 200 rounds, at most 3,200) on the timeline bundle's
276 model features (`CORE + FLOWN + SURFACE`, the v7 "combined" contract) plus
the 27 label-free ADS-B features (16 `adsb_*` departure-segment features and 11
`nb_*` neighbour features from the pass-2 matches with 1,332 stands under
`data/external/adsblol/matches/`). Target: the departure label. Folds are the
frozen seasonal (fit Feb–Jun + Aug–Sep, early stop Oct, score Jan + Jul) and
forward (fit Jan–Sep, early stop Oct, score Nov) splits of the parent bundle;
the refit uses the eleven 2025 months at the median best iteration of the two
folds. December is never read. Its out-of-sample predictions are meant as a
**stacked feature for the residual stage**, not as a submission: the residual
model (v12 recipe, LIRF excluded) decides row by row how much to trust it.

## Development folds

| Variant | Fold | best iter | score RMSE (all rows) | matched | unmatched |
| --- | --- | ---: | ---: | ---: | ---: |
| all airports fitted | seasonal (Jan / Jul) | 223 | 451.5 (387.8 / 496.9) | 331.1 | 554.3 |
| all airports fitted | forward (Nov) | 226 | 261.6 | 159.9 | 414.9 |
| without LIRF (`--exclude-airports LIRF`) | seasonal (Jan / Jul) | 1,419 | 569.8 (460.5 / 644.5; LIRF 1,230 / 2,157) | 415.9 | 700.7 |
| without LIRF | forward (Nov) | 1,194 | 389.0 (LIRF 1,344) | 204.9 | 645.2 |

Per airport the learner alone beats v9 at the well-covered airports in Jan/Jul
2025 (Jan: EDDF 148 vs 178, EDDM 120 vs 168, LEBL 107 vs 144, EHAM 174 vs 198,
LSZH 179 vs 207; Jul: EDDF 145 vs 190, EHAM 134 vs 168, LSZH 157 vs 194) but is
catastrophic at LIRF (Jan 802 vs 408, Jul 1,586 vs 652: the artifact labels)
and slightly worse than v9 on unmatched rows everywhere. On Nov 2025 it is
slightly behind v12 at most covered airports (EDDF 128 vs 118, EDDM 94 vs 80,
EHAM 111 vs 99, LSZH 128 vs 111) and ahead at LEMD (139 vs 143). With LIRF in
the fit and early-stop sets, early stopping triggers after ~225 rounds because
the LIRF rows dominate the RMSE; a variant fitted **without LIRF** runs far
longer (early-stop RMSE 173 s at round 1,300 versus 229–231 s with LIRF,
different row sets).

Without LIRF the learner is better at every covered airport than the
all-airport variant (Jan: EDDF 140 vs 148, EDDM 112 vs 120, EHAM 162 vs 174,
LEBL 98 vs 107; Jul: EDDF 137 vs 145, EHAM 125 vs 134) and beats even v12 at
EHAM and LEBL in January (162 vs 167, 98 vs 105). A fixed 50/50 blend with the
v12 development predictions on non-LIRF rows: Jan 325.3 vs 326.3, Jul 212.1 vs
212.3. The no-LIRF variant is the one stacked into the v12_v3 residual.

On Nov 2025, restricted to non-LIRF rows, the no-LIRF learner alone ties v12
(165.0 vs 164.5) and a fixed 50/50 blend with v12 gives **160.3 (−2.6%)**:
matched rows 126.9 vs 132.2, unmatched rows 222.2 vs 225.3; per airport EDDF
116 vs 118, EDDM 78 vs 81, EGLL 179 vs 183, EHAM 96 vs 99, LEBL 119 vs 124,
LEMD 131 vs 143 (the learner alone 130), LFPG 253 vs 261, LSZH 112 vs 112, LTFM
207 vs 208. Refit: eleven months without LIRF at the median best iteration
(1,306 = median of 1,419 / 1,194); `runs/adsb_full_nolirf/refit/`.
