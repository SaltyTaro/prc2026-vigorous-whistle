# adsb_rnn2 design: a 32-seed residual network (24 September 2026) -- stopped at the design

The user asked to try doubling v26's residual network from 16 to 32 seeds. The October design reading
(`tools/rnn2_design_report.py`, `design_report.json`; the sixteen adsb_rnn leave-one-month-out October fits; no recipe or ranking
month read) says the lever is spent, so no fit was declared or launched.

- Seed curve of the network alone on October's scope: 1 seed 115.88, 2 114.48, 4 113.62, 8 113.02, 16 112.76 s (trees 114.29);
  the RMSE^2 = B + V/k fit puts the infinite-seed limit near 112.67 s.
- October recipe (v26 analogue = 0.05 adj2 + 0.05 cat + 0.9 net16, v25's non-LIRF recipe rebuilt on October): doubling 8 -> 16 seeds
  gained -0.0091%, so 16 -> 32 is expected near -0.005%, a quarter of the 0.02% bar. Network weight 1.0 (no trees): -0.0055%.

Two further October readings (descriptive, scratch; not declared):
- Adding the October design fits of the wide network (rwide, 4 seeds) as a residual member improves the scope (112.55 -> 112.10 s at
  0.3) but worsens the recipe at every share (+0.002% to +0.022%): it overlaps the committee, which uses the same features.
- A larger residual-path weight (0.35 / 0.40 / 0.50 instead of 0.30) reads -0.09 / -0.17 / -0.25% on October, but that reading is
  biased: in development the residual models are fitted on the eleven other months while the committee's January/July/October
  predictions come from seasonal-fold models fitted on seven months (November: nine). Only December compares like with like (both
  eleven months, as in the twelve-month refits that ship), and December preferred 0.30 (the disclosed adsb_rnn smoke reading:
  +0.038% at 0.35 with v25's residual). Not pursued.

v26 (234.1438) remains the incumbent. No Azure worker was started.
