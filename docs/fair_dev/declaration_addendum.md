# Addendum to configs/fair_dev_v1.json (27 September 2026, before any fair-set score)

The declaration file is left unchanged, because its hash is referenced.

**1. The known_caveat understates the training-row channel (wording).**

- The v24 cross-fit groups are whole calendar months: A Jan/May/Sep, B Feb/Jun/Oct, C Mar/Jul/Nov, D Apr/Aug/Dec.
- So the cross-fit v9 on the residual stage's training rows came from v7 models that saw the ENTIRE scored months:
  - seasonal fold: every fit month's base saw January or July, and B/D months saw both; they also saw November and December;
  - forward fold: every fit month except March/July saw all of November;
  - december fold: every fit month except April/August saw all of December.
- The channel is still second order. A scored month's labels shape a training row's base value, which then shapes that row's residual
  target and its v9 input. No scored label is a target or feature of a scored row, and the sign of the effect is ambiguous.
- The ranking has no analogue: the shipped bases never saw 2026.
- Removing it would need a nested cross-fit v7 within each fold's fitting months, which is not done.

**2. October favours the fold side slightly.**

- The committee, the context networks and the fold v9 used October for early stopping.
- The fair residual stage uses no October rows and no stopping set: its inner split lies within the fitting months.
- The fold side therefore carries a little October information that the v9 side's residual does not. This is judged negligible and
  recorded here.

**3. End-to-end check (December).**

- The december fold's training set is identical to the shipped leave-one-month-out December fits: January–November, cross-fit v9, the
  same seeds.
- All 16 rnb seeds reproduce the shipped December training histories exactly (maximum per-epoch loss difference 0.0; 1,006,411 rows
  each).
- The fair December residual differs from the shipped one only through the v9 input (the January–November refit v9 instead of the
  cross-fit v9): seed 501 has an in-scope RMS difference of 13.2 s and a correlation of 0.995.
- The tree fits get the same check once they finish (fit_december.json against runs/adsb_v25/dev/month_2025-12.json).

**4. Guards.**

- The reproduction checks in tools/run_fair_dev.py and tools/run_fair_weights.py now reject NaN, so a NaN no longer passes silently as
  `NaN > 1e-9` did.
- tools/run_fold_residual.py checks for an existing run before fitting.
- No completed fit was affected.
