# Dig tunnel 3: Istanbul (LTFM) — no GO (3 October 2026)

Scratch screen, 2025 only. No official score and no leaderboard was used, nothing was declared, built or uploaded. Code and outputs are
in `runs/dig/ltfm/` (`build.py` features, `screen.py` month-blocked residual screen, `increment.py` transfer test, `*.json` results).

The control is the shipped chain's development proxy, `runs/scout/refit/ctl_v55m.parquet` column `new`. On it, LTFM RMSE is 199.0 /
240.1 / 187.4 / 187.2 s (January / July / November / December). Every reading below is the % change of the whole month's RMSE (all
airports, LTFM rows replaced). Paired-day 95% intervals resample UTC take-off days (2,000 replicates); they are in the JSON files.

## What LTFM's inputs hold

- At LTFM the NM record carries almost no off-block information. LOBT equals EOBT_1 on every 2025 row, IOBT equals EOBT_1 on almost
  every row, and EOBT_1 is the schedule in most rows. AOBT_3 is take-off minus a fixed allowance: about 1,200 s on 35L/36 (north
  flow), 1,440–1,500 s on 34L/17R/16R.
- The residual (label − `new`) is strongly shared in time:
  - correlation 0.245 between consecutive take-offs on the same runway less than 10 minutes apart;
  - 0.27 with the leave-one-out mean residual of all take-offs within ±15 minutes (0.27 when ordered by predicted off-block instead).
- That shared state does not show in any label-free signal tested. Correlations with the ±15-minute state are all at or below 0.076.
  The signals tested were:
  - backlog of scheduled against actual departures over 1 h, 2 h and 6 h (0.076 at 6 h);
  - mean schedule delay of the last 30 minutes' take-offs (0.074);
  - arrival taxi-in, take-off and landing rates;
  - queue position, busy period, peers predicted to push ahead, and configuration flips.
- 1.34% of LTFM rows with an error above 600 s hold 34% of LTFM's squared error; 77% of them are under-predictions.

## Screen 1: residual model on the four development months (control `new`, fit on three months, read the fourth)

Each run uses fixed LightGBM settings (15 leaves, minimum 400 rows per leaf, 300 trees). The base model sees only control-side
features: prediction, a, hour, weekday, NM proxy, allowance residual, runway, stand letter, THY flag and wake. Every other run adds one
feature group to the base.

| Feature set | Equal-footing mean, w 0.25 / 0.5 / 1.0 |
|---|---|
| base | +0.005 / +0.024 / +0.105% |
| + NM structure (IOBT/EOBT/LOBT offsets, seconds, flown vs plan) | +0.004 / +0.022 / +0.097% |
| + runway configuration (flow, flips, active runways, mixed mode, arrivals, taxi-in) | +0.002 / +0.017 / +0.091% |
| + runway queue (take-offs on own runway in 5/10/20 min, busy period, gaps) | +0.004 / +0.022 / +0.096% |
| + peers (iterated with the twelve-month predictor: pushed ahead, taxiing at push) | +0.007 / +0.029 / +0.115% |
| + THY bank | +0.005 / +0.024 / +0.103% |
| + stand (stand × runway shrunk residual encodings, leave-month-out) | −0.005 / +0.007 / +0.078% |

Nothing improves the shipped chain. December worsens most in every set.

## Screen 2: twelve months, older predictor (rows12 `stack`, leave-one-month-out, eleven months fitted)

| Feature set | Mean over 12 months, w 0.25 / 0.5 / 1.0 | Months better (w 0.5) |
|---|---|---|
| base | −0.044 / −0.070 / −0.069% | 12/12 |
| all new groups except stand | −0.079 / −0.133 / −0.170% | 12/12 |
| base + stand | −0.055 / −0.091 / −0.106% | 12/12 |

The new groups do carry information against the older, weaker predictor.

## Screen 3: does that increment transfer onto the shipped chain?

Method: fit full and base models on the eleven other months against the older predictor. Then add w × (full − base) to `new` on the
LTFM rows of the scored month.

| Increment | Equal-footing mean, w 0.25 / 0.5 / 1.0 | November |
|---|---|---|
| configuration + queue + peers + bank + NM + backlog | +0.003 / +0.035 / +0.187% | +0.006 / +0.036 / +0.167% |
| the same + stand | −0.012 / +0.005 / +0.130% | −0.025 / −0.022 / +0.065% |
| stand alone | −0.0105 / −0.0134 / +0.004% | −0.022 / −0.033 / −0.026% |

- The shipped chain already holds the configuration, queue and peer information. Its context networks see these movements.
- What remains is a small stand × runway bias. Adding the shrunk encoding directly gives −0.003 / −0.004 (January), −0.004 / −0.001
  (July), −0.023 / −0.037 (November) and −0.017 / −0.021 (December) at w 0.25 / 0.5. That is about a tenth of the −0.05% bar.

## Verdict

**No GO.** At LTFM the remaining error is a shared temporal state (±15-minute correlation 0.27) plus row noise. The stand × runway residual bias is real but worth only about −0.01% of month RMSE.
