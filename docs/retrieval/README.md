# Retrieval committee member (lever A), from 29 September 2026

**Status: closed by the screen (29 September 2026, 07:00 UTC). NO-GO: the best reading is −0.020% on the recipe proxy
against a bar of −0.05%. No network was built; nothing is declared, exported or uploaded.**

This is lever A of the clean-line search toward a legitimate score below 220 (see the list in
`docs/thinned_regime/README.md` and the session notes): a committee member that looks up the most similar
2025 departures and uses their actual taxi times.

## Why a screen first

Every committee member so far is a parametric function of a departure's own inputs (and, for the context
network, of the movements around it). A retrieval member adds one thing: the labels of similar departures.
A ranking row can only retrieve 2025 departures, all from other months. So the member can only help if
the out-of-sample residuals of similar departures **of other months** say something about a departure's
own residual. That is measurable without building the network.

Any gain must come from a learned similarity.

## The screen (`tools/retrieval_screen.py`, `tools/retrieval_screen_report.py`)

- **Seven fits.** The committee network (`configs/adsb_nn_network_v1.json`, variant `nnb`, seed 1; recipe,
  314-column contract and data path unchanged) is cross-fitted inside the seasonal fold's fit partition
  (2025-02 to 06, 08, 09; LIRF excluded): each fit leaves one month out.
- **Candidates** are the held-out month's rows. Their residual (label minus prediction) comes from a network
  that never fitted them.
- **Queries** are the October rows (the fold's early-stop partition, 171,532 non-LIRF rows). For every
  query the 64 nearest candidates are found in the network's last hidden layer, under two metrics
  (Euclidean; Euclidean with every unit scaled by its head weight) and two scopes (any candidate; same
  airport only).
- **Corrections.** The mean over the seven held-out months of the neighbours' mean residual, each residual
  clipped to ±1,800 s, at k = 4, 16, 64; the same after removing each candidate's airport-month mean
  residual; and, as the plain calibration a retrieval member must beat, the airport-month mean alone.
- **Readings** (October only), for weights 0.25, 0.5 and 1.0 on the correction:
  - against the seven networks' own mean;
  - against the incumbent context pool (ctx seeds 101–116 and ctx_e24 seeds 117–132);
  - on the recipe proxy at network share 0.75 (`tools/nn_design_report.RecipeProxy`), with the paired-day
    95% interval;
  - the correlation of the correction with each reference's October residual.

**Rows read:** the seasonal fold's fit and early-stop partitions only. No scoring month (2025-01, 07, 11,
12) and no ranking row is loaded. A test pins this (`tests/test_retrieval_screen.py`, 5 tests).

**Rule, fixed before any reading:** GO for a declared retrieval-network design round if the best proxy
reading over the grid is at or below −0.05% with its paired-day interval below zero. The design round
would keep its own gate (the context network's was −0.10%). Otherwise the lever is recorded as closed by
the screen.

**What the screen cannot show.** A trained retrieval network learns its own similarity and how to use the
neighbours' labels; it can beat a neighbour average in a fixed embedding. A null screen therefore says the
residuals of similar departures carry little, not that no retrieval design could gain. The bar is set low
for that reason.

## Compute

arcw4–arcw7 (Azure, `Standard_D16ds_v7`), two fits per worker at 8 threads (one on arcw7). No GPU.
The workers are deallocated after the mirror.

| worker | held-out months |
|---|---|
| arcw4 | 2025-02, 2025-03 |
| arcw5 | 2025-04, 2025-05 |
| arcw6 | 2025-06, 2025-08 |
| arcw7 | 2025-09 |

A one-epoch smoke run on 20,000 rows (arcw4, 06:19 UTC) checked the code path only; its numbers are not a
reading and its output was removed before the launch.

## Results (29 September, 07:00 UTC) — [`screen_report.json`](screen_report.json)

All seven fits finished (06:21:50–06:50 UTC, 843–1,248 s per fit, 229–276 s per neighbour search). The mirror is
hash-checked (28 files) and arcw4–arcw7 were deallocated at 06:59 UTC.

**References on October** (169,734 non-LIRF rows of the proxy table): seven-network mean 162.351 s, committee
159.035 s, context pool 154.016 s; control recipe proxy 176.262 s (the recorded value of the GPU sprint, so
the reference chain reproduces).

| correction | sd (s) | r, own networks | r, context pool | r, recipe proxy | proxy at w 0.25 | w 0.5 | w 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| neighbours, any airport, k = 4 | 30.1 | +0.042 | +0.035 | +0.021 | **−0.020%** | +0.006% | +0.199% |
| neighbours, any airport, k = 16 | 17.4 | +0.049 | +0.039 | +0.018 | −0.012% | −0.009% | +0.044% |
| neighbours, any airport, k = 64 | 11.7 | +0.054 | +0.043 | +0.019 | −0.010% | −0.013% | +0.003% |
| neighbours, same airport, k = 64 | 12.4 | +0.059 | +0.041 | +0.021 | −0.010% | −0.013% | +0.007% |
| centred neighbours, any airport, k = 16 | 17.1 | +0.028 | +0.025 | +0.004 | −0.001% | +0.014% | +0.091% |
| centred neighbours, any airport, k = 64 | 11.2 | +0.025 | +0.023 | −0.001 | +0.001% | +0.009% | +0.046% |
| airport-month mean alone (no neighbours) | 7.9 | +0.052 | +0.032 | +0.031 | −0.010% | −0.016% | −0.018% |

(r = correlation with the reference's October residual, residuals clipped to ±1,800 s. The head-weighted metric
reads the same as the Euclidean one; all 54 grid entries are in the report.)

- **Best grid entry:** neighbours of any airport, k = 4, w = 0.25: −0.020% on the proxy, paired-day interval
  [−0.065, −0.005] s. The bar is −0.05%. **NO-GO.**
- **What little the neighbours carry is the airport's average bias, not local structure.** Once each
  candidate's airport-month mean residual is removed, the correlation with the recipe's residual is
  −0.001 to +0.013 and every proxy reading at w ≥ 0.5 is worse than the control. The airport-month mean alone is
  the best correction at full weight (−0.018%), and it uses no neighbour.
- **The stronger the reference, the less is left.** The correlation falls from 0.04–0.06 against the plain
  networks to 0.03–0.04 against the context pool and 0.02 against the full recipe. A correlation of 0.02 allows
  a gain of at most about 0.02% (ρ²/2).
- **By scope:** the best entry moves matched rows −0.078% and unmatched rows −0.002%. The rows that carry
  most of the error (unmatched LTFM, LFPG) get nothing: LTFM +0.014%, LFPG −0.022%.
- Neighbours in the network's last hidden layer are only about 70% from the query's own airport; restricting
  the search to the same airport changes nothing.

**Reading.** Together with the flight-number screen (`docs/clean_line_screens/README.md`), two independent
retrieval readings agree: the out-of-sample residuals of similar 2025 departures, whether matched by identity
or by a learned similarity, say almost nothing about a departure's own residual. The committee's trees already
average over adaptive neighbourhoods, and its CatBoost members already retrieve labels by categorical
combinations. Lever A is closed; no retrieval network was built.

**By-product.** `runs/retrieval/screen/<month>/candidates.parquet` holds out-of-sample predictions of the plain
network for the seven fit months (each from a fit that left the month out), should a later screen need
cross-fitted network residuals.
