# adsb_v27 — the longer lr-0.05 CatBoost residual (declared 22 September 2026, 13:07:21 UTC)

Declaration: `configs/adsb_v27_v1.json` (117 hash-bound parents, commit `6aa6605`).
Incumbent: **`vigorous-whistle_v23` = 238.8197 s** official on 344,841 rows
(SHA-256 `f960b72e0b729d47729eed2272634ba8525e57cff9bad243291d96c8fd14fac7`).
Nothing in this round is uploaded without the user's approval.

## Why this round

`adsb_v25` shipped a CatBoost residual member (`cat`: depth 8, l2 leaf 3, learning rate 0.05,
seed 20260915, cap 8,000). Its inner best iterations were 7,898–8,000 on **every** fold of
**every** month, and 7,999–8,000 on the shipped refit: the loss was still falling when the model
ran out of rounds. That signal has two readings — take bigger steps, or take more of them.

`adsb_v26` tested the first: the same model at learning rate 0.10. It converged before the cap
(5,111–6,408 rounds) and was **worse** than the shipped member on its own scope in 8 of the 11
months fitted (January +1.15 s); every candidate was worse on January, the rule's "not worse on
any recipe month" failed, and the round selected nothing. The remaining reading is the one this
round tests: **more small steps**.

## What changes

One new development variant:

| variant | kind | learning rate | depth / l2 | seed | cap |
|---|---|---|---|---|---|
| `cat` (shipped, inherited) | CatBoost | 0.05 | 8 / 3 | 20260915 | 8,000 |
| **`cat_long` (new)** | CatBoost | 0.05 | 8 / 3 | 20260915 | **16,000** |

The cap is the only difference, in development and at export. The two shipped members (`adj2`,
the month-weighted LightGBM refit per ranking target month, and `cat`) are **not** refitted: their
development residuals come from the hash-bound `runs/adsb_v25/dev` columns, and at export they
would be the shipped v23 model files themselves.

Candidates (row-wise means of their members' residuals):

| candidate | residual members |
|---|---|
| `repro` (check, not selectable) | adj2, cat |
| `adj2_long` | adj2, cat_long |
| `adj2_cat_long` | adj2, cat, cat_long |
| `long_only` | cat_long |

## Protocol

Unchanged from v25/v26. Twelve-month leave-one-month-out on the `scratch/v24` cross-fitted-v9
frames: for each held-out month `cat_long` is fitted on the in-scope rows (`matched_no_lirf`) of
the other eleven months, rounds from a 5-fold day-grouped inner split (median best iteration,
patience 200, cap 16,000), and predicts the held-out month. Reading 1 is each variant's and each
candidate's pooled twelve-month residual-scope RMSE; reading 2 scores every candidate on **all**
rows of the four recipe months (January, July, November, December 2025) through the frozen blend
arithmetic against the incumbent's own development prediction.

Selection rule (pre-registered, identical to v25/v26): eligible when the candidate is **not worse
on any recipe month**, its mean ratio over the four is at most 0.9998 (0.02%), the paired-day
bootstrap interval of the RMSE difference is below zero on every recipe month it touches, and its
pooled twelve-month residual-scope RMSE is strictly below the control's (the incumbent's
mean of `adj2` and `cat` on the same rows). Lowest mean ratio wins; otherwise nothing is selected
and v23 stands.

## Execution

Azure only, workers `arcw0`–`arcw7` (`Standard_D16ds_v7`, `eastus`), one `(month, cat_long)` job
per worker at 16 threads, chains in `runs/adsb_v27/chain_dev_<host>.sh`. Twelve jobs over eight
workers, **the four recipe months first** (arcw0 January, arcw1 July, arcw2 November,
arcw3 December), so the decisive read arrives after the first wave and a null can be cut early as
in v26. Scoring, selection, export assembly and verification run on the workstation.

A CatBoost fit at cap 8,000 took ≈5,300 s per month in v25; at 16,000, with no early stop
expected, ≈10,600 s (≈3 h) per month, so ≈3 h to the recipe-month read and ≈6 h to all twelve.

Smoke test (disclosed in the declaration): before it was written, the tool's fit and score stages
were exercised against a scratch copy with `--smoke` (30,000 subsampled rows, cap 60) on 2025-11
and 2025-02; `cat_long` fitted and predicted, the four candidates assembled, and `repro`
reproduced the incumbent's recorded November RMSE (189.35263008727185) exactly. Those runs scored
no candidate at protocol and are not development results. The development tool
(`tools/run_adsb_v26.py`) is byte-identical to adsb_v26's and is hash-bound as a parent.

## Result

Pending — nothing has been scored at protocol yet.
