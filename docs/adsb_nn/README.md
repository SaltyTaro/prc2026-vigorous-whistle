# adsb_nn — a neural-network committee member (from 22 September 2026, 21:45 UTC)

**Status: SHIPPED. `vigorous-whistle_v24` (candidate `nn50`) was uploaded with the user's approval ("upload it")
and officially scores 235.3074 on all 344,841 rows — 3.5123 s (1.471%) better than v23 (238.8197). v24 is the
incumbent. Never re-upload v1–v24.**

## Why

Every lever the data supports has been tested: residual-stage tuning (v25 shipped, v26/v27 null),
committee seeds and re-weighting (spent at v18), calibration, test-time adaptation, the matcher fixes and
the blend weights (v28 null). The one model family never fitted in this project is a neural network — the
committee is 9 LightGBMs and 13 combination CatBoosts, all gradient-boosted trees. A differently-biased
learner is the remaining way to add diversity to the committee, which carries 0.7 of every non-LIRF
prediction (including the unmatched rows that hold 60% of the non-LIRF error, where the residual stage
does nothing).

A checked prerequisite: every committee CatBoost stopped at 9–15k rounds under its 20k cap, so v27's
"train longer" finding does not carry over to the committee.

## Protocol

1. **Design (October only)** — `configs/adsb_nn_design_v1.json`. Networks are fitted on the seasonal
   fold's fit partition (2025-02..06, 08, 09; LIRF excluded, like the committee: 1,130,307 rows) and
   monitored on its early-stop partition (October 2025, 171,532 non-LIRF rows) — the month every committee
   learner already used for early stopping. The reading is the RMSE of
   `(1 − a) × committee + a × network` on October, where the committee is `ens_v20_bag_q_all` rebuilt from
   its members' seasonal-fold models (`tools/nn_committee_october.py`; those models early-stopped on
   October, so the reading is conservative for the network). **No design run predicts, scores or reads a
   label of the four recipe months (2025-01/07/11/12) or of the ranking months.**
2. **Declaration** — the chosen variant, its seeds, the mixing weights and the rule are fixed in
   `configs/adsb_nn_v1.json` before any development score.
3. **Development** — each seed fitted like a committee learner: seasonal fold (predicts January/July),
   forward fold (predicts November), January–November refit (predicts December). The v23 recipe is rebuilt
   on the four months with only the committee table replaced by the mix (`tools/run_adsb_nn_dev.py`); the
   zero-weight control must reproduce v23's recorded development RMSE on every month.
4. **Ship (only if selected)** — twelve-month refit of every seed, export with only the committee
   ranking table changed, independent verification, audit, then the user's approval before upload.

## The network (`src/prc2026_nn/`)

- Inputs: the x19 committee learners' 314 columns (276 timeline + 27 ADS-B + 11 v19), same rows
  (LIRF out of fitting), same data path (`tools/run_adsb_learner.py`, imported unchanged).
- Numeric features: piecewise-linear encoding against fitting-row quantile edges (32 bins), a per-feature
  linear map to 8 dimensions and a ReLU (the "Q-LR" embedding of Gorishniy et al., 2022), plus a
  missing-value flag per feature. Categorical features: embeddings (index 0 missing, 1 rare/unseen,
  categories with ≥ 10 fitting rows kept).
- Backbone: ReLU MLP; optional TabM-mini ensemble (k multiplicative input adapters, shared backbone).
- Training: MSE on the standardised label, AdamW, warm-up + cosine over a **fixed epoch budget**, gradient
  clipping, optional weight EMA. Monitoring never changes training; the saved model is the final epoch's,
  so the refits repeat the development recipe exactly.
- Tools: `tools/run_adsb_nn.py` (fit/predict one variant × seed × fold), `tools/nn_committee_october.py`
  (design reference), `tools/nn_design_report.py` (design readings), `tools/run_adsb_nn_dev.py`
  (declared development scoring and selection). Tests: `tests/test_nn.py`.

## Compute

Azure workers arcw0–arcw7 (Standard_D16ds_v7, 8 physical cores; 8 threads beat 16 in a measured
benchmark). PyTorch 2.13.0 (CPU build) was installed into each worker's project venv. The base network
trains at ~37 s per epoch on the 1.13M-row seasonal fit partition. No GPU was provisioned.

## Execution log

- **22 Sep 2026, ~21:45 UTC** — arcw0–arcw7 started; PyTorch 2.13.0 (CPU) installed in each worker's venv;
  the learner ADS-B caches of the v23 twelve-month refits (identical SHA-256 on all eight workers) copied to
  `runs/adsb_nn/features`. `tests/test_nn.py` (6 tests) passes on the workstation and on arcw0.
- **Design round 1** (`configs/adsb_nn_design_v1.json`): base, rank, clip21600, clip5400, ema, big, tabm4,
  slow — one per worker, seed 1, 8 threads. Committee October reference
  (`tools/nn_committee_october.py`, arcw4): two members re-predict their saved January fold predictions with
  difference 0.0 (label-free check) before October is scored; committee October RMSE 163.413 s on the
  171,532 monitored non-LIRF rows.
- **Design round 2** (`configs/adsb_nn_design_v2a.json`, `v2b`, …): started on workers as round-1 runs
  finished — a second base seed (seed noise), stronger regularisation, a 10-epoch budget, clip + EMA,
  richer numeric embeddings, larger categorical embeddings.
- The `created_at_utc` fields inside the design configuration files are placeholders written with the
  files; the actual start of every design fit is the `created_at_utc`/log of its `run.json`.
- **Smoke test of the development tool** (`tools/run_adsb_nn_dev.py --smoke`, provisional declaration
  `runs/adsb_nn/smoke_decl.json`, output `runs/adsb_nn/smoke_dev/`): the network table is replaced by the
  committee's own, so every candidate collapses onto the control and no network is read. The rebuilt
  control reproduces v23's recorded development RMSE on all four months (328.7524 / 267.5352 / 189.3526 /
  176.3497) and every candidate equals it within 1.8e-12 s.

## Design results (October only) — `docs/adsb_nn/design_report_1.json`

Committee (`ens_v20_bag_q_all`, seasonal-fold members) on the 171,532 monitored October rows: **163.413 s**.
"Blend" is `(1 − a) × committee + a × network` at its best `a`; "recipe" is the October proxy of the v23
recipe (cross-fitted v9, the leave-one-month-out October residual columns, 0.7 × the committee table on
non-LIRF rows; LIRF rows at v9 for every candidate) on all 183,761 October departures.

| run | network | blend (best a) | recipe (best a) | error corr. | note |
|---|---|---|---|---|---|
| base s1 … s8 | 170.67–172.55 | −0.73 … −1.00% (0.25–0.30) | −0.33 … −0.44% (0.30–0.35) | 0.890–0.894 | seed noise |
| **base, mean of 2** | 167.74 | −1.12% (0.35) | −0.50% (0.40) | 0.912 | |
| **base, mean of 4** | 165.82 | −1.25% (0.40) | −0.57% (0.45) | 0.925 | |
| **base, mean of 8** | **164.51** | **−1.44% (0.45)** | **−0.66% (0.50)** | 0.930 | flat 0.45–0.55 |
| rank (scalar inputs) | 182.76 | −0.07% (0.05) | −0.01% | 0.876 | embeddings matter |
| clip21600 | 171.54 | −0.86% | −0.37% | 0.893 | within seed noise |
| clip5400 | 174.08 | −0.74% | −0.32% | 0.881 | worse |
| ema 0.999 | 171.17 | −0.84% | −0.37% | 0.897 | within seed noise |
| clip21600 + ema | 171.12 | −0.85% | −0.37% | 0.897 | within seed noise |
| e10 (10 epochs) | 170.99 | −0.69% | −0.27% | 0.906 | better alone, less diverse |
| big (1024×4, dropout 0.2) | 172.10 | −0.69% | −0.27% | 0.899 | worse |
| reg (dropout 0.2, wd 1e-3) | 172.43 | −0.52% | −0.19% | 0.904 | worse |
| slow (lr 5e-4, 24 epochs) | 174.66 | −0.71% | −0.30% | 0.878 | worse |
| emb16 (48 bins, 16-dim) | 172.03 | −0.83% | −0.38% | 0.891 | within seed noise |
| cat32 (categorical cap 32) | 172.07 | −0.80% | −0.34% | 0.892 | within seed noise |
| tabm4 (TabM-mini, k = 4) | 167.49 | −0.89% (0.35) | −0.36% | 0.925 | ~3.2× a base seed's compute |

**Decision.** Seed averaging dominates every architecture change: the base recipe's seed mean improves
steadily (2 → 8 seeds: −1.12% → −1.44% at committee level, −0.50% → −0.66% on the recipe proxy) while no
variant beats a single base seed by more than the seed noise. TabM-mini is the strongest single network but
its members are correlated with each other and with the committee (0.925); one TabM seed blends below two
base seeds at about 1.6× their compute. The declaration therefore fixes the **base recipe
(`configs/adsb_nn_network_v1.json`, variant `nnb`, identical settings and variant hash to the design base)
with 16 fresh seeds (101–116)** and the network shares **0.3 / 0.4 / 0.5 / 0.6** around the recipe optimum.
A second tabm4 design seed was still running at declaration time; it is reported below as an addendum and
cannot change the choice.

## Declaration — `configs/adsb_nn_v1.json` (22 Sep 2026, 22:41:07 UTC)

- Candidates: control (share 0) and nn30 / nn40 / nn50 / nn60 — the committee table becomes
  `(1 − a) × ens_v20_bag_q_all + a × ens_nn` inside the otherwise unchanged v23 recipe.
- Rule (adsb_v25/v26/v27's, with the control in place of the baseline): no recipe month worse on all rows;
  mean ratio over 2025-01/07/11/12 at least 0.02% below 1; on every touched month the paired-day 95% interval
  of the RMSE difference below zero; lowest mean ratio wins.
- The control must reproduce v23's recorded development RMSE (328.7524 / 267.5352 / 189.3526 / 176.3497).
- 124 hash-bound parents, including the network code, both configuration files, the design report and the
  committee October reference. Code on all eight workers verified byte-identical (combined hash
  `be8cebe7aa9aa695…`) before the launch.
- Development chains launched right after the declaration: arcw*n* runs seeds 101+*n* then 109+*n*, each
  seasonal → forward → January–November refit (`tools/nn_chain.sh`); arcw4 waits for its design run.

## Design addendum (after the declaration) — `docs/adsb_nn/design_report_addendum.json`

The second tabm4 design seed finished at ~22:50 UTC, after the declaration. tabm4 seed 2 alone: 167.25,
blend −0.97% (0.35), recipe proxy −0.41%; the mean of the two tabm4 seeds: 165.73, blend −1.07% (0.40),
recipe proxy −0.45% (0.45), error correlation 0.933. Two base seeds (a third of the compute) reach
−1.12% / −0.50% and eight base seeds (about the compute of 2.5 tabm4 seeds) −1.44% / −0.66%. The
declared choice stands; nothing in the declaration was changed.

## Development execution notes

- The seven first-wave chains (seeds 101–104, 106–108) finished 23:24–23:28 UTC; each seed's seasonal,
  forward and refit folds take ~12 / ~14 / ~17 minutes. First refit record checked: 1,770,734 fitted rows
  (January–November, LIRF excluded), all 344,841 ranking and 165,677 December rows predicted, December leaf
  cache written.
- **Launch bug on arcw4 (no effect on any fit):** its wrapper was told to wait for the tabm4 design run with
  `pgrep -f "design/tabm4_s2 "`, which matched the wrapper's own command line, so it waited on itself and
  never started seed 105. Found at 23:44 UTC; the waiting wrapper was killed and seeds 105 and 113 were
  started side by side on arcw4 (two 8-thread jobs; measured 1.2× the throughput of one). No fit had started,
  so nothing was repeated or discarded.
- Mirroring helper `tools/nn_mirror.sh`: its first run passed a newline-separated fold list into the remote
  command and copied only the seasonal folds; fixed (space-separated) and re-run before any file was used.

## Development results — `docs/adsb_nn/development_results.json`, `runs/adsb_nn/dev/selection.json`

All sixteen seeds × three folds fitted on arcw0–arcw7 (22:41–00:48 UTC); every run record carries the declared
variant, code and configuration hashes; fitted rows equal the committee learners' (seasonal 1,130,307, forward
1,448,198, January–November refit 1,770,734). The seed mean is `runs/adsb_nn/ens_nn` (tools/ensemble_adsb_full.py;
mean pairwise seed correlation 0.969 / 0.978 / 0.971). The control reproduced v23 on all four months before any
candidate was compared.

Whole-month RMSE change against the control (v23's development predictions), all rows:

| candidate (network share) | 2025-01 | 2025-07 | 2025-11 | 2025-12 | mean | intervals < 0 | eligible |
|---|---|---|---|---|---|---|---|
| nn30 (0.3) | −0.210% | −0.258% | −0.471% | −1.044% | −0.496% | 4/4 | yes |
| nn40 (0.4) | −0.235% | −0.281% | −0.531% | −1.281% | −0.582% | 4/4 | yes |
| **nn50 (0.5)** | **−0.238%** | **−0.271%** | **−0.542%** | **−1.462%** | **−0.628%** | **4/4** | **selected** |
| nn60 (0.6) | −0.218% | −0.230% | −0.504% | −1.587% | −0.635% | 3/4 | no |

**Selected: `nn50`** — the committee table becomes `0.5 × ens_v20_bag_q_all + 0.5 × ens_nn`, so the network
carries 0.35 of every non-LIRF prediction; LIRF rows are untouched. nn60 has a marginally lower mean but its
July interval does not lie below zero, so the rule excludes it.

nn50 in detail (RMSE, s): January 328.752 → 327.972 (interval −1.74 … −0.23 s, 25/31 days better); July
267.535 → 266.810 (−1.33 … −0.09, 24/31); November 189.353 → 188.327 (−1.54 … −0.43, 28/30); **December
176.350 → 173.772 (−3.36 … −1.83, 31/31 days)**. The gain is on matched and unmatched non-LIRF rows alike
(December matched 117.19 → 113.90, unmatched 218.60 → 215.22) and on almost every airport (December: EDDM
−10.2%, EGLL −2.9%, EHAM −2.7%, LSZH −2.6%, EDDF −1.5%, LFPG −1.5%, LTFM −1.4%); LFPG is slightly worse in
July/November (+0.69% / +0.59%). On December — out of sample for every model involved — the sixteen-seed
network alone (154.12 s on non-LIRF rows) is better than the committee (156.92 s).

For scale: the last shipped round (v25 → v23) was −0.057% on the same four months; past official changes
landed at or above their December read (v22: −0.25% → −0.19%; v23: −0.11% → −0.13%).

## Ship: twelve-month refits, export v24, verification, audit (23 Sep 2026, UTC)

- **Twelve-month refits:** all 16 seeds refit on January–December 2025 at the declared 16-epoch budget
  (`--fold refit12`, 00:08–01:40 UTC). They were started on workers as their development chains finished, before
  the selection, purely for scheduling; they are used only because `nn50` was selected. One queued refit
  (seed 113) was moved from arcw1 to the idle arcw4 before it started. All eight workers were then deallocated.
- **Export** (`tools/export_adsb_nn.py --version 24`, `docs/adsb_nn/export.json`):
  `runs/submissions/vigorous-whistle_v24.parquet`, SHA-256 `7ed96d4620dd7bbcec263a477a652555ef33ad20b13e3c673e1818d9e2ffb011`. The v23 ranking values rebuilt from
  their recorded components with the committee table unchanged equal the recorded v23 to **0.0 s** and the
  uploaded v23 integers exactly; then the committee table becomes 0.5 × committee + 0.5 × the 16-seed network
  mean. 306,038 rows change against v23 (median 7 s, p99 105 s, max 1,293 s, RMS 26.6 s; mean −0.35 s;
  January −1.8 s, July +0.75 s); every LIRF row equals v23. The network and the committee agree on the ranking
  (non-LIRF means 1,027.4 / 1,028.5 s, correlation 0.98, row-level SD of their difference 79 s).
- **Verification** (`tools/verify_adsb_nn.py --version 24 --tolerance 0.01`, `verification.json`): passed —
  the network table equals the plain mean of the 16 saved seed predictions (difference 0.0); on non-LIRF rows
  `max(0.3 × (v9 + residual) + 0.7 × mix, −12)` rounds to the exported integers on every row; every LIRF row
  equals the uploaded v23; file IDs, order and types equal the template and the SHA-256 equals the receipt.
- **Two verifier findings, neither a model or export defect:**
  1. The first run rebuilt the ADS-B ranking table on the workstation and found network differences up to 6.2 s.
     The rebuilt table is not byte-identical to the one the networks (and the shipped committee) used
     (`ca057b84…` vs `1dcf5c7f…`): two stand-distance columns differ by ≤ 1.8e-12 m (libm), and
     `nb_runway_prev_taxi_s` / `nb_runway_next_taxi_s` differ on 39 / 36 of 344,841 rows because the frozen
     `neighbour_features` sorts by takeoff time with an unstable sort, so ties can pick a different
     neighbour on a rebuild. This is a property of the frozen feature code shared by every learner since v14.
     The verifier now reads the exact cached table (hash-checked against every run's recorded input).
  2. With the exact inputs, the Windows reload differs from the Linux-saved predictions by at most 1.58e-3 s
     (seed 110), above the verifier's a-priori 1e-3 s default. The same reload on Linux (arcw0,
     `reload_check_linux.json`) differs by at most 4.9e-5 s on all 16 seeds; the Windows differences come in
     steps of ~2e-4 s, the float32 rounding of the output at the target scale (`reload_check_windows.json`).
     The tolerance was set to 0.01 s for the final run — above the observed float noise and 50× below the
     0.5 s that could change an exported integer. This was a verifier-parameter change made after seeing the
     float noise; no model, prediction or export changed.
- **Audit** (`tools/audit_export.py --version 24 --compare 23 22 9`, `export_audit_v24.json`): IDs exact, no
  NaN, 10 negative predictions (none below −12; v23 17); extremes 883 / 37 / 12 / 2 rows above 1 h / 4 h / 12 h /
  24 h (v23 944 / 37 / 12 / 2); the same five January airport-months above the 2025 development means and the
  EHAM 2–7 January / LFPG 7 January disruption days as v23; the LIRF fill population (383 rows) unchanged;
  27 rows move by more than 600 s, none by more than 1,800 s.

## Official result — `official_result.json`, `leaderboard_after_v24.txt`

Uploaded with the user's "upload it" (`tools/prc_submit.py upload --calendar-day`, first upload of 23 Sep UTC;
the uploader re-checked the live ranking/template inputs); processed 02:06:13 UTC 23 September 2026:
**235.3074 on all 344,841 rows, −3.5123 s (−1.471%) versus v23's 238.8197**. The December development read was
−1.462% and the four-month mean −0.628%: the transfer again matches the December read (v22: −0.25% → −0.19%;
v23: −0.11% → −0.13%; v24: −1.46% → −1.47%). **v24 is the incumbent. Never re-upload v1–v24.** Standings
snapshot for the record: `leaderboard_after_v24.txt` (first of 178 scored teams).

What this suggests next (not declared): the network gain came entirely through the non-LIRF committee table.
A LIRF-inclusive network for the LIRF learner table, more seeds (the October curve was still rising at 8), and a
network member for the residual stage are the natural follow-ups; each needs its own declaration.
