# Six-month refit of the chain's late learners (10-11 October 2026)

**Why.** The final file adds February and June 2026. Every residual learner after the model part was fitted on the four development
months (January, July, November, December 2025): it has never seen a February or a June. Tier B (`configs/febjun_tierb_v1.json`) now
builds February / June 2025 chain tables that are out of sample for every component. Those tables are extra training rows of the
seasonal analogues. This gamble refits the late learners with them and keeps everything else frozen. The user asked for gambles worth
taking on 10 October ("we fight till the end"). Declaration: `configs/sixmonth_refit_v1.json`. Tool: `tools/run_sixmonth.py`.

## Learners in scope

| layer (Tier B reading) | learner | shipped fit | training target |
|---|---|---|---|
| C6 guarded stacker (step 6) | `stack_weight_v1` `w65_bag10`: v2 bag (46 columns, LIRF), v3 bag (65, other airports), OPDI bag (79, LSZH with an OPDI match); ten seeds each; weight 0.65 | `run_stack_resid.fit_bag` on the four development months' non-fixed rows; early stopping on days with day-of-month % 5 = 0 | `clip(label - new, +-3600)`, new = the chain entering step 6 |
| C14 second pass | `stack_second_v1` (candidates `sp_hicov7`, `sp_nolirf`; the chain re-selects on each file) | five-seed LightGBM bag, 400 rounds, on the development table's non-fixed rows of the candidate's airports | `clip(label - v71, +-1800)` |
| C15 third pass | `stack_third_v1` (candidates `t3_ro_h7`, `t3_ro_3`) | five-seed bag, 400 rounds, fit airports per candidate | `clip(label - v72, +-1800)` |

The W25 season passes and the flight-number memory act on January 2026 only and stay as shipped. The published-day layers, the bounds
and the neighbour-state layers have no learner fitted on development rows in the same sense. They are out of scope.

## The February / June 2025 training rows

They come from the Tier B chain's work directory (`runs/febjun_tierb/chain/work`). These are the same tables the chain builds for the
final file, so the features are computed by the ranking path, exactly as on the final file:

- **C6:** `steps/stack/stacker_rank.parquet` (the 79 `po` columns; X46 and X65 are their prefixes), `steps/stack/preds.parquet` (`new`,
  fixed rows, guards). Target `clip(label - new, +-3600)`; day of month from the take-off time, for the shipped validation split.
- **C14:** `late/stack_second/rank_table.parquet` (features, `v71`, `kind`, `ap`). Target `clip(label - v71, +-1800)`.
- **C15:** `late/stack_third/rank_t3.parquet` (features including the second pass's prediction and v72). Target `clip(label - v72, +-1800)`.

Labels come from `data/raw/training_2025-*.parquet`. Only **lane M's rows read** enter as training rows: clock-present departures with
an out-of-sample v9, the readings' governing population. Fixed rows (LIRF population and guard rows) never enter. Placeholder rows
(missing clock, LIRF population) carry an airport-month median as their model part, so their residuals are not learnable targets.

## What stays frozen

Features, LightGBM parameters, seeds (ten for C6, five for C14 / C15), rounds and early-stopping rule, weights (0.65 / 0.5), caps and
floors, airport sets, candidates and their clauses 1-3 (read on the stored development predictions, as the chain does). C6's guards
keep their shipped support: the four development months' non-fixed rows. That also lets the final early chain's `--stack-preds` hook
accept the predictions unchanged. C14 / C15's transfer guard (clause 4) is recomputed on each file, exactly as the chain does.

## Control (before any February / June table)

`tools/run_sixmonth.py control --kind stack|ss|st`, on an Azure VM. With no extra rows the new code must reproduce the shipped
ranking-mode predictions exactly:
- C6: `runs/final_early_chain/rehearsal_shipped3/steps/stack/preds.parquet` (the ranking-file rehearsal that reproduces v69 byte for
  byte), all 30 bag columns and the three guards;
- C14: `runs/stack_second/preds_rank.parquet`, both candidates;
- C15: `runs/stack_third/preds_rank.parquet`, both candidates.

Records: `runs/sixmonth/control_<kind>.json`, copied to `docs/sixmonth/control/`.

**Result (arcw2, 10 October 18:35-18:49 UTC, one job at a time): every control is exact.**
- **C6:** 30 bag columns, max |difference| 0.0. Guards h2 / h3 / ho equal. The w65_bag10 value through `run_stack_weight.rank_value`,
  rounded, equals the rehearsal's step-6 file on every row.
- **C14:** `sp_hicov7` and `sp_nolirf`, max |difference| 0.0, same NaN pattern.
- **C15:** `t3_ro_3` and `t3_ro_h7`, max |difference| 0.0, same NaN pattern.

Details:
- Each step ran under strace. The only paths the tool looked for and did not find were the default `--work` directory (resolved, not
  read by the controls) and the output files themselves. No input was skipped.
- The controls ran on revision `751a1c33...` of the tool (`docs/sixmonth/control/run_sixmonth_control_rev.py`). The committed revision
  only adds the final-file functions and their command-line wiring; every function the controls use is textually identical (checked
  with `ast`).
- An earlier attempt ran C6 and C14 at the same time. Two 16-thread LightGBM jobs on 16 cores slowed both to minutes per fit, so they
  were stopped and the controls rerun sequentially. The first try had also lacked `runs/lomo_dev/rows.parquet` on the VM, and failed on it
  loudly.

## Evaluation (declared in `configs/sixmonth_refit_v1.json`)

For held month h in {February, June 2025} the learner is refitted on the four development months plus the *other* Tier B month
(five months), and predicts h's rows of F (`held`). Each layer is read at its position in the chain: every earlier layer is as in the
Tier B readings, and the base is the layer's own input file in the Tier B chain.
- **Control:** the Tier B chain's output of that layer (four-month learner).
- **Candidate:** the same layer with the five-month learner's predictions on h. For C14 / C15, the layer's selection and clause 4 run on
  both months pooled, as the chain does.
- **Population:** lane M's rows read in month h.
- **Statistics:** complete-month RMSE change, and the paired-day 95% interval (`run_intruder_bound.interval`, 2,000 replicates, one
  generator seeded 20261004 per learner, February then June).

A learner's six-month refit is accepted only if **both months improve and both intervals lie entirely below zero**. For C6 there is also
a transfer check in the style of `stack_weight_v1` clause 4. Per airport, the RMS of the guarded bag prediction on h is compared with the
stored out-of-month development RMS: the refit fails if the four-month bag is inside 0.5-2x there and the five-month bag is outside.

If accepted, the final file's chain uses that learner fitted on all six months: the development rows plus both Tier B months' rows read.
Otherwise it keeps the shipped learner.

## The final file, if a learner is accepted

`tools/run_sixmonth.py fit-final` refuses unless the declaration is committed and the readings accepted that learner.
- **C6:** `fit-final --kind stack --target <early-chain out>/steps/stack --out <preds file>` writes a preds file in which only the 30 bag
  columns change. `tools/final_early_chain.py --stack-preds <preds file>` takes it: that hook checks the rows, base and guard columns
  against its own recomputation. Steps 0-5 of the final candidate do not change.
- **C14 / C15:** inside the late chain, run `fit-final --kind ss|st --target <late work>/stack_second|stack_third` in place of
  `final_late_chain.py fit --stage ss|st`, after the build step and before the score step. It writes `preds_{dev,rank}.parquet` in the
  shipped layout, with the stored development predictions.

## Status

- [x] control C6 / C14 / C15 (exact)
- [x] declaration committed (4465d52)
- [x] Tier B base complete
- [x] held fits and readings, on arcw6, 22:55-23:28 UTC 10 October

## Readings (`docs/sixmonth/results_v1.json`, SHA-256 `a152728f...`)

Six held fits ran on arcw6, one after another, under strace: the C6 bags (30 fits per month), then C14 and C15 (2 candidates x 5 seeds
per month). No input was missing. The population is lane M's rows read: February 142,622 and June 180,640. Each figure is the change of
the five-month learner against the Tier B chain's four-month learner, at the layer's position, with its paired-day 95% interval.

| layer | February 2025 | June 2025 | verdict |
|---|---|---|---|
| C6 guarded stacker | -0.0325% [-0.0735, +0.0041], 99,081 rows moved | -0.1016% [-0.1750, -0.0380], 125,149 | **not accepted** (February interval includes zero) |
| C14 second pass | -0.0062% [-0.0187, +0.0046], 60,388 | -0.0144% [-0.0240, -0.0050], 80,376 | **not accepted** (February interval includes zero) |
| C15 third pass | -0.0039% [-0.0070, -0.0013], 25,320 | -0.0073% [-0.0151, -0.0019], 35,751 | **ACCEPTED** |

Supporting checks:
- **C6:** no transfer failure (the five-month bags' per-airport RMS ratios stay within 0.5-2x wherever the four-month ones are; the
  lowest is LIRF February, 0.52 against 0.71). Every row of the chain's step-6 file is the bag arithmetic.
- **C14 / C15:** the six-month variants select the same candidates as the chain (`sp_hicov7`, `t3_ro_3`), and no airport falls
  outside the transfer range.

All three refits point the same way: lower RMSE in both months. Only the third pass clears the declared bar, and its gain is small. RMSE
on the governing rows goes 178.179 -> 178.172 s in February and 185.306 -> 185.292 s in June (Tier B positions).

**Final file:** C15 uses `t3_ro_3` fitted on all six months: the development rows plus both Tier B months' rows read. C6 and C14 keep
the shipped learners. On the final file the third pass applies to every month's rows at EDDF / EHAM / LSZH, as shipped.
