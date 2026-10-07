# Ten-seed bag of the variance re-allocation (`configs/var_alloc_bag_v1.json`) — NOT accepted (7 October 2026)

Fork V, at the user's word ("yes", to "ten-seed bag for the variance re-allocation layer"; expectation stated beforehand: under 0.02 s).
Declared and committed before any reading (7d7c909, 08:26:53 UTC). Tool `tools/run_var_alloc_bag.py`; result
[`results_v1.json`](results_v1.json) (`011219ff...`); runs `runs/var_alloc_bag/` (git-ignored). No official score, nothing under
runs/submissions, nothing uploaded.

## What was tested

v57's variance re-allocation (step 5 of `docs/final_file/v69_chain.md`) weights each eligible row by exp(variance model) x P(late), two
LightGBMs fitted with one seed. Candidate `va_bag10`: both models fitted with ten seeds (20261002, 11-19; only `seed` changes),
w = exp(mean log-variance) x mean P(late), k refitted by the shipped least-squares rule; everything else unchanged. Development as
`tools/run_pubrefit.py`'s pinned ship layers (ctl and cand, own_all_w100 mixed tables), then v69's stacker (w 0.65, ten seeds) fitted on the
shipped development table and predicting from the candidate's, then the intruder bound, identity extension and own-motion v1 / v2.

## Identities (all hold)

- (a) single-seed path = `runs/pubrefit/layers/ship_{ctl,cand}` to 4.5e-13 s; (b) single-seed chain = v69's development proxy (`c088e409...`)
  to 4.5e-13 s.
- Ranking: the single-seed step 5 reproduces the shipped `Y/vigorous-whistle_v66.parquet` (`12a50ea0...`) byte for byte, and steps 6-10 on it
  reproduce **v69 (`192c4a30...`) byte for byte**.

## Governing reading (va_bag10 against the same-path single-seed control; paired take-off-day 95% interval, seed 20261004)

| month | control | va_bag10 | change | interval |
|---|---:|---:|---:|---|
| January 2025 | 320.0147 | 320.0156 | +0.0003% | [-0.0011, +0.0017] |
| July 2025 | 250.8358 | 250.8313 | -0.0018% | [-0.0047, +0.0009] |
| November 2025 | 181.8835 | 181.8829 | -0.0003% | [-0.0058, +0.0047] |
| December 2025 | 163.3537 | 163.3527 | -0.0006% | [-0.0051, +0.0034] |

Equal-footing mean **-0.0007%** (bar -0.02%); no equal-footing interval below zero; clause 1 holds. Transfer guard holds (ratios 0.85-1.71).
**Not accepted; no ranking candidate file.** The bag moves the layer's value by RMS 0.52 s (max 24 s) and the final chain by a few tenths of a
second per row: the single-seed weights were not noisy enough to matter. v69 stays.

## Compute

arcw1 only (08:30-09:20 UTC): workspace on the temp disk from ~/prc-v67 (723 cached inputs re-hashed against the workstation: equal), missing
inputs fetched lazily through a reverse tunnel (62 files, all re-hashed: equal). Two stack months lost a concurrent fetch of the same file at
start and were rerun (logs kept). Results mirrored (`vab_out.tar` 716aa311..., hash equal), arcw1 deallocated, server and tunnel stopped.
All eight VMs deallocated.
