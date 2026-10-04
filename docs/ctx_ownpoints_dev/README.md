# Context network with the own-aircraft turnaround columns: development round (4 October 2026)

Dig tunnel 21's development round. The design batch (`docs/dig/ctx_ownpoints.md`, October fold) read the 15 own-aircraft turnaround
columns of tunnel 18 as network inputs at GO; the user chose "Start it on the GPU". Declaration
[`configs/ctx_ownpoints_dev_v1.json`](../../configs/ctx_ownpoints_dev_v1.json) (f3a3e69, 01:23 UTC, before any fit and any score), network
settings [`configs/ctx_ownpoints_network_v1.json`](../../configs/ctx_ownpoints_network_v1.json), GPU authorisation
`infra/aws/gpu_authorization_20261004.json`. No official score enters any step; nothing is uploaded by this round.

## Question

The shipped combination member (v51 onwards: dl32 on ADS-B-matched rows, m25dl32 on the others, weight 0.40 in the committee's network
table) never sees the aircraft's own ADS-B points between its inbound in-block and the matcher's last coverage gap. Does the same member
with those 15 columns (`dl32own` / `m25dl32own`), **replacing** the combination member, improve (stage 1) the model part and (stage 2)
the whole v64 chain, which already carries the same information in stack_resid_v3?

## Fits

- 2 variants x 16 seeds (601-616) x 5 folds (leave-one-month-out January and July 2025, the forward fold for November, the
  January-November refit for December, the twelve-month refit for the ranking file) = 160 fits, `tools/run_adsb_ctx_lomo.py --device cuda`
  unchanged, `--extra-features data/external/adsblol/features_v19own` (14 tables hash-checked against
  `docs/dig/ctx_ownpoints_features_manifest.json` locally and on every node), through `tools/combo_queue.sh` (new optional `EXTRA`;
  default behaviour unchanged) and `tools/combo_nodes.sh`.
- Hardware: four AWS g6e.2xlarge (one L40S each) via SkyPilot, two lanes per GPU, launched 01:26 UTC; every node's token build equal to
  the workstation's and the context tests passing before the fits; a 20,000-row smoke fit on the head (removed). Queues from 02:05 UTC;
  about 17.6 minutes per development fit; no fit failed; every development fit ran 32 epochs (last-epoch training loss 0.081-0.092 on the
  July fold).

## Stage 1 (member level; `tools/run_ctx_ownpoints_dev.py`, [`results_v1.json`](results_v1.json))

Control: the v46 development stack with the combination member at 0.40 (v51's model part); it reproduces the combination round's
recorded w40 RMSEs on all four months before any candidate is read, and an identity run (the combination standing in for the candidate)
changes no row. Candidate: the combination removed and the new member in its place at the declared weight.

| candidate | January 2025 | July 2025 | December 2025 | mean (rule months) | November (reported) | eligible |
|---|---|---|---|---|---|---|
| own_all_w15 | -0.027% [-0.29, +0.05] | -0.077% | -0.121% | -0.075% | -0.132% | no (January) |
| own_all_w25 | -0.086% | -0.173% | -0.256% | -0.172% | -0.232% | yes |
| **own_all_w40** | **-0.161%** [-1.08, -0.26] | **-0.282%** [-0.95, -0.52] | **-0.423%** [-0.90, -0.53] | **-0.289%** | -0.358% [-1.01, -0.40] | **yes, selected** |
| own_matched_w15 | +0.005% | -0.016% | -0.026% | -0.012% | -0.050% | no |
| own_matched_w25 | -0.029% | -0.065% | -0.105% | -0.066% | -0.100% | yes |
| own_matched_w40 | -0.061% | -0.116% | -0.199% | -0.125% | -0.149% | yes |

Brackets: paired-day 95% interval of the RMSE difference in seconds. The selected candidate improves matched non-LIRF rows by 0.57-0.93% and
unmatched non-LIRF rows by 0.12-0.62% in every month; standalone, on its rows, the new member beats the shipped network table in January,
July and December (317.41 / 199.34 / 146.68 s against 317.88 / 200.84 / 147.04 s), which the combination member did not. Disclosed: the
alignment of the selected change is 2.1-2.4 (the gain would keep growing above 0.40); no weight outside the declared grid was read.

## Stage 2 (v64's chain; `tools/run_ctx_ownpoints_chain.py`, [`results_chain_v1.json`](results_chain_v1.json))

Control: the v64 development proxy (the v63 proxy with own_motion_v2 last). Candidate: the development model part rebuilt with own_all_w40
in place of the combination (v52m-v55m unchanged), then no-NM rematch, v48, extreme-day, variance re-allocation with their shipped
parameters, stack_resid_v3 refitted on the rebuilt components, intruder bound, identity extension, own-motion v1 and own_motion_v2.
Before any candidate: the combination at 0.40 reproduces `runs/dev_model_part/rows.parquet` exactly, and the shipped files through the
same code reproduce the v63 proxy exactly (max 0.0).

| month | v64 proxy | candidate | change | paired-day 95% (%) |
|---|---|---|---|---|
| January 2025 | 320.5824 | 320.3415 | -0.0751% | [-0.283, -0.025] |
| July 2025 | 251.8561 | 251.4886 | -0.1459% | [-0.196, -0.103] |
| November 2025 | 182.5165 | 182.1391 | -0.2068% | [-0.309, -0.129] |
| December 2025 | 164.2566 | 163.8910 | -0.2226% | [-0.282, -0.167] |

v57's clauses: (1) no month worse: yes; (2) equal-footing mean -0.148% (bar -0.02%): yes; (3) every equal-footing interval below zero:
yes. **Accepted.** About half of the member-level gain survives the chain (stack_resid_v3 already used the same points).

## Ranking file: v65 = v64's chain with the member replaced (built at the user's standing word, not uploaded)

`tools/export_ctx_ownpoints.py` (the steps of `docs/final_file/v61_chain.md` plus own_motion_v2). Before the candidate: the same code with
the combination member reproduces v64 byte for byte (`319b6340...`), with every intermediate equal to the shipped one (the v55 model part,
step 5 `c67f20d6...`, v63's stack step). Ranking member tables: the 16 twelve-month refits (`refit12`), routed by the match flag.

- Dry run `runs/ctx_ownpoints_dev/rank/dry_v65/vigorous-whistle_v65.parquet`; build `--build --version 65` reran the chain into
  `rank/build_v65` and matched it byte for byte: **`runs/submissions/vigorous-whistle_v65.parquet`, SHA-256
  `45160819c67ed4827280d6968ca26ec3f09d915d31e83cee62090b7217bf3cc3`** (receipt [`export_v65.json`](export_v65.json); audit
  `docs/export_audit/v65_audit.json`, integrity clean). User decision recorded: "if you get anything build and keep it ready, we will upload
  it when slots reset" (3 October) and "Start it on the GPU" (4 October). Never uploaded by this round.
- Against v64: 301,815 rows differ, squared change 5.34e7, RMS 12.4 s, mean -0.31 s, largest +535 / -863 s; RMS by airport EHAM 18.3,
  LFPG 15.0, EGLL 14.8, EDDF 14.0, LSZH 12.0, LTFM 10.1, EDDM 9.2, LEMD 9.0, LEBL 8.0, LIRF 4.6 s. No LIRF population row and neither guard
  row changes. The 19,610 LIRF rows that move (RMS 4.6 s, at most 44 s) move only in the stacking step, where LIRF takes the v2 learner
  refitted on the rebuilt components, exactly as in the stage-2 development chain that was scored.
- Model part: 265,693 non-LIRF rows of the v55 model part move (the member enters every non-LIRF row with a network value); the own-motion
  v1 step moves 67 rows (v63: 96), own_motion_v2 2,386 (v64: 2,399).
- Expected: the stage-2 change is -0.075 / -0.146 / -0.207 / -0.223% of month RMSE; at a gain of about 1x the squared change about
  -0.15 s, and more if the ranking months behave like the development months' alignment.

GPU: four g6e.2xlarge from about 01:30 to 08:01-08:04 UTC; every mirrored file (4 x 256) SHA-256-equal
to its node; all four clusters torn down; the jobs controller untouched (stopped).
