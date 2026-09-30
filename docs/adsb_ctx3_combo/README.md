# Context network, combination batch (29 September 2026)

Declaration:
[`configs/adsb_ctx3_combo_v1.json`](../../configs/adsb_ctx3_combo_v1.json), committed before any fit (db4d304, 12:00 UTC).
Report: [`design_report.json`](design_report.json) (`tools/ctx3_combo_report.py`). October only: no scoring month, no
ranking row and no official score is read. Nothing is exported or uploaded.

## Question

The GPU sprint (`docs/adsb_ctx3/README.md`) read three items that leaned positive and failed their bar alone: numeric
embeddings 16 wide (+0.033 points), a doubled learning rate (+0.027), and a network trained with a quarter of its matched
rows shown as unmatched, used on unmatched rows only (+0.054, interval touching zero). They were never fitted together.

## Design

- Four variants on the 32-epoch base, seeds 9–12, the October design fold, `tools/run_adsb_ctx3.py` unchanged:
  `ref32`; `dl32` (embeddings 16 wide and learning rate 0.002); `m25_32` (masking 0.25); `m25dl32` (both).
- Hardware: CPU, arcw0–arcw7, two lanes per worker, 12:25–16:15 UTC, 3.2–3.8 hours per fit. The reference was refitted on
  the same hardware. All eight workers deallocated afterwards; 96 run files mirrored and hash-checked.
- Items: `combo_dl` (dl32 on every row); `route_m25` (ref32 on matched rows, m25_32 on the others); `combo_all` (dl32 on
  matched rows, m25dl32 on the others).
- Rule: GO when the recipe proxy at member weight b = 0.5 (network share 0.75) is at least 0.05 points below ref32's, the
  four-seed standalone RMSE is below ref32's, and the paired-day interval against ref32's table is below zero.

## Result: two items pass

October non-LIRF rows; control proxy 176.262 s; ref32 reads +0.047% at b = 0.5.

| item | standalone against ref32 | matched / unmatched RMSE (s) | gain at b = 0.5 (points) | interval (s) | gain at b = 1 | decision |
|---|---:|---:|---:|---|---:|---|
| ref32 | 156.847 s | 117.79 / 212.13 | | | | reference |
| `combo_dl` | +0.24% | 118.34 / 212.35 | +0.049 | [−0.26, +0.08] | 0.000 | no |
| `route_m25` | **−0.66%** | 117.79 / **209.91** | **+0.092** | [−0.26, −0.07] | **+0.190** | **GO** |
| `combo_all` | −0.30% | 118.34 / 210.55 | **+0.107** | [−0.36, −0.04] | +0.123 | **GO** |

- **The masked network is the gain.** On unmatched rows it reads 209.91 s against 212.13 s (−1.05%). All four masked seeds
  are below the reference's seed mean. This is the second independent measurement of the routed item: the GPU sprint read
  +0.054 points on seeds 5–8, this batch +0.092 on seeds 9–12.
- **`route_m25` is better at every airport** (−0.07% to −1.24% against ref32). `combo_all` is mixed: better at five
  airports, worse at EDDM (+1.9%), EHAM (+2.4%), EGLL and LSZH.
- **The wider embeddings and the doubled learning rate do not add accuracy.** Alone they are worse than the reference
  (+0.24%), and they make the masked network worse on unmatched rows (210.55 against 209.91 s). Their proxy gain at b = 0.5
  comes from being different from the pool, and it is gone at b = 1.
- At network share 0.85 the picture is the same (`route_m25` −0.042%, `combo_all` −0.061% at b = 0.5).

## What is carried forward

By the declared tie-break (lowest proxy at b = 0.5) `combo_all` is carried forward. The two items differ by 0.015 points at
b = 0.5, which is inside the noise, and the other readings favour `route_m25`: standalone, b = 1, every airport, and two
measurements. `route_m25` also needs half the fits, because its matched rows keep networks that already exist. Which item
goes to the development round is put to the user with this record; carrying `route_m25` would be a disclosed departure
from the tie-break.

## Next

A development round, declared separately before any development score: the masked network at 16 seeds on the clean line's
footing (leave-one-month-out January and July 2025, the forward fold, the January–November refit, the twelve-month refit),
as a member for unmatched rows only, control = the v46 stack, scored with `tools/run_network_member.py`. Every upload is the
user's decision.

## Development round (declared 17:35 UTC 29 September; fits 29–30 September on three L40S nodes)

`configs/adsb_ctx3_combo_dev_v1.json`: the user chose the full combination (`combo_all`) and GPU hardware. Member = per seed and leaf,
the dl32 network on rows with `adsb_matched = 1` and the m25dl32 network on the other rows (`tools/build_combo_member.py`); LIRF rows
are left to the shipped table by the scorer. Seeds 601–616, folds leave-one-month-out January and July 2025, the forward fold
(November, reported only), the January–November refit (December) and the twelve-month refit (ranking): 160 fits with
`tools/run_adsb_ctx_lomo.py --device cuda`, two lanes per GPU, 13–16 minutes per fit. Candidates: member weight 0.15 / 0.25 / 0.40
inside the network table; control = the v46 development stack; the clean line's unchanged rule.

AWS supplied three of the four nodes (the fourth shut itself down after a failed runtime set-up); the round ran on three nodes with
two lanes each from 21:00 UTC. Two fits finished before a carriage-return fault in the job lists was fixed (`tools/combo_jobs.py`,
commit 6b889fc); their run folders were renamed, four unfinished fits restarted. No fit failed; every fit ran 32 epochs; the
monitors flagged one forward fit (m25dl32 seed 608) for a last-epoch training-loss bump, its October and November readings are in
line with its peers.

**Governing run (`docs/adsb_ctx3_combo/results_v1.json`, 03:05 UTC 30 September): every candidate is eligible; `w40` is selected.**

| candidate | January 2025 | July 2025 | December 2025 | mean (rule months) | November 2025 (reported only) |
|---|---:|---:|---:|---:|---:|
| control (v46 stack) | 322.8585 | 257.7185 | 167.5137 | | 184.9050 |
| `w15` | −0.037% [−0.23, −0.07] | −0.040% [−0.16, −0.06] | −0.066% [−0.15, −0.07] | −0.048% | −0.032% [−0.12, +0.01] |
| `w25` | −0.057% [−0.34, −0.11] | −0.060% [−0.24, −0.08] | −0.094% [−0.23, −0.09] | −0.070% | −0.045% [−0.18, +0.02] |
| **`w40`** | **−0.066%** [−0.41, −0.11] | **−0.071%** [−0.33, −0.05] | **−0.119%** [−0.31, −0.10] | **−0.086%** | −0.044% [−0.25, +0.11] |

Brackets: paired-day 95% interval of the RMSE difference (seconds). December: 25 of 31 days better at `w40`.

- **The gain grows with the weight up to the largest declared weight.** The alignment of the change (the factor by which the change
  would have to be scaled to minimise the error) is 1.5 / 1.2 / 0.8 / 1.1 at `w40` on January / July / November / December, so 0.40
  sits near the optimum; no weight was added after the scores.
- **Standalone the member is worse than the shipped network table** (318.67 against 317.88 s in January, 202.17 against 200.84 s in
  July, 148.90 against 147.04 s in December): it helps as an added member, not as a replacement. On unmatched non-LIRF rows it is
  about equal to the table (447.58 against 447.58 s in January); the gain of the stack comes from the mixture.
- By rows: unmatched non-LIRF rows −0.07% / −0.20% / −0.11% / −0.18% (January / July / November / December) at `w40`; matched
  non-LIRF rows −0.09% / +0.03% / −0.03% / −0.16%. The July matched rows are the one slice that does not improve.
- On the ranking file the reading is worth about −0.1% to −0.12% (December is the closest month): about −0.25 s on v46's 223.93.

**Disclosed interim readings.** While the other fits ran, the scorer's code was run on the finished months (January and July at
00:10 UTC; January, July and December at 01:45 UTC; `runs/adsb_ctx3_combo/dev/interim_*.json`, `interim3_*.json`). They read the
same numbers as the governing run (the tables are the same); the rule and the candidates were fixed before either. Nothing was
changed after them.

**Tool fix before the December leaf.** `tools/build_combo_member.py` read the match flags from `runs/adsb_nn/features`, which lacks
December's table. It now reads the tables the fits read (`runs/adsb_ctx2/inputs/adsb_features_arcw0`, twelve months and the ranking
rows) and checks every table's hash against every fit's record (`inputs.adsb_cache_sha256`). The January and July tables were rebuilt
with it and are unchanged (the two caches carry identical flags for those months).
