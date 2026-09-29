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
