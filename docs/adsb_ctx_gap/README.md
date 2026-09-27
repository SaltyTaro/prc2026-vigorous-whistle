# Time-gap test (27 September 2026) — [`configs/adsb_ctx_gap_v1.json`](../../configs/adsb_ctx_gap_v1.json), [`report.json`](report.json)

Question: v30 (more 32-epoch network weight) transferred only ~6% of its development gain to the leaderboard, against >100% for the
earlier network rounds, and the ranking months lie 1–18 months after the twelve-month refit while every development fold has a 0–1
month gap. Does a longer-trained network's advantage shrink when the scored months are further from the fitted ones?

Design: the GPU context network (the `ctx_e32` settings) at 16/24/32 epochs, 4 seeds, fitted on January–June 2025 (`gap`) or on
May–October 2025 (`near`, the same six-month size ending just before the scored months), both scoring November + December 2025
(non-LIRF rows, 4-seed means). `tools/run_adsb_ctx_gap.py`, `tools/ctx_gap_chain.sh`, `tools/ctx_gap_report.py`; 24 fits on four AWS
L40S in ~20 minutes.

| fold | g16 | g24 | g32 | matched (g16) | unmatched (g16) |
|---|---:|---:|---:|---:|---:|
| gap (fit Jan–Jun) | 163.694 | **163.189** | 163.651 | 131.55 | **222.32** |
| near (fit May–Oct) | 165.787 | **165.544** | 166.099 | 130.76 | 228.59 |

- **No: longer training does not age worse.** 32 vs 16 epochs: −0.03% with the gap, +0.19% without; 24 epochs: −0.31% / −0.15%.
  Difference-in-differences −0.21 / −0.16 points (intervals [−0.82, +0.36] / [−0.56, +0.23]) — if anything longer training helps more
  across a gap. v30's weak transfer is not explained by epoch count. (On six months of data 24 epochs beat 32; on twelve months 32
  won the development round.)
- **Season beats recency:** fitting on January–June (winter included, 5–6 months older) scores November–December **1.3% better** than
  fitting on May–October, almost all of it on **unmatched rows (−2.7%: 222.3 vs 228.6 s)**; matched rows are equal (131.6 vs 130.8).
  The twelve-month refit weights all months equally → next test: season-weighted training (configs/adsb_ctx_season_v1.json).
