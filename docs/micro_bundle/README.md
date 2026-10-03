# Micro bundle (A + B + C) — NOT ACCEPTED by its declared rule (3 October 2026)

Declaration [`configs/micro_bundle_v1.json`](../../configs/micro_bundle_v1.json) (09:55Z, before the governing run); tool
`tools/run_micro_bundle.py --score`; results [`results_v1.json`](results_v1.json). No official score, no leaderboard, nothing under
runs/submissions. LightGBM at 4 threads (the shipped chain used 6): the rebuilt development controls reproduce `ctl_v55m` to <= 2e-12 s,
and the ranking chain without the pieces reproduces c67f20d6 byte for byte (and, through stack_resid_v2 and the intruder bound, v60's
30021455... byte for byte: `runs/micro_bundle/repro_chain`, `runs/stack_resid/dry_v958`, `runs/micro_bundle/repro_v60`).

## Governing run (control: ctl_v55m + the intruder bound; candidate: C + B in the chain, intruder bound, A)

| month | change | paired-day 95% | squared-error change on C rows / A rows / other |
|---|---:|---|---|
| January 2025 | -0.0081% | [-0.028, -0.001] | 0 / -2.38e6 / -0.20e6 |
| July 2025 | **+0.1504%** | [-0.067, +0.625] | **+4.59e7** / -9.26e6 / +0.12e6 |
| November 2025 | -0.0004% | [-0.003, +0.003] | -0.02e6 / +0.10e6 / -0.13e6 |
| December 2025 | -0.0053% | [-0.021, +0.004] | 0 / -0.58e6 / +0.10e6 |

Equal-footing mean **+0.046%**; rules (1) no month worse, (2) mean <= -0.02%, (3) intervals below zero all fail. Twelve-month check:
12 of 12 months better, mean -0.050% (February -0.15%, June -0.21%); rule (4) passes. **Not accepted.**

## Why: a flaw in C's detection

The declared detection lets a pair move a row to the LATER day even when its take-off precedes that midnight; the declared bound
(value <= take-off - midnight - 60 s) then sets a physically impossible negative value. C's other ranking repairs: LEBL 1/2 July and
EHAM 18/19 January. A (MARS families) gains in three of four months (-1.2e7 squared error over the four);
B fired on no development row and on eight twelve-month days.

## Ranking dry run (as declared; do not build)

`runs/micro_bundle/dry_v61/vigorous-whistle_v61.parquet`, SHA-256 `0c76263e0ba3d219162bc558941910888952044aa40e1d58fecefb8250e8d0e2`
(chain with C + B `cf3e6a1e...`, stack_resid_v2 `363f2dd7...`, intruder bound `726d59eb...`, then A on 15 rows at EGLL / LTFM). Against v60:
20,205 rows differ, squared change 1.64e8 (EHAM 1.39e8, LTFM 2.09e7, EGLL 3.4e6), moves -3,355 to +2,156 s; guard rows unchanged.
B's ranking days: EHAM 6 January (I' +113 s), LTFM 15 and 19 January (-83 / -199 s).
