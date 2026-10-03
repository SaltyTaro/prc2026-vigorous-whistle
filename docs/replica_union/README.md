# Replica union (`configs/replica_union_v1.json`): NOT ACCEPTED by its rule (3 October 2026)

This round came from dig tunnel 16 (`docs/dig/replica_union.md`). It was declared at 15:40 UTC and committed in `0bb9eb5`, before the
governing run.
- Governing tool: `tools/run_replica_union.py` (`19ce668`).
- Results: [`results_v1.json`](results_v1.json) against the declared v62 control, and [`results_v1_v63.json`](results_v1_v63.json)
  against the v63 development proxy (`runs/stack_resid/dev_v63_proxy.parquet` column `v63`, SHA-256 `1628a1d6...`). The v63 run is
  required by the declaration's sequencing because v63 was built first.

No official score was used. Nothing was dry-run or written under `runs/submissions`, nothing was built and nothing was uploaded.

## What was run

**Extraction.** The staging-0 replica was extracted for every 2025 day that has one, on arcw0–arcw3. All four VMs were deallocated
after the hash-checked mirror (four tars, `sha256sum -c` OK).
- 349 day files are in `data/external/adsblol/surface_staging/` (git-ignored). This includes 2025-12-31, from `globe_history_2026`.
- 2025-10-04's staging archive is 98 MB and yields no in-box rows, so that day keeps prod-0 alone.
- 2025-10-05 has no staging replica. 2025-05-28 to 06-10 have neither replica.
- The ranking side covers the 52 days of January and July 2026 that have a staging replica.

**Union matches.** The shipped base matcher (`match_departures` with the learned stands) was run on each union day file:
- `runs/replica_union/union_matches_2025.parquet`: 1,203,414 rows, 408,405 gate segments at the known stand.
- `runs/replica_union/union_matches_rank.parquet`: 227,299 rows, 76,022 gate segments at the known stand.
- Re-running the matcher on prod-0 alone reproduces `matches2` exactly (checked on 2026-01-08).

**Candidate rows.** These are rows gate-traced at their known stand on the union, not gate-traced in prod-0's `matches2`, and not keyed
by v35 / v59, at eight airports (LIRF excluded).
- 2025 by month: 28 / 10 / 25 / 60 / 21 / 227 / 240 / 147 / 111 / 743 / 129 / 182.
- One outage day, 2025-10-15, holds 335 of them.
- Ranking: 412 rows.

## Governing run (candidate `ru8_w50`)

| month | v62 control | v63 control | rows (upgraded) | gain / squared change |
|---|---|---|---:|---:|
| January 2025 | +0.0324% [−0.003, +0.194] | +0.0325% [−0.002, +0.195] | 28 (9) | −0.91 |
| July 2025 | **+0.3285%** [+0.088, +0.648] | **+0.3290%** [+0.088, +0.649] | 240 (12) | −1.00 |
| November 2025 | −0.0116% [−0.020, −0.004] | −0.0115% [−0.020, −0.004] | 129 (9) | +0.85 |
| December 2025 | −0.0116% [−0.034, +0.005] | −0.0112% [−0.032, +0.006] | 182 (32) | +0.48 |

The equal-footing mean is +0.116% (v62 control) and +0.117% (v63 control).

**Twelve months:**
- January to June: +0.031 / +0.001 / +0.028 / +0.010 / +0.116 / +0.017%
- July to December: +0.283 / +0.021 / +0.062 / −0.002 / −0.015 / −0.013%
- 3 of 12 months better, 9 worse, mean +0.045%.

**Rule (v57's), on both controls:**

| clause | requirement | reading | result |
|---|---|---|---|
| (1) | no development month worse by more than 0.005% | July +0.329% | **fail** |
| (2) | equal-footing mean at most −0.02% | +0.117% | **fail** |
| (3) | every equal-footing interval below zero | upper ends January +0.195, July +0.649, December +0.006 | **fail** |
| (4) | every twelve-month month better or unchanged | 9 of 12 months worse | **fail** |

**Not accepted.** Every sensitivity variant also fails:
- without the upgraded rows: equal-footing +0.117%;
- with LIRF: +0.117%;
- at w 1.0: +0.490%, 11 of 12 months worse.

## Why it fails (post-hoc reading, after the governing run; no new candidate)

In most months the gain per unit of squared change is close to −1. That means the label sits near the chain's prediction and the union
taxi carries no information. In July the correlation of (T − p) with (label − p) is 0.00.

Many union-only "gate" segments are sparse tracks: 3 to 7 points spread over thousands of seconds. They pass the stationary-at-stand
test only because the second replica adds a point. Their motion start is spurious: T runs from 98 s to 10,905 s against labels of
1,000–1,800 s, at EGLL, LSZH and LFPG.

Dense union tracks (November and December, correlation 0.52–0.63) help a little, but not enough to carry the month. A point-count
filter is a post-hoc variant the governing run did not test. It would need its own declaration, and the exploration has now read
labels. The union also adds nothing at the airports where coverage is missing (LTFM, LIRF July 2026).

**Closed.** The chain stays at v63. Per the sequencing, nothing is built.
