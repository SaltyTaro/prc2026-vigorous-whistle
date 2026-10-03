# Own-motion bound (`configs/own_motion_v1.json`) — ACCEPTED by its rule (3 October 2026)

This layer was found in dig tunnel 17 (`docs/dig/case_files.md`). It was declared at 14:45 UTC and committed in `bb21fb4` before the
governing run. The exploration behind it is disclosed in the declaration.
- Governing tool: `tools/run_own_motion.py`.
- Results: [`results_v1.json`](results_v1.json).
- Ranking dry-run audit: [`dry_v62_audit.json`](dry_v62_audit.json).

No official score was used and no leaderboard was read. Nothing was written under `runs/submissions` and nothing was uploaded. A
build and an upload are the user's word.

## Mechanism

Some departures at EDDF, EGLL and LFPG are not traced from the gate by the shipped ADS-B matcher (they are dark or unmatched). The
matcher's segment starts after the last coverage gap, so earlier motion never reaches the features. The aircraft can still be seen
moving through its ICAO address (`runs/intruder_identity/ids.parquet`, from the adsb.lol day files):

- its latest identified arrival at the airport blocked in at the **departure's own stand**, 10 min to 18 h before take-off;
- its first ground point at **≥ 3 kt** after that in-block + 120 s lies **B** seconds before take-off, with 60 < B ≤ 2,700 s.

The off-block is not earlier than that motion, so label ≥ B. Where v61's prediction p < B, the new value is p + w (B + o − p).
o(airport) is the mean of label − B over the fitting months' candidates, shrunk with weight 20 to the pooled mean. For the ranking it
is fitted on all twelve 2025 months: EDDF +142, EGLL +273, LFPG +470 s (pooled +183 s), from 687 / 106 / 41 fitting rows. Validity of
the bound on those rows is 96.5% / 93.4% / 100%.

Dig tunnel 15 (`docs/dig/own_bounds.md`) closed a different idea: hard bounds from the departure's own **matched** segment, which v61
already respects or which fail. This layer uses points found by identity after the aircraft's own inbound at the same stand, on rows
the matcher does not trace from the gate, with B capped at 2,700 s.

## Governing run

The control is the v61 development proxy (`runs/dig/census61/dev_v61.parquet` `v61`, SHA-256 `9739d9c5...`). It reproduces census61's
month RMSEs exactly: 321.01656591795944 / 252.4266617291699 / 183.21452377717506 / 165.12517551018345 s. The twelve-month control is
rows12 `stack` after the accepted intruder bound and the identity extension, both applied month-blocked.

The 2025 trigger, before the prediction is involved, covers 205,868 departures. Columns are the change in month RMSE with its
paired-day 95% interval:

| | January | July | November | December |
|---|---|---|---|---|
| **om_b2700_w100 (candidate)** | −0.0380% [−0.136, −0.014] | −0.1103% [−0.191, −0.045] | −0.0814% [−0.188, +0.015] | −0.1630% [−0.252, −0.084] |
| rows | 76 | 64 | 92 | 95 |
| validity on moved rows | 98.7% | 98.4% | 94.6% | 95.8% |
| gain / squared change | 1.14 | 1.28 | 0.40 | 0.92 |
| om_b2700_w50 (sensitivity) | −0.0273% [−0.094, −0.011] | −0.0767% [−0.130, −0.034] | −0.0920% [−0.156, −0.043] | −0.1256% [−0.182, −0.074] |

**Twelve months**, candidate:
- January to June: −0.050 / −0.010 / −0.012 / −0.025 / −0.039 / −0.015%
- July to December: −0.152 / −0.024 / −0.018 / −0.040 / −0.113 / −0.169%
- 12 of 12 better, 10 of 12 intervals below zero, mean −0.055%.

The sensitivity variant is also 12 of 12 better, with 12 of 12 intervals below zero and a mean of −0.043%.

**Rule (v57's), candidate om_b2700_w100:**

| clause | requirement | reading | result |
|---|---|---|---|
| (1) | no development month worse by more than 0.005% | worst month −0.038% (January) | pass |
| (2) | equal-footing mean at most −0.02% | −0.1037% | pass |
| (3) | every equal-footing interval below zero (upper ends) | January −0.014, July −0.045, December −0.084 | pass |
| (4) | every twelve-month month better or unchanged | 12 better, 0 worse | pass |

**Accepted: `selected = om_b2700_w100`.** The sensitivity variant passes as well; it is reported only and cannot be selected.

## Ranking dry run on v61 (label-free)

Command: `tools/run_own_motion.py --dry-run --version 62`. Base: the uploaded v61 (`fe137f09...`, read only).

**Output:** `runs/own_motion/dry_v62/vigorous-whistle_v62.parquet`
- SHA-256 **`cba143f84debdca0d9aff345ff4669ff6a3f0f7f5c4e02d08bcd3e0d02c32d9d`**, 2,225,978 bytes.
- A second run into a scratch directory reproduced it byte for byte.

**What changed:**
- **155 rows**: EDDF 89, EGLL 50, LFPG 16; January 2026 79, July 2026 76. The ranking trigger covers 44,147 departures.
- Squared change **4.08e7**. Every move is upward: minimum +142 s, maximum **+2,270 s**. Mean moves are EDDF +317, EGLL +421,
  LFPG +812 s.
- On the moved rows, values go from 256–2,515 s (median 1,247) to 479–2,912 s (median 1,624). At most 6 rows change on one
  airport-day.

**What did not change:**
- No LIRF row, no LIRF population row and neither guard row changes.
- Every other row is byte-identical to v61.
- IDs, order and types (float64 / int32) are unchanged, and `validate_submission` passes against the template.

`tools/audit_export.py` reads only `runs/submissions`, so it was not run; the equivalent checks are in `dry_v62_audit.json`.

**Expected effect:** about −0.24 s if the scored months follow the development gain ratio (0.88). The range is −0.11 to −0.35 s over
the development months' ratios (0.40–1.28).

## Build (the user's word only)

    tools/run_own_motion.py --build --version 62 --user-decision "<the user's words>"

This reruns the dry run into `runs/own_motion/build_v62`, which must be new and empty, and requires byte identity with `dry_v62`. It
refuses to overwrite and writes `runs/submissions/vigorous-whistle_v62.parquet`, its receipt (from `validate_submission`) and
`docs/own_motion/export_v62.json`.

The final-file chain gains a step 9 (`docs/final_file/v61_chain.md`).
