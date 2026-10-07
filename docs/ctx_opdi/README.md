# Lever 9: OPDI flight-event columns in the context-network member — design NO GO (7 October 2026)

Rank-1 hunt, fork J. Question: does the shipped own-points member own_all (dl32own on ADS-B-matched rows, m25dl32own on the others; v66 / v68
network table at weight 1.00) gain from ten OPDI flight-event columns, the way the own-aircraft points went stacker (v63) -> network (v65)?
OPDI had already entered the stacker at LSZH (v68). No official score, leaderboard read or upload entered any step; no LIRF population or
guard row is involved (LIRF rows are outside the network's rows). Hand-off log: [`HANDOFF.md`](HANDOFF.md).

## Declaration

[`configs/ctx_opdi_design_v1.json`](../../configs/ctx_opdi_design_v1.json) (ebd568e, before any table or fit), GPU authorisation
`infra/aws/gpu_authorization_20261007.json`. Candidates dl32opdi / m25dl32opdi = dl32own / m25dl32own + the ten OPDI columns, seeds 9-16, October
design fold, L40S; control = lever 5's GPU fits of dl32own / m25dl32own (same seeds, code and hardware; variant hashes checked by the report).
Decision item opdi_all (both networks with OPDI); bar as tunnel 21 / lever 5: +0.05 points at b = 1.0, standalone better, interval below zero.

**Amendment 1** (73e53e9, after the label-free coverage tables existed and before any fit): as declared, the coverage-shift mask compared
stand-exit shares of 0.01-0.08% and blanked nine of ten airports ([`manifest_rule_as_declared.json`](manifest_rule_as_declared.json)); the
stand-exit criterion now applies only where that share is material (>= 2% in 2025 or the ranking months). Masked: EDDF (stand exit 4.5% ->
19.7%), LEBL (3.1% -> 0.13%), LTFM (matched 0.08%).

## Data and features

- OPDI v0.0.2 flight events (open, organiser 4 October): 21 ten-day windows fetched and filtered on Azure arcw4 (`tools/ctx_opdi_fetch.py`,
  lever 7's filter unchanged) plus lever 7's 23 extracts read in place ([`sources.json`](sources.json)). **The window 2-12 October 2025 is not
  published (HTTP 404)**: those October days have no OPDI columns.
- `tools/ctx_opdi_features.py` (lever 7's matching, reproduced exactly on January 2025): ten columns (times before take-off, no chain-relative
  column) appended to the shipped own-points tables for 2025-01..12, 2026-01, 2026-07 ([`features_manifest.json`](features_manifest.json),
  9c9de534...). Matched departures 40-60% per month; LSZH stand exits 67% (2025) / 62% (ranking). Tables mirrored hash-checked; arcw4 deallocated.

## Design result ([`design_report.json`](design_report.json), `tools/ctx_opdi_design_report.py`)

October non-LIRF, 8-seed means; proxy R2 at b = 1.0 (share 0.75) against the control:

| item | standalone (s) | vs control | gain at b = 1.0 (pts) | interval (s) | GO |
|---|---:|---:|---:|---|---|
| control (own_all, lever 5's GPU fits) | 154.123 | | | | |
| **opdi_all** (decision) | 154.115 | -0.005% | +0.041 | [-0.30, +0.15] | **no** |
| opdi_matched (reported) | 153.936 | -0.121% | +0.069 | [-0.24, -0.02] | (reported only) |
| opdi_unmatched (reported) | 154.301 | +0.116% | -0.029 | [-0.16, +0.27] | (reported only) |

- The gain sits on rows with an OPDI stand exit (-13.2%; LSZH -7.6% of its RMSE), which v68's LSZH stacker layer already uses; rows without
  OPDI get worse (+0.22%), mostly through the unmatched network. EDDF -0.6%, EDDM -0.7%, EHAM -1.1%, LEBL +1.1%, LEMD +1.4%.
- **Verdict: NO GO** by the declared rule; no development round was declared, no ranking file built. The reported half item opdi_matched
  would clear the bar, but it was not a decision item and its gain is LSZH-concentrated (-5.0% there), the part v68 already carries; a
  development round on it would be a post-hoc choice and is not proposed here.

## Compute

Azure: arcw4 only (fetch, features), deallocated after the mirror. The workstation ran orchestration,
git, hash checks, one January-2025 matching check and the design report.
