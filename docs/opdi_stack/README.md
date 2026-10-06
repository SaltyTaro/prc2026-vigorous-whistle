# Lever 7: OPDI flight events as stacker inputs — screen GO, declared rule NOT accepted (6 October 2026)

Final-week campaign, reserve lever (user, 6 October: "if all else is closed then we will look into this"). Nothing was built under
`runs/submissions`, nothing was uploaded, no official score or leaderboard entered any step. LIRF population and guard rows
 never change. Hand-off log: [`HANDOFF.md`](HANDOFF.md).

## Data

EUROCONTROL Open Performance Data Initiative (OPDI) flight events v0.0.2 (opdi.aero), derived from OpenSky Network state vectors; open
data with attribution. 23 ten-day files
covering January / July / November / December 2025 and January / July 2026, filtered to surface and take-off events in boxes around
the ten airports (`tools/opdi_stack_fetch.py`); URLs, bytes and SHA-256 in [`sources.json`](sources.json); extracts
`data/external/opdi/stack_v1/` (git-ignored). **OPDI has no data for 2-10 January 2026** (the 2025-12-31 file holds only 31 December
and 1 January). Downloads and filtering ran on Azure arcw4 after the user's instruction (~17:50 UTC: "use the azure vm's please keep as
much compute off this workstation"); only the first four January 2025 windows were fetched locally.

## Features and coverage

`tools/run_opdi_stack.py --features`: departures matched to OPDI flights by airport + callsign (CALLSIGN_flt, else the rematch key
map) with the OPDI take-off reference (take-off event, else last runway entry) within [-900, +300] s of the supplied take-off; only
events up to that reference. 14 columns: take-off minus the OPDI reference, stand exit, apron exit, first taxiway entry, first seen,
runway entry, taxiway count, de-icing flag and duration, match flag, and four times minus the chain's prediction. Matched: 333,684 of
672,428 development and 154,475 of 344,841 ranking departures. Coverage by airport-month: [`coverage.json`](coverage.json). OPDI's
surface coverage moves a lot between months: LSZH stand exits 58-79% (dev), 42% / 78% (January / July 2026); EDDF stand exits 0-1%
in January / July 2025, 15-18% in November / December 2025, 14% / 24% in January / July 2026; LEMD 2-74% by month.

## Screen (in v66's pinned chain; [`screen.json`](screen.json))

stack_resid_v3's guarded learner refitted with the 14 OPDI columns appended; its value replaces v3's only on OPDI-matched rows its
guard does not hold; every other row keeps v3, LIRF keeps v2. Control: the pinned v66 development proxy
(`runs/pubrefit/chain/ship/proxy.parquet`), reproduced exactly (max |diff| 0.0) by the same copied chain on the workstation and on arcw4.

| variant | January | July | November | December | equal-footing mean | GO bar |
|---|---|---|---|---|---|---|
| `opdi` (all non-LIRF) | -0.0322% [-0.104, -0.015] s | -0.0925% [-0.111, -0.076] | -0.116% [-0.167, -0.079] | -0.123% [-0.148, -0.100] | **-0.0826%** | GO |
| `opdi3` (LSZH / EDDF / LEMD) | -0.0277% | -0.0859% | -0.1118% | -0.1002% | -0.0713% | GO |

The gain sits mostly at LSZH (its squared error falls by 0.05-0.20% of each month's total), with small parts at EDDF, EGLL, EHAM, LFPG.

## Declared round ([`configs/opdi_stack_v1.json`](../../configs/opdi_stack_v1.json), 18:53 UTC, before the governing run)

Candidate `opdi`; rule = v57's clauses with own_motion_v2's values plus stack_resid_v3's transfer guard (clause 4: per airport with at
least 200 used rows on both sides, ranking RMS of the OPDI change over development RMS within [0.5, 2.0]). Results:
[`results_v1.json`](results_v1.json) (`tools/run_opdi_stack_decl.py`).

- Governing run (arcw4): the recomputed proxy equals the screen's exactly; clauses 1-3 hold (eq -0.0826%, every interval below zero).
- Ranking dry run: the identity path (unchanged `export_ctx_ownpoints.step_stack` + shipped chain from v66's step-5 outputs) reproduces
  the uploaded v66 byte for byte; the v3 part of the OPDI step reproduces the shipped stack step (max |diff| 0.0). The candidate file
  (`runs/opdi_stack/rank/dry_v67/NOT_ACCEPTED_vigorous-whistle_v67.parquet`, SHA-256 `7699b454...`) uses OPDI on 143,599 ranking rows and
  differs from v66 on 102,117 rows: squared change 1.15e7, RMS 5.8 s, max +368 / -130 s; LSZH 59% and EDDF 32% of the squared change;
  no LIRF, population or guard row.
- **Clause 4 fails**: ranking / development RMS of the OPDI change = EDDF **2.75**, LEBL **2.04**, EDDM **0.42**, LEMD **0.36** (outside
  [0.5, 2.0]); EGLL 0.74, EHAM 0.89, LFPG 0.93, LSZH 1.05 inside. The out-of-range airports are exactly those whose OPDI coverage
  differs most between the 2025 development months and 2026 (EDDF stand exits appear only from November 2025 on; LEMD collapses in
  2026). **Not accepted; nothing selected.** The file stays a renamed dry run.

Observation, not a selection (post hoc, disclosed): LSZH alone carries most of the development gain and transfers (ratio 1.05). An
LSZH-only layer was not declared before these readings; whether to declare one is the coordinator's / user's decision.

## Compute

Azure arcw4 (and arcw5 only to copy its January / February / December 2025 adsb.lol surface days to arcw4; arcw4's VM key was appended to
arcw5's `authorized_keys`). Packages uploaded hash-listed (`runs/opdi_stack/packages.sha256`, `packages2.sha256`, per-file
`surface_missing.sha256`, `rank_deps_files.sha256`), outputs mirrored hash-checked (`runs/opdi_stack/mirror*/`), both VMs deallocated
after their mirrors. The workstation ran one local identity check (before the user's instruction), feature tests and the orchestration.
