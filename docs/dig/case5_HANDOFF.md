# Case files round four (2026-like blind spots), fork L: hand-off log

Lever #10 of the rank-1 hunt (7 October 2026). Scratch `runs/dig/case5/` (git-ignored). Control: v68 development proxy
`runs/opdi_stack_lszh/dev_v68_proxy.parquet` (938a91cd...; 320.0400 / 250.8650 / 181.9933 / 163.5496 s). Nothing declared yet.
Compute rule: heavy jobs on Azure arcw6-arcw7 only; the workstation reads small tables.

- 02:20 UTC: started. Plan: (1) blind-row census on the v68 proxy (rows without a surface trace, the 2026 condition at LIRF July and
  most of EDDF); (2) read the worst unexplained blind rows with every non-trace source (supplied arrivals, NM clocks, schedule, OPDI events,
  airborne points); (3) screen any mechanism month-blocked against the v68 proxy.
- 03:10 UTC: census of the v68 proxy by airport x ADS-B state; OPDI hard bounds rejected (worse everywhere); OPDI surface absent at
  LFPG / LIRF / EGLL / LTFM; 200 blind rows read (LIRF 100, EDDF 50, LFPG 50): early / late recording classes, no trigger; supplied
  turnaround bias negligible. **Closed, no GO** (`case_files5.md`). Nothing declared, no dry run; arcw6-7 never started.
