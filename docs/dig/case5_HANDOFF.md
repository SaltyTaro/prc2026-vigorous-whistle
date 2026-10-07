# Case files round four (2026-like blind spots), fork L: hand-off log

Lever #10 of the rank-1 hunt (7 October 2026). Scratch `runs/dig/case5/` (git-ignored). Control: v68 development proxy
`runs/opdi_stack_lszh/dev_v68_proxy.parquet` (938a91cd...; 320.0400 / 250.8650 / 181.9933 / 163.5496 s). Nothing declared yet.
Compute rule: heavy jobs on Azure arcw6-arcw7 only; the workstation reads small tables.

- 02:20 UTC: started. Plan: (1) blind-row census on the v68 proxy (rows without a surface trace, the 2026 condition at LIRF July and
  most of EDDF); (2) read the worst unexplained blind rows with every non-trace source (supplied arrivals, NM clocks, schedule, OPDI events,
  airborne points); (3) screen any mechanism month-blocked against the v68 proxy.
