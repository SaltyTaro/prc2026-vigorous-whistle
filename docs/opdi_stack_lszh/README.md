# OPDI stacker layer at LSZH only (6-7 October 2026)

Declaration [`configs/opdi_stack_lszh_v1.json`](../../configs/opdi_stack_lszh_v1.json) (2045536), committed before the reading. A post-hoc
variant of lever 7 (`docs/opdi_stack/`): the full layer passed clauses 1-3 but failed its transfer guard at EDDF / LEBL / EDDM / LEMD;
LSZH's ratio was 1.05. The user: "run it here itself". No official score is used.

## Reading (development, against the pinned v66 proxy; [`results_v1.json`](results_v1.json), `tools/run_opdi_stack_lszh.py`)

| month | change | paired-day 95% (%) | rows changed |
|---|---:|---|---:|
| January 2025 | -0.0268% | [-0.087, -0.012] | 7,857 |
| July 2025 | -0.0829% | [-0.101, -0.068] | 11,989 |
| November 2025 | -0.1004% | [-0.145, -0.070] | 8,756 |
| December 2025 | -0.0999% | [-0.122, -0.082] | 10,227 |

Equal-footing mean -0.0698% (bar -0.02%), no month worse, every interval below zero, transfer guard at LSZH 1.05: **accepted by the
declared clauses** (the variant itself is a post-hoc choice, disclosed above). LSZH carries -0.070 of the full layer's -0.083%.

## Ranking dry run on v66 ([`dry_v68_report.json`](dry_v68_report.json))

`tools/run_opdi_stack_lszh.py rank`: lever 7's stack step restricted to LSZH, then v66's chain. The v3 part reproduces v66's shipped
stack step exactly; the result equals an independent row splice of lever 7's ranking file at LSZH. `runs/opdi_stack_lszh/rank/dry_v68/
vigorous-whistle_v68.parquet`, SHA-256 `d41d8705e75a71e865fe21a8d6ad2f62044a5ed07f125fd208067914b2b10eab`: against v66 17,338 LSZH rows
change (none elsewhere), squared change 6.75e6, RMS 4.4 s, largest +143 / -57 s; no LIRF population or guard row. Not built under
`runs/submissions`; a build and an upload only at the user's word. Ran on the workstation at the user's word ("here itself"; light).
