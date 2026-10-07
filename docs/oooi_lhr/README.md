# Declared round oooi_lhr_v1: Heathrow data-link OUT pin on dark / unmatched rows (7 October 2026)

Declaration: [`configs/oooi_lhr_v1.json`](../../configs/oooi_lhr_v1.json) (dbbdc06 + 9cb6a8d, declared 12:14:43 UTC). It
was run exactly as written by [`tools/run_oooi_lhr.py`](../../tools/run_oooi_lhr.py), and every number is in
[`results_v1.json`](results_v1.json). No VM was used: the filtered message tables from the extension screen
([`docs/oooi_ext/`](../oooi_ext/README.md), 84df504) were already local. Nothing was written under `runs/submissions`,
nothing was uploaded, and no leaderboard was read.

## Decision: not accepted; v70 stays

Clauses 1, 2, 3 and 5 pass. **Clause 4 fails**, and it is the binding clean check: October 2025 is +0.0005% (worse)
against rows12 `stack`. The candidate `lhr_w50` is therefore not selected, and no ranking dry run was written.

| clause | reading | result |
|---|---|---|
| 1. no development month worse by > 0.005% | Jan 0, Jul 0 (no source), Nov −0.0868%, Dec −0.0147% | pass |
| 2. mean of Nov and Dec ≤ −0.02% | −0.0507% | pass |
| 3. Nov and Dec paired-day intervals < 0 | Nov [−0.1541, −0.0407], Dec [−0.0348, −0.0001] | pass |
| 4. October 2025 change < 0 and ≤ 2% of pins off by > 1,800 s | **+0.0005%** [−0.0036, +0.0049] (interval reported only); 84 pins, 0% beyond 1,800 s | **fail** |
| 5. ranking / development change RMS within 0.5–2× | 65.6 s / 77.3 s = 0.849 | pass |

## Details

- **Control.** v70's development proxy is the v69 proxy (c088e409…) with oooi_pin_v1 w50 applied through
  `tools/run_oooi_pin.py`'s own functions. It reproduces `docs/oooi_pin/results_v1.json`'s w50 complete-month RMSEs
  exactly (320.0147 / 250.7120 / 181.6080 / 162.7895). It is saved as `runs/oooi_lhr/dev_v70_proxy.parquet`
  (SHA-256 51aaa6a6…). Before rebuilding, the run checks that `joined_dev.parquet` is unchanged since oooi_pin_v1.
- **Scope.** EGLL departures with a guarded EGLL_family pin: the screen's formats table `formats.parquet` (822b7cf5…),
  family BA_MVT16 / UA_OUT13 / MVA_AD / QP_VS in that priority. The ADS-B class must be dark or unmatched, read from
  the `matches2` tables as `tools/hunt_census.py` does. Fixed rows keep the chain. Value = max(0.5 × pin + 0.5 × chain, −12).
- **Development rows.**
  - November: 632 in scope (dark 528 + unm 104; gate 229 left out). By format: BA_MVT16 308, MVA_AD 159, QP_VS 100,
    UA_OUT13 65.
  - December: 157 in scope (dark 143 + unm 14; gate 105 left out).
  - Offsets (November fitted on December / December on November): BA_MVT16 +145 / +137 s, UA_OUT13 +87 / +92 s,
    MVA_AD +27 / +27 s, QP_VS +23 / −3 s. No format needed the pooled fallback.
  - Gains: November 9.29e6 s², December 1.29e6 s².
- **October 2025 (clean).**
  - Offsets from November + December: BA_MVT16 +139, UA_OUT13 +92, MVA_AD +27, QP_VS +1.4 s.
  - 84 pinned rows (QP_VS 37, MVA_AD 27, UA_OUT13 20; dark 81, unm 3). BA_MVT16, which carried most of November's
    gain, does not appear before November, so October tests only the three smaller formats.
  - Squared error rises by 9.5e4 s² against `stack`.
  - October labels were read from `runs/daily_totals/rows12.parquet` for this scoring step only.
    `runs/oooi_ext/scan/formats.parquet` is unchanged; its October block times and labels are still blank.
- **Ranking (computed for clause 5 only).** 551 July 2026 EGLL rows in scope, dark or unmatched. No file was written.

After the first run, one change was made: a JSON serialisation fix (`bool()` around one numpy boolean). The second run
reproduced every printed number exactly; `results_v1.json` comes from the second run.
