# One more member in the committee's network table: scorer and export path

Prepared on 29 September 2026 for the arrival second-task network, so that a GO from its
design screen can go straight to a declared development round and an export. Generic: any network member with the same table layout can
use it. Nothing here has been scored on a new member; no file was exported.

## What a declaration contains

```json
{
 "experiment_id": "...", "declared_at_utc": "...", "audit_used": false,
 "member": {"name": "...", "column": "adsb_full",
            "tables": {"2025-01": ["<per-seed table>", "..."], "2025-07": [], "2025-11": [], "2025-12": [], "ranking": []}},
 "candidates": {"w15": {"member_weight": 0.15}, "w25": {"member_weight": 0.25}},
 "selection": "runs/<round>/selection.json", "report": "docs/<round>/results_v1.json"
}
```

Tables: one parquet per seed with `MVT_ID_mvt` and `adsb_full` (seconds). Footing, equal to the control rows'
(`runs/lomo_dev/rows.parquet`):

| Month | Fit |
|---|---|
| January 2025, July 2025 | leave-one-month-out on the other eleven months of 2025 |
| November 2025 | forward fold (reported only; nine-month fold side) |
| December 2025 | January-November refit |
| ranking | twelve-month refit |

## Semantics

- `table' = (1 - w) x (0.25 e24 + 0.25 e24b + 0.5 e32) + w x member seed mean` where both are finite, the shipped table elsewhere. A row
  missing in any seed is missing. LIRF rows never take the member (they do not use the committee mix).
- The mix is v46's on `table'`: network share 0.85; share 0.5 where the committee sits more than 1,200 s above `table'`.
- Development: the whole v46 stack (`tools/run_share_gate_reread.full_stack`). Ranking: `tools/export_stack.layers`, as for v46.
- Rule (defaults; a declaration may state `equal_footing_months` and `minimum_mean_improvement_fraction`): eligible when no equal-footing
  month (January, July, December) is worse on all rows, the mean ratio is at least 0.02% below 1 and every equal-footing month's
  paired-day 95% interval lies below zero. Lowest mean ratio among the eligible; otherwise nothing.

## Commands

```
.venv\Scripts\python.exe -X utf8 -B tools\run_network_member.py --declaration <decl> --check-control
.venv\Scripts\python.exe -X utf8 -B tools\run_network_member.py --declaration <decl> --smoke
.venv\Scripts\python.exe -X utf8 -B tools\run_network_member.py --declaration <decl> --score --select
.venv\Scripts\python.exe -X utf8 -B tools\export_network_member.py --declaration <decl> --candidate <name> --dry-run <dir>
.venv\Scripts\python.exe -X utf8 -B tools\export_network_member.py --declaration <decl> --candidate <name> --version 48
```

The export's base is v46 (223.9339). v47 (`docs/published_means/`) is a separate file on the same base; the two changes stay separately
attributable. A shipped member also needs a layer in `src/prc2026_final/stack.py`; that is not written.

## Checks run on 29 September (code paths only)

| Check | Result |
|---|---|
| Control on the development rows | month RMSEs equal the v46 re-read to 1e-9 (322.85847 / 257.71846 / 184.90496 / 167.51368) |
| Member weight 0, and member equal to the table | no row changes in any month |
| Ranking: v38 stack | rounds to the uploaded v38 on every row |
| Ranking: v46 stack | reproduces the uploaded v46 on every row |
| Ranking dry run, member equal to the table | output byte-identical to the uploaded v46 (SHA-256 `8abdd87a...`) |
| Ranking dry run, member = the table's own e32 column, w 0.25 | 170,204 rows change, no LIRF row, none outside the member's rows |
| Tests | `tests/test_network_member.py`, 5 tests pass |

The smoke members are columns the table already holds; their scores were not read or recorded (`--smoke` prints pass / fail only).
