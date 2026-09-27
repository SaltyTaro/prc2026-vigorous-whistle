"""Build the LTFM IOBT sensitivity input (configs/ltfm_iobt_sensitivity_v1.json): a copy of data/raw/ranking.parquet with the 2025
early-IOBT convention restored on July-2026 THY LTFM departures whose IOBT equals the schedule. IOBT := SCHED - 600 s, and EOBT_1 / LOBT
:= SCHED - 600 s where they equal the original IOBT. Departure rows only; every other value is byte-identical (checked).

Writes runs/final_ltfm_iobt/input/final.parquet and modification.json. Label-free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, write_json  # noqa: E402

RAW = ROOT / "data/raw/ranking.parquet"
OUT = ROOT / "runs/final_ltfm_iobt/input"
EDITED = ["IOBT_flt", "EOBT_1_flt", "LOBT_flt"]
SHIFT = pd.Timedelta(seconds=600)


def selection(table: pd.DataFrame) -> np.ndarray:
    carrier = table["FLIGHT_mvt"].astype("string").str.upper().str.extract(r"^([A-Z]+)", expand=False).str[:3]
    return (table["PHASE_mvt"].eq("DEP") & table["ADEP_mvt"].eq("LTFM") & table["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq("2026-07")
            & carrier.eq("THY").fillna(False) & table["IOBT_flt"].eq(table["SCHED_TIME_UTC_mvt"]).fillna(False)).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    raw = pq.read_table(RAW)
    frame = raw.to_pandas()
    sel = selection(frame)
    new = frame[EDITED].copy()
    target = frame["SCHED_TIME_UTC_mvt"] - SHIFT
    counts = {}
    for col in EDITED:
        hit = sel & (frame[col].eq(frame["IOBT_flt"]).fillna(False).to_numpy() if col != "IOBT_flt" else True)
        new.loc[hit, col] = target[hit]
        counts[col] = int(hit.sum())
    columns = []
    for name in raw.schema.names:
        if name in EDITED:
            columns.append(pa.array(new[name], type=raw.schema.field(name).type))
        else:
            columns.append(raw.column(name))
    table = pa.Table.from_arrays(columns, schema=raw.schema)
    for name in raw.schema.names:
        if name not in EDITED and not table.column(name).equals(raw.column(name)):
            raise ValueError(f"{name} changed.")
    changed = np.zeros(len(frame), dtype=bool)
    for col in EDITED:
        a, b = frame[col], new[col]
        changed |= ~((a == b).fillna(False) | (a.isna() & b.isna())).to_numpy()
    if (changed & ~sel).any():
        raise ValueError("A row outside the selection changed.")
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUT / "final.parquet")
    gap_before = (frame.loc[sel, "SCHED_TIME_UTC_mvt"] - frame.loc[sel, "IOBT_flt"]).dt.total_seconds()
    gap_after = (frame.loc[sel, "SCHED_TIME_UTC_mvt"] - new.loc[sel, "IOBT_flt"]).dt.total_seconds()
    jul = (frame["PHASE_mvt"].eq("DEP") & frame["ADEP_mvt"].eq("LTFM") & frame["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq("2026-07")).to_numpy()
    report = {"declaration": "configs/ltfm_iobt_sensitivity_v1.json", "declaration_sha256": sha256_file(ROOT / "configs/ltfm_iobt_sensitivity_v1.json"),
              "rows_selected": int(sel.sum()), "rows_changed": int(changed.sum()), "ltfm_july_departures": int(jul.sum()),
              "column_edits": counts, "sched_minus_iobt_before": {str(k): int(v) for k, v in gap_before.value_counts().items()},
              "sched_minus_iobt_after": {str(k): int(v) for k, v in gap_after.value_counts().items()},
              "other_columns_identical": True, "sha256_input": sha256_file(OUT / "final.parquet"), "sha256_ranking": sha256_file(RAW)}
    write_json(OUT / "modification.json", report)
    print(report)


if __name__ == "__main__":
    main()
