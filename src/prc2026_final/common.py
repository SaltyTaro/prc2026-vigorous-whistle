"""Run root layout, resumable stage receipts and the input movement file."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import environment, read_json, sha256_file, utc_now, write_json
from prc2026.features import BLOCK, ID, PHASE, TARGET, TIME, ranking_view
from prc2026.validation import identity_hash

ROOT = Path(__file__).resolve().parents[2]
TRAINING_MONTHS = [f"2025-{m:02}" for m in range(1, 12)]  # the frozen context/fitting convention of every ranking feature (December never entered it)
RAW_COLUMNS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'FLIGHT_RULE_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt',
               'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt', 'RUNWAY_mvt', 'STAND_mvt', 'TAXITIME_SEC_mvt', 'LOBT_flt',
               'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt', 'ADES_FILED_flt', 'MARKET_SEGMENT_flt', 'IOBT_flt', 'FLIGHT_RULE_flt', 'FLIGHT_TYPE_flt',
               'AIRCRAFT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt', 'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt']


def source_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


class Run:
    """A run root: <root>/input.json (the file's identity), <root>/stages/<name>.json (one receipt per completed stage)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        out = self.root.joinpath(*parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def receipt_path(self, stage: str) -> Path:
        return self.root / "stages" / f"{stage}.json"

    def done(self, stage: str) -> bool:
        return self.receipt_path(stage).exists()

    def receipt(self, stage: str) -> dict:
        return read_json(self.receipt_path(stage))

    def finish(self, stage: str, record: dict, outputs: list[Path], started: float) -> dict:
        """Write the stage receipt last, so an interrupted stage is simply rerun."""
        record = {"stage": stage, "created_at_utc": utc_now(), "seconds": round(time.time() - started, 1), **record,
                  "outputs": {p.relative_to(self.root).as_posix(): sha256_file(p) for p in outputs}}
        write_json(self.receipt_path(stage), record)
        return record

    def verify_outputs(self, stage: str) -> None:
        for relative, digest in self.receipt(stage)["outputs"].items():
            if sha256_file(self.root / relative) != digest:
                raise ValueError(f"{stage}: {relative} changed after its receipt was written; delete the receipt to rebuild it.")

    def input(self) -> dict:
        return read_json(self.root / "input.json")


def load_movements(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """The movement file with every departure's label and airport block time masked (the frozen ranking_view), whatever the file holds."""
    frame = pd.read_parquet(path)
    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"The movement file lacks the ranking schema's columns: {missing}")
    frame = ranking_view(frame[RAW_COLUMNS])
    return frame if columns is None else frame[columns]


def departures_of(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame[PHASE].eq("DEP")].reset_index(drop=True)


def file_months(frame: pd.DataFrame) -> list[str]:
    return sorted(frame.loc[frame[PHASE].eq("DEP"), TIME].dt.strftime("%Y-%m").unique().tolist())


def inspect(raw_path: Path, template_path: Path) -> dict:
    """S0: the file's schema, months, phases and its departures against the template (IDs, order); no label is read."""
    raw = pd.read_parquet(raw_path)
    missing = sorted(set(RAW_COLUMNS) - set(raw.columns))
    extra = sorted(set(raw.columns) - set(RAW_COLUMNS))
    if missing:
        raise ValueError(f"The movement file lacks the ranking schema's columns: {missing}")
    if raw[ID].duplicated().any() or raw[ID].isna().any():
        raise ValueError("Null or duplicate movement IDs.")
    if raw[TIME].dt.tz is None:
        raise ValueError("Movement times must be UTC-aware.")
    dep = raw[PHASE].eq("DEP")
    arrivals = raw.loc[~dep]
    template = pd.read_parquet(template_path)
    template_ids = template[ID].to_numpy()
    dep_ids = raw.loc[dep, ID].to_numpy()
    if len(template_ids) != len(dep_ids) or set(template_ids) != set(dep_ids):
        raise ValueError("The template does not list exactly the file's departures.")
    months = sorted(raw.loc[dep, TIME].dt.strftime("%Y-%m").unique().tolist())
    return {"raw": raw_path.as_posix(), "raw_sha256": sha256_file(raw_path), "template": template_path.as_posix(), "template_sha256": sha256_file(template_path),
            "movements": int(len(raw)), "departures": int(dep.sum()), "arrivals": int((~dep).sum()),
            "arrivals_with_in_block": int(arrivals[BLOCK].notna().sum()), "arrivals_with_taxi_in": int(arrivals[TARGET].notna().sum()),
            "departures_with_label_or_block_in_file": int(raw.loc[dep, [TARGET, BLOCK]].notna().any(axis=1).sum()),
            "months": months, "departure_months": {m: int(n) for m, n in raw.loc[dep, TIME].dt.strftime("%Y-%m").value_counts().sort_index().items()},
            "airports": sorted(raw.loc[dep, "ADEP_mvt"].unique().tolist()), "extra_columns_ignored": extra,
            "departure_ids_sha256": identity_hash(dep_ids), "departures_in_template_order": bool(np.array_equal(dep_ids, template_ids)),
            "first_departure_utc": raw.loc[dep, TIME].min().isoformat(), "last_departure_utc": raw.loc[dep, TIME].max().isoformat(),
            "note": "departure labels and airport block times are masked before every stage (prc2026.features.ranking_view)",
            "environment": environment(ROOT)}


def save_frame(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return path


def aligned_ids(frame: pd.DataFrame, ids: np.ndarray, what: str) -> pd.DataFrame:
    out = frame.set_index(ID).reindex(ids).reset_index()
    if len(out) != len(ids) or out[ID].isna().any():
        raise ValueError(f"{what} does not cover every departure.")
    return out
