"""adsb_rnn: the label-free ranking table of the residual network (built once on the workstation, then hash-bound by every refit).

The ranking departures' residual-stage columns (tools/export_adsb_v11.ranking_table with neighbour features, the path of v25's residual
members; matches_v11, stands) on v25's recorded ranking v9 (runs/adsb_nn2/export_v25/ranking_values.parquet). The scope flag
(matched_no_lirf) must equal v25's recorded residual scope row for row. ADS-B neighbour columns are not bit-reproducible across rebuilds
(an unstable sort moves a few dozen nb_runway_prev/next values), which is why every refit and the verifier read this one saved table.
Output: runs/adsb_rnn/ranking_nb.parquet (ID, in_scope, the nb columns) and a JSON receipt next to it. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS  # noqa: E402
from export_adsb_v11 import ranking_table  # noqa: E402
from run_lirf_mixture import MOVEMENT_COLUMNS  # noqa: E402
from run_adsb_v11 import scope_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-config", type=Path, default=Path("configs/adsb_rnn_network_v1.json"))
    parser.add_argument("--v25-values", type=Path, default=Path("runs/adsb_nn2/export_v25/ranking_values.parquet"))
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_rnn/ranking_nb.parquet"))
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise ValueError(f"Preserve the existing ranking table: {out}")
    net_config = read_json(ROOT / args.network_config)
    stage = net_config["residual_stage"]
    matches_root = ROOT / stage["matches_dirs"]["v11"]
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))  # the columns export_adsb_v25 read
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    values = pd.read_parquet(ROOT / args.v25_values).set_index(ID).reindex(departures[ID])
    if values["v9"].isna().any():
        raise ValueError("v25's ranking values do not cover every ranking departure.")
    config = {"matches_dir": stage["matches_dirs"]["v11"], "residual_model": {"neighbour_features": True}, "extra_feature_tables": {}}
    table, _ = ranking_table(config, departures, pd.read_parquet(matches_root / "stands.parquet"), values["v9"].to_numpy(dtype=float))
    if not np.array_equal(table[ID].to_numpy(), departures[ID].to_numpy()):
        raise ValueError("Ranking table order differs from the departures.")
    inside = scope_mask(table, stage["scope"])
    if not np.array_equal(inside, values["scope"].eq(stage["scope"]).to_numpy()):
        raise ValueError("The residual scope differs from v25's recorded scope.")
    frame = table[[ID, *stage["nb_features"]]].copy()
    frame.insert(1, "in_scope", inside)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    receipt = {"created_at_utc": utc_now(), "rows": int(len(frame)), "rows_in_scope": int(inside.sum()), "ids_sha256": identity_hash(frame[ID]),
               "table_sha256": sha256_file(out), "v25_values_sha256": sha256_file(ROOT / args.v25_values), "network_config_sha256": sha256_file(ROOT / args.network_config),
               "matches": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in ("2026-01", "2026-07")}, "stands": sha256_file(matches_root / "stands.parquet"),
               "code": {p: sha256_file(ROOT / p) for p in ["tools/rnn_ranking_table.py", "tools/export_adsb_v11.py", "tools/run_adsb_v11.py", "src/prc2026_adsb/features.py"]}}
    write_json(out.with_suffix(".json"), receipt)
    print(receipt)


if __name__ == "__main__":
    main()
