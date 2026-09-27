"""Fair development rows (configs/fair_dev_v1.json): per recipe month, every component of v30's recipe on equal footing -- fold v9
(tools/run_adsb_v24.load_frame(..., "fold")) and the fold-provenance residual stage (tools/run_fold_residual.py: adj2, cat, 16 rnb seeds,
combined with v26's rnet_w90 weights) on the v9 side; the committee, context-network ensembles and LIRF tables' existing development
values on the fold side.

Before the fair rows are built, the same assembly with the SHIPPED provenance (cross-fit v9, the 12-month leave-one-month-out residual
stage) must reproduce v30's development rows (runs/error_map/dev_rows.parquet) to 1e-9 s per row, so the only difference between the two
tables is the v9 side's provenance.

Writes runs/fair_dev/rows.parquet: ID, month, day, airport, label, floor, matched, lirf scope flags, v9, residual, base (v9 + residual on
matched_no_lirf), mix (the committee mix at share 0.75), lirf_tree, lirf_net, v30 and v33 (outer weight LTFM 1.0 / other matched 0.6),
for the fair provenance, plus v30_shipped (the reproduced development value). Report: docs/fair_dev/rows_summary.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_rnn_dev import combine, residual_columns  # noqa: E402
import run_adsb_v24  # noqa: E402,F401  (registers the lirf_nonpop_unmatched scope)
from run_adsb_v11 import SCOPES, candidate_values, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

DECLARATION = "configs/fair_dev_v1.json"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
FOLD_RESIDUAL = ROOT / "runs/fold_residual"
SEEDS = list(range(501, 517))
OUT = ROOT / "runs/fair_dev/rows.parquet"
DEV_ROWS = "runs/error_map/dev_rows.parquet"


def fair_residual(month: str, ids: pd.Series) -> dict[str, np.ndarray]:
    trees = pd.read_parquet(FOLD_RESIDUAL / "trees" / f"predictions_{month}.parquet").set_index(ID).reindex(ids)
    out = {m: trees[f"residual:{m}"].to_numpy(dtype=float) for m in ["adj2", "cat"]}
    nets = [pd.read_parquet(FOLD_RESIDUAL / f"rnb_s{s}" / month / "predictions.parquet").set_index(ID)["residual"].reindex(ids).to_numpy(dtype=float) for s in SEEDS]
    out["rnet"] = np.mean(nets, axis=0)
    for name, values in out.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Fair residual column {name} does not cover {month}.")
    return out


def main() -> None:
    check_only = "--check-shipped" in sys.argv  # reproduce the shipped provenance only (before the fold residual exists)
    decl = read_json(ROOT / DECLARATION)
    decl5, decl_b = read_json(ROOT / "configs/adsb_ctx_v5.json"), read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    config = merged_config({**decl5, "networks": {**decl5["networks"], "ens_ctx_lirf": decl_b["networks"]["ens_ctx_lirf"]}})
    v26, v25 = config["candidates"]["rnet_w90"], config["v25_recipe"]
    lirf_b = {**v25, "lirf": decl_b["candidates"]["lctx_b100_up"]["lirf"], "lirf_network": decl_b["candidates"]["lctx_b100_up"]["lirf_network"]}
    net = decl5["candidates"]["c_e24x32_e32h"]
    dev_rows = pd.read_parquet(ROOT / DEV_ROWS, columns=[ID, "month", "v30"])
    matches_root, cache = ROOT / config["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"
    frames, summary = [], {"created_at_utc": utc_now(), "declaration_sha256": object_hash(decl), "months": {}}
    for month in MONTHS:
        month_out = {}
        built = {}
        for provenance, source in [("shipped", "cross_fit"), ("fair", "fold")][:1 if check_only else 2]:
            table = month_table(load_frame(config, month, source), adsb_features(month, matches_root, cache), month)
            ids = table[ID]
            label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
            columns = residual_columns(config, month, ids, smoke=False) if provenance == "shipped" else fair_residual(month, ids)
            residual = combine(v26, columns, "rnet_w90")
            floor = float(table["floor"].iloc[0])
            for name, values in extra_columns(config, month, ids.to_numpy()).items():
                table[name] = values
            table["mix:v30"] = mixed(table[config["committee_table"]].to_numpy(dtype=float), weighted(table, net["network"]), float(net["network_share"]))
            v30, _ = candidate_values(table, v9, residual, spec_for(config, "mix:v30", lirf_blends(config, table, lirf_b, "control")), floor)
            matched = table["adsb_matched"].eq(1).to_numpy()
            airport = table["airport"].to_numpy()
            base = v9 + np.where(np.asarray(SCOPES[config["residual_scope"]](table), dtype=bool), residual, 0.0)
            mix = table["mix:v30"].to_numpy(dtype=float)
            w = np.where(airport == "LTFM", 1.0, np.where(matched & (airport != "LIRF"), 0.6, 0.7))
            v33 = np.where(airport != "LIRF", np.maximum((1 - w) * base + w * mix, floor), v30)
            recon = np.where(airport != "LIRF", np.maximum(0.3 * base + 0.7 * mix, floor), v30)
            if float(np.max(np.abs(recon - v30))) > 1e-9:
                raise ValueError(f"{provenance} {month}: base/mix reconstruction does not equal the recipe.")
            built[provenance] = dict(table=table, label=label, v9=v9, residual=residual, base=base, mix=mix, v30=v30, v33=v33, floor=floor, ids=ids)
            month_out[provenance] = {"v30": rmse(label, v30), "v33": rmse(label, v33), "v33_vs_v30_pct": 100 * (rmse(label, v33) / rmse(label, v30) - 1),
                                     "base_scope_rmse": rmse(label[matched & (airport != "LIRF")], base[matched & (airport != "LIRF")]),
                                     "mix_scope_rmse": rmse(label[matched & (airport != "LIRF")], mix[matched & (airport != "LIRF")])}
        ship = built["shipped"]
        ref = dev_rows[dev_rows["month"] == month].set_index(ID).reindex(ship["ids"])["v30"].to_numpy(dtype=float)
        month_out["shipped_reproduction_max_abs"] = float(np.max(np.abs(ship["v30"] - ref)))
        if np.isnan(ref).any() or month_out["shipped_reproduction_max_abs"] > 1e-9:
            raise ValueError(f"The shipped provenance does not reproduce v30's development rows on {month}.")
        if check_only:  # shipped-provenance rows in the same layout (for testing the round tools and for comparison)
            t = ship["table"]
            frames.append(pd.DataFrame({ID: ship["ids"].to_numpy(), "month": month, "day": t[TIME].dt.strftime("%Y-%m-%d").to_numpy(),
                                        "airport": t["airport"].to_numpy(), "label": ship["label"], "floor": ship["floor"],
                                        "matched": t["adsb_matched"].eq(1).to_numpy(),
                                        "lirf_m": np.asarray(SCOPES["lirf_nonpop_matched"](t), dtype=bool), "lirf_u": np.asarray(SCOPES["lirf_nonpop_unmatched"](t), dtype=bool),
                                        "v9": ship["v9"], "residual": ship["residual"], "base": ship["base"], "mix": ship["mix"],
                                        "lirf_tree": t[config["lirf_table"]].to_numpy(dtype=float), "lirf_net": weighted(t, decl_b["candidates"]["lctx_b100_up"]["lirf_network"]),
                                        "v30": ship["v30"], "v33": ship["v33"]}))
            print(month, month_out, flush=True)
            continue
        fair = built["fair"]
        if not (fair["ids"].to_numpy() == ship["ids"].to_numpy()).all():
            raise ValueError(f"Row order differs between provenances on {month}.")
        t = fair["table"]
        frames.append(pd.DataFrame({ID: fair["ids"].to_numpy(), "month": month, "day": t[TIME].dt.strftime("%Y-%m-%d").to_numpy(),
                                    "airport": t["airport"].to_numpy(), "label": fair["label"], "floor": fair["floor"],
                                    "matched": t["adsb_matched"].eq(1).to_numpy(),
                                    "lirf_m": np.asarray(SCOPES["lirf_nonpop_matched"](t), dtype=bool), "lirf_u": np.asarray(SCOPES["lirf_nonpop_unmatched"](t), dtype=bool),
                                    "v9": fair["v9"], "residual": fair["residual"], "base": fair["base"], "mix": fair["mix"],
                                    "lirf_tree": t[config["lirf_table"]].to_numpy(dtype=float), "lirf_net": weighted(t, decl_b["candidates"]["lctx_b100_up"]["lirf_network"]),
                                    "v30": fair["v30"], "v33": fair["v33"], "v30_shipped": ship["v30"], "v33_shipped": ship["v33"]}))
        summary["months"][month] = month_out
        print(month, {p: {k: round(v, 4) for k, v in e.items()} for p, e in month_out.items() if isinstance(e, dict)}, flush=True)
    if check_only:
        shipped = ROOT / "runs/fair_dev/rows_shipped.parquet"
        shipped.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(shipped, index=False)
        print("wrote", shipped.relative_to(ROOT).as_posix(), sha256_file(shipped))
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
    summary["rows_sha256"] = sha256_file(OUT)
    summary["mean_v33_vs_v30_pct"] = {p: float(np.mean([summary["months"][m][p]["v33_vs_v30_pct"] for m in MONTHS])) for p in ["shipped", "fair"]}
    write_json(ROOT / "docs/fair_dev/rows_summary.json", summary)
    print(summary["mean_v33_vs_v30_pct"])


if __name__ == "__main__":
    main()
