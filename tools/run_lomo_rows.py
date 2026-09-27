"""LOMO development rows (configs/lomo_dev_v1.json + configs/lomo_dev_networks_v1.json): January and July 2025 with every component on
eleven-month leave-one-month-out footing, in the layout of runs/fair_dev/rows.parquet so that tools/run_fair_weights.build and the rounds'
runners work unchanged.

Per held month (2025-01, 2025-07), starting from the fair rows (labels, flags, the LIRF fill population's values):
  v9          runs/lomo_dev/v9/<m>.parquet (the helper's LOMO v7 -> v9 on clock-present rows; fold v9 on missing-clock / fill-population rows)
  residual    v26's rnet_w90 combination of runs/lomo_dev/residual/<m>/predictions.parquet (adj2, cat, rnet re-predicted with LOMO v9)
  committee   runs/lomo_dev/ens/ens_v20_bag_q_all/lomo/predictions.parquet;  lirf_tree  runs/lomo_dev/ens/ens_v22_lirf_q/lomo/...
  networks    seed means of runs/lomo_dev/nets/<variant>_s<seed>/lomo_<m>/predictions.parquet: ens_ctx_e24 = 117-132, ens_ctx_e24b = 133-148,
              ens_ctx_e32 = 401-416, ens_ctx_lirf = 301-316; mix = the committee mix at share 0.75 of 0.25 e24 + 0.25 e24b + 0.5 e32
  v30 / v33   rebuilt with tools/run_fair_weights.build at v30's and v33's weights
November and December rows are copied from the fair set (provenance column says which). Before any LOMO input is read, the same assembly on
the FOLD inputs (tools/run_adsb_v24.extra_columns, the fair v9 and residual) must reproduce the fair rows' mix, lirf_tree, lirf_net, base and
v33 to 1e-9 s per row.

  --check-fold       the reproduction only
  (default)          -> runs/lomo_dev/rows.parquet + docs/lomo_dev/rows_summary.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import weighted  # noqa: E402
from run_adsb_rnn_dev import combine  # noqa: E402
from run_adsb_v11 import rmse  # noqa: E402
from run_adsb_v24 import extra_columns  # noqa: E402
from run_fair_weights import build  # noqa: E402

FAIR = "runs/fair_dev/rows.parquet"
OUT = ROOT / "runs/lomo_dev/rows.parquet"
HELD = ["2025-01", "2025-07"]
FAMILIES = {"ens_ctx_e24": ("ctx_e24", range(117, 133)), "ens_ctx_e24b": ("ctx_e24", range(133, 149)),
            "ens_ctx_e32": ("ctx_e32", range(401, 417)), "ens_ctx_lirf": ("ctx_lirf", range(301, 317))}
V30 = ({"ltfm": 0.7, "matched_other": 0.7, "unmatched_other": 0.7},
       {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.40, "net": 0.35, "tree": 0.25}})


def configs():
    decl5, decl_b = read_json(ROOT / "configs/adsb_ctx_v5.json"), read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    config = merged_config({**decl5, "networks": {**decl5["networks"], "ens_ctx_lirf": decl_b["networks"]["ens_ctx_lirf"]}})
    net = decl5["candidates"]["c_e24x32_e32h"]
    lirf_net = decl_b["candidates"]["lctx_b100_up"]["lirf_network"]
    v33 = read_json(ROOT / "configs/fair_weights_v1.json")["control"]
    return config, net, lirf_net, v33


def assemble(f: pd.DataFrame, v9, residual, tables: dict, config, net, lirf_net, v33) -> pd.DataFrame:
    """The fair-row layout from components; LIRF fill-population rows keep the input frame's v30 / v33."""
    g = f.copy()
    matched_non_lirf = g["matched"].to_numpy() & (g["airport"].to_numpy() != "LIRF")
    g["v9"], g["residual"] = v9, np.where(matched_non_lirf, residual, 0.0)
    g["base"] = g["v9"] + g["residual"]
    t = pd.DataFrame(tables)
    g["mix"] = mixed(t[config["committee_table"]].to_numpy(dtype=float), weighted(t, net["network"]), float(net["network_share"]))
    g["lirf_tree"], g["lirf_net"] = t[config["lirf_table"]].to_numpy(dtype=float), weighted(t, lirf_net)
    g["v30"] = build(g, *V30)
    g["v33"] = build(g, v33["outer"], v33["lirf"])
    return g


def check_fold(fair: pd.DataFrame, config, net, lirf_net, v33) -> dict:
    out = {}
    for month in HELD:
        f = fair[fair["month"] == month].reset_index(drop=True)
        ids = f[ID].to_numpy()
        tables = extra_columns(config, month, ids)
        g = assemble(f, f["v9"].to_numpy(), f["residual"].to_numpy(), tables, config, net, lirf_net, v33)
        out[month] = {c: float(np.nanmax(np.abs(g[c].to_numpy(dtype=float) - f[c].to_numpy(dtype=float)))) for c in ["base", "mix", "lirf_tree", "lirf_net", "v30", "v33"]}
        if not all(v <= 1e-9 for v in out[month].values()):
            raise ValueError(f"The fold assembly does not reproduce the fair rows on {month}: {out[month]}")
    return out


def lomo_tables(month: str, ids: np.ndarray) -> tuple[dict, dict]:
    tables, sources = {}, {}
    for name, rel in [("ens_v20_bag_q_all", "runs/lomo_dev/ens/ens_v20_bag_q_all/lomo/predictions.parquet"),
                      ("ens_v22_lirf_q", "runs/lomo_dev/ens/ens_v22_lirf_q/lomo/predictions.parquet")]:
        tables[name] = pd.read_parquet(ROOT / rel).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        sources[rel] = sha256_file(ROOT / rel)
    for name, (variant, seeds) in FAMILIES.items():
        values = []
        for seed in seeds:
            rel = f"runs/lomo_dev/nets/{variant}_s{seed}/lomo_{month}/predictions.parquet"
            values.append(pd.read_parquet(ROOT / rel).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float))
            sources[rel] = sha256_file(ROOT / rel)
        tables[name] = np.mean(values, axis=0)
    return tables, sources


def main() -> None:
    config, net, lirf_net, v33 = configs()
    fair = pd.read_parquet(ROOT / FAIR)
    summary = {"created_at_utc": utc_now(), "fair_rows_sha256": sha256_file(ROOT / FAIR), "fold_reproduction_max_abs": check_fold(fair, config, net, lirf_net, v33)}
    print("fold reproduction", summary["fold_reproduction_max_abs"], flush=True)
    if "--check-fold" in sys.argv:
        return
    v26 = config["candidates"]["rnet_w90"]
    frames, summary["months"], summary["sources"] = [], {}, {}
    for month in HELD:
        f = fair[fair["month"] == month].reset_index(drop=True)
        ids = f[ID].to_numpy()
        v9 = pd.read_parquet(ROOT / f"runs/lomo_dev/v9/{month}.parquet", columns=[ID, "v9"]).set_index(ID)["v9"].reindex(ids).to_numpy(dtype=float)
        res = pd.read_parquet(ROOT / f"runs/lomo_dev/residual/{month}/predictions.parquet").set_index(ID).reindex(ids)
        columns = {m: res[f"residual:{m}"].to_numpy(dtype=float) for m in ["adj2", "cat", "rnet"]}
        residual = combine(v26, columns, "rnet_w90")
        tables, sources = lomo_tables(month, ids)
        summary["sources"].update(sources)
        summary["sources"][f"runs/lomo_dev/v9/{month}.parquet"] = sha256_file(ROOT / f"runs/lomo_dev/v9/{month}.parquet")
        summary["sources"][f"runs/lomo_dev/residual/{month}/predictions.parquet"] = sha256_file(ROOT / f"runs/lomo_dev/residual/{month}/predictions.parquet")
        g = assemble(f, v9, residual, tables, config, net, lirf_net, v33)
        for c in ["v9", "base", "mix", "lirf_tree", "lirf_net", "v30", "v33"]:
            if not np.isfinite(g[c].to_numpy(dtype=float)).all():
                raise ValueError(f"LOMO {month}: {c} has non-finite values.")
        y = g["label"].to_numpy(dtype=float)
        inside = g["matched"].to_numpy() & (g["airport"].to_numpy() != "LIRF")
        summary["months"][month] = {"rows": int(len(g)), "v30_fair": rmse(y, f["v30"].to_numpy()), "v30_lomo": rmse(y, g["v30"].to_numpy()),
                                    "v33_fair": rmse(y, f["v33"].to_numpy()), "v33_lomo": rmse(y, g["v33"].to_numpy()),
                                    "matched_scope_base_fair": rmse(y[inside], f["base"].to_numpy()[inside]), "matched_scope_base_lomo": rmse(y[inside], g["base"].to_numpy()[inside]),
                                    "matched_scope_mix_fair": rmse(y[inside], f["mix"].to_numpy()[inside]), "matched_scope_mix_lomo": rmse(y[inside], g["mix"].to_numpy()[inside])}
        frames.append(g.assign(provenance="lomo"))
        print(month, {k: round(v, 3) for k, v in summary["months"][month].items()}, flush=True)
    rest = fair[~fair["month"].isin(HELD)].assign(provenance="fair")
    rows = pd.concat([*frames, rest], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUT, index=False)
    summary["rows_sha256"] = sha256_file(OUT)
    summary["code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_lomo_rows.py", "tools/run_lomo_residual.py", "tools/run_fair_weights.py"]}
    write_json(ROOT / "docs/lomo_dev/rows_summary.json", summary)
    print("wrote", OUT.relative_to(ROOT).as_posix(), summary["rows_sha256"])


if __name__ == "__main__":
    main()
