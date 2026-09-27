"""Descriptive error map of v30: where its squared error sits on the four recipe months, and where it is expected to sit on the ranking set.

Development: rebuilds v30's recipe per row on 2025-01/07/11/12 (the combined round's committee mix c_e24x32_e32h plus round B's LIRF
blends lctx_b100_up), after checking three reproductions to 1e-9 s: v28 (the combined round's control), c_e24x32_e32h with v25's LIRF
blends (its recorded development RMSE) and v27 + lctx_b100_up (round B's recorded development RMSE). Rows are split into label-free cells
(airport x ADS-B status gate/dark/unmatched x NM clock present x LIRF fill population).

Ranking: the same label-free cells for the 344,841 ranking departures (the exporter's ranking table). Each ranking row gets the mean
development squared error of its cell (season-matched month first, backing off to all four months, then coarser cells), which projects
the ranking RMSE the development error rates imply under the ranking months' own coverage. The gap between that projection and the
official score is error the development months do not show (ranking-only regimes, extreme-row luck). Descriptive only: nothing is
selected or exported, and no ranking label exists or is inferred. Report: docs/error_map/v30_error_map.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET, TIME  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS  # noqa: E402
from export_adsb_v11 import ranking_table  # noqa: E402
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_rnn_dev import combine, residual_columns  # noqa: E402
from run_adsb_v11 import candidate_values, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402
from run_lirf_mixture import MOVEMENT_COLUMNS  # noqa: E402

OUT = ROOT / "runs/error_map"
OFFICIAL_V30 = 229.6009
SEASON = {"2026-01": "2025-01", "2026-07": "2025-07"}
RECENT = {"2026-01": "2025-12", "2026-07": "2025-07"}


def cells(frame: pd.DataFrame) -> pd.DataFrame:
    matched = frame["adsb_matched"].eq(1)
    status = np.where(matched & frame["adsb_gate"].eq(1), "gate", np.where(matched, "dark", "unmatched"))
    return pd.DataFrame({"airport": frame["airport"].to_numpy(), "status": status, "clock": np.where(frame["proxy_s"].notna(), "clock", "noclock"),
                         "pop": frame["population"].fillna(False).astype(bool).to_numpy()})


def development_rows() -> pd.DataFrame:
    decl5, decl_b = read_json(ROOT / "configs/adsb_ctx_v5.json"), read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    combined = {**decl5, "networks": {**decl5["networks"], "ens_ctx_lirf": decl_b["networks"]["ens_ctx_lirf"]}}
    config = merged_config(combined)
    v26, v25 = config["candidates"]["rnet_w90"], config["v25_recipe"]
    lirf_b = {**v25, "lirf": decl_b["candidates"]["lctx_b100_up"]["lirf"], "lirf_network": decl_b["candidates"]["lctx_b100_up"]["lirf_network"]}
    rec5 = {m: e["rmse"] for m, e in read_json(ROOT / "docs/adsb_ctx/development_results_v5.json")["candidates"]["c_e24x32_e32h"]["months"].items()}
    rec_b = {m: e["rmse"] for m, e in read_json(ROOT / "docs/adsb_ctx_lirf/development_results.json")["candidates"]["lctx_b100_up"]["months"].items()}
    cand = decl5["candidates"]["c_e24x32_e32h"]
    frames = []
    for month in decl5["recipe_months"]:
        table = month_table(load_frame(config, month, "cross_fit"), adsb_features(month, ROOT / config["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), month)
        ids = table[ID]
        y, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
        residual = combine(v26, residual_columns(config, month, ids, smoke=False), "rnet_w90")
        floor = float(table["floor"].iloc[0])
        for name, values in extra_columns(config, month, ids.to_numpy()).items():
            table[name] = values
        committee = table[config["committee_table"]].to_numpy(dtype=float)
        table["mix:v28"] = mixed(committee, weighted(table, decl5["control"]["network_table"]), float(decl5["control"]["network_share"]))
        table["mix:v27"] = mixed(committee, weighted(table, decl_b["control"]["committee_network"]), float(decl_b["control"]["committee_share"]))
        table["mix:v30"] = mixed(committee, weighted(table, cand["network"]), float(cand["network_share"]))
        runs = {"v28": ("mix:v28", v25, "v26"), "v30_v25lirf": ("mix:v30", v25, "v26"), "v27_b": ("mix:v27", lirf_b, "lctx_b100_up"),
                "v30": ("mix:v30", lirf_b, "lctx_b100_up")}
        values = {k: candidate_values(table, v9, residual, spec_for(config, col, lirf_blends(config, table, blend, tag)), floor)[0] for k, (col, blend, tag) in runs.items()}
        checks = {"v28": decl5["control"]["recorded_development_rmse"][month], "v30_v25lirf": rec5[month], "v27_b": rec_b[month]}
        for k, recorded in checks.items():
            if abs(rmse(y, values[k]) - recorded) > 1e-9:
                raise ValueError(f"{k} does not reproduce its recorded development RMSE on {month}: {rmse(y, values[k])} vs {recorded}.")
        print(month, {k: round(rmse(y, v), 4) for k, v in values.items()}, flush=True)
        frame = pd.concat([pd.DataFrame({ID: ids.to_numpy(), "month": month, "label": y, "v30": values["v30"], "v28": values["v28"],
                                         "day": table[TIME].dt.strftime("%Y-%m-%d").to_numpy(), "hour": table[TIME].dt.hour.to_numpy()}),
                           cells(table)], axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ranking_rows() -> pd.DataFrame:
    detail = pd.read_parquet(ROOT / "runs/adsb_ctx/export_v30/ranking_values.parquet")
    config = merged_config(read_json(ROOT / "configs/adsb_ctx_v5.json"))
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), detail[ID].to_numpy()):
        raise ValueError("The v30 detail is not in ranking departure order.")
    runtime = {**config, "matches_dir": config["matches_dirs"]["v11"], "residual_model": {"neighbour_features": False},
               "extra_feature_tables": {k: v for k, v in config["extra_feature_tables"].items() if k not in config["ensembles"]}}
    stands = pd.read_parquet(ROOT / config["matches_dirs"]["v11"] / "stands.parquet")
    table, _ = ranking_table(runtime, departures, stands, detail["v9"].to_numpy(dtype=float))
    return pd.concat([pd.DataFrame({ID: detail[ID].to_numpy(), "month": detail["month"].to_numpy(), "v30": detail["v30"].to_numpy(dtype=float),
                                    "day": departures[TIME].dt.strftime("%Y-%m-%d").to_numpy(), "hour": departures[TIME].dt.hour.to_numpy()}),
                      cells(table)], axis=1)


def project(dev: pd.DataFrame, rank: pd.DataFrame, month_map: dict, minimum: int = 150) -> np.ndarray:
    """Expected squared error per ranking row: mean development squared error of its cell, with back-off to coarser cells."""
    dev = dev.assign(se=(dev["v30"] - dev["label"]) ** 2, src=dev["month"])
    rank = rank.assign(src=rank["month"].map(month_map))
    levels = [["src", "airport", "status", "clock", "pop"], ["airport", "status", "clock", "pop"], ["airport", "status", "pop"], ["airport", "pop"], ["airport"]]
    out = np.full(len(rank), np.nan)
    for keys in levels:
        stats = dev.groupby(keys)["se"].agg(["mean", "size"]).reset_index()
        stats = stats[stats["size"] >= minimum]
        merged = rank[keys].merge(stats, on=keys, how="left")
        fill = np.isnan(out) & merged["mean"].notna().to_numpy()
        out[fill] = merged["mean"].to_numpy()[fill]
    if np.isnan(out).any():
        raise ValueError("Some ranking cells have no development estimate.")
    return out


def group(frame: pd.DataFrame) -> pd.Series:
    lirf = frame["airport"].eq("LIRF")
    return pd.Series(np.where(lirf & frame["pop"], "LIRF:population", np.where(lirf, "LIRF:" + frame["status"], frame["airport"] + ":" + frame["status"])),
                     index=frame.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev_path, rank_path = OUT / "dev_rows.parquet", OUT / "ranking_rows.parquet"
    dev = pd.read_parquet(dev_path) if dev_path.exists() else development_rows()
    dev.to_parquet(dev_path, index=False)
    rank = pd.read_parquet(rank_path) if rank_path.exists() else ranking_rows()
    rank.to_parquet(rank_path, index=False)
    n_rank = len(rank)
    official_sse = OFFICIAL_V30 ** 2 * n_rank
    dev["se"] = (dev["v30"] - dev["label"]) ** 2
    dev["extreme"] = dev["label"] > 10800
    report = {"created_at_utc": utc_now(), "recipe": "v30 = c_e24x32_e32h committee mix + round B lctx_b100_up LIRF blends (development rebuild)",
              "official_v30_rmse": OFFICIAL_V30, "ranking_rows": n_rank, "development": {}, "projection": {}}
    total = float(dev["se"].sum())
    dev["group"] = group(dev)
    report["development"]["rmse_by_month"] = {m: float(np.sqrt(f["se"].mean())) for m, f in dev.groupby("month")}
    report["development"]["pooled_rmse"] = float(np.sqrt(dev["se"].mean()))
    report["development"]["extreme_rows"] = {"rows": int(dev["extreme"].sum()), "sse_share": float(dev.loc[dev["extreme"], "se"].sum() / total)}
    report["development"]["by_group"] = {g: {"rows": int(len(f)), "rmse": float(np.sqrt(f["se"].mean())), "sse_share": float(f["se"].sum() / total),
                                             "extreme_rows": int(f["extreme"].sum()), "sse_share_ordinary": float(f.loc[~f["extreme"], "se"].sum() / total)}
                                         for g, f in dev.groupby("group")}
    rank["group"] = group(rank)
    for name, mapping in [("season", SEASON), ("recent", RECENT), ("pooled", {"2026-01": None, "2026-07": None})]:
        for ordinary in (False, True):
            src = dev[~dev["extreme"]] if ordinary else dev
            if name == "pooled":
                src = src.assign(month="all")
                mapping_used = {"2026-01": "all", "2026-07": "all"}
            else:
                mapping_used = mapping
            expected = project(src, rank, mapping_used)
            key = f"{name}{'_ordinary_only' if ordinary else ''}"
            sse = float(expected.sum())
            entry = {"projected_rmse": float(np.sqrt(sse / n_rank)),
                     "projected_rmse_by_month": {m: float(np.sqrt(expected[rank["month"].eq(m).to_numpy()].mean())) for m in ["2026-01", "2026-07"]},
                     "official_minus_projected_sse_share_of_official": (official_sse - sse) / official_sse,
                     "by_group": {}}
            for g, idx in rank.groupby("group").groups.items():
                e = expected[rank.index.get_indexer(idx)]
                entry["by_group"][g] = {"rows": int(len(e)), "projected_rmse": float(np.sqrt(e.mean())), "projected_sse_share": float(e.sum() / sse)}
            report["projection"][key] = entry
            print(key, round(entry["projected_rmse"], 2), entry["projected_rmse_by_month"], "unexplained share", round(entry["official_minus_projected_sse_share_of_official"], 3), flush=True)
    report["ranking_composition"] = {m: rank[rank["month"].eq(m)].groupby("group").size().astype(int).to_dict() for m in ["2026-01", "2026-07"]}
    report["development_composition"] = {m: dev[dev["month"].eq(m)].groupby("group").size().astype(int).to_dict() for m in sorted(dev["month"].unique())}
    write_json(ROOT / "docs/error_map/v30_error_map.json", report)
    top = sorted(report["projection"]["season"]["by_group"].items(), key=lambda kv: -kv[1]["projected_sse_share"])[:14]
    print("season projection, largest groups:", [(g, round(v["projected_sse_share"] * 100, 1), round(v["projected_rmse"], 1), v["rows"]) for g, v in top])


if __name__ == "__main__":
    main()
