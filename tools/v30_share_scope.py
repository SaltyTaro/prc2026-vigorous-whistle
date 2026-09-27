"""Exploratory screen (27 September 2026): does v30's committee mix want a different network share on ADS-B-matched and unmatched
non-LIRF rows? v30 uses one share (0.75 of the network table 0.25 e24 + 0.25 e24b + 0.5 e32, against the tree committee) everywhere,
while on October unmatched rows the network table scores 207.3 s against the committee's 215.9 s.

Rebuilds v30 per row on the four recipe months at shares {0.5, 0.6, 0.75, 0.85, 1.0} (the v30_error_map development path: combined-round
mix + round-B LIRF blends; the share-0.75 rebuild must equal the recorded combined candidate on non-LIRF rows), then scores every
(share on matched, share on unmatched) pair: month RMSE ratios against v30 and the paired-day interval. Descriptive screen only —
anything it suggests needs its own declaration under the unchanged rule. Report: docs/share_scope/screen.json.
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_rnn_dev import combine, residual_columns  # noqa: E402
from run_adsb_v11 import candidate_values, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

SHARES = [0.5, 0.6, 0.75, 0.85, 1.0]
OUT = ROOT / "runs/share_scope/rows.parquet"


def build() -> pd.DataFrame:
    decl5, decl_b = read_json(ROOT / "configs/adsb_ctx_v5.json"), read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    config = merged_config({**decl5, "networks": {**decl5["networks"], "ens_ctx_lirf": decl_b["networks"]["ens_ctx_lirf"]}})
    v26, v25 = config["candidates"]["rnet_w90"], config["v25_recipe"]
    lirf_b = {**v25, "lirf": decl_b["candidates"]["lctx_b100_up"]["lirf"], "lirf_network": decl_b["candidates"]["lctx_b100_up"]["lirf_network"]}
    rec5 = {m: e["rmse"] for m, e in read_json(ROOT / "docs/adsb_ctx/development_results_v5.json")["candidates"]["c_e24x32_e32h"]["months"].items()}
    net = decl5["candidates"]["c_e24x32_e32h"]["network"]
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
        network = weighted(table, net)
        frame = pd.DataFrame({ID: ids.to_numpy(), "month": month, "label": y, "airport": table["airport"].to_numpy(),
                              "matched": table["adsb_matched"].eq(1).to_numpy(), "day": table[TIME].dt.strftime("%Y-%m-%d").to_numpy()})
        for s in SHARES:
            table[f"mix:{s}"] = mixed(committee, network, float(s))
            frame[f"s{s}"], _ = candidate_values(table, v9, residual, spec_for(config, f"mix:{s}", lirf_blends(config, table, lirf_b, "lctx_b100_up")), floor)
        check, _ = candidate_values(table, v9, residual, spec_for(config, "mix:0.75", lirf_blends(config, table, v25, "v26")), floor)
        if abs(rmse(y, check) - rec5[month]) > 1e-9:
            raise ValueError(f"Share 0.75 does not reproduce the combined candidate on {month}.")
        print(month, {s: round(rmse(y, frame[f"s{s}"]), 4) for s in SHARES}, flush=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def day_interval(days, y, a, b, rng, replicates=2000):
    uniq, inv = np.unique(days, return_inverse=True)
    sa, sb, n = np.bincount(inv, (a - y) ** 2), np.bincount(inv, (b - y) ** 2), np.bincount(inv).astype(float)
    draws = [np.sqrt(sa[p].sum() / n[p].sum()) - np.sqrt(sb[p].sum() / n[p].sum()) for p in (rng.integers(0, len(uniq), len(uniq)) for _ in range(replicates))]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(OUT) if OUT.exists() else build()
    rows.to_parquet(OUT, index=False)
    non_lirf = rows["airport"].ne("LIRF").to_numpy()
    rng = np.random.default_rng(20260927)
    report = {"created_at_utc": utc_now(), "shares": SHARES, "pairs": {}}
    for sm in SHARES:
        for su in SHARES:
            key = f"matched{sm}_unmatched{su}"
            entry = {"months": {}}
            for month, f in rows.groupby("month"):
                k = f["airport"].ne("LIRF").to_numpy() & f["matched"].to_numpy()
                values = np.where(f["airport"].ne("LIRF").to_numpy(), np.where(f["matched"].to_numpy(), f[f"s{sm}"], f[f"s{su}"]), f["s0.75"])
                y, base = f["label"].to_numpy(), f["s0.75"].to_numpy()
                entry["months"][month] = {"ratio_pct": 100 * (rmse(y, values) / rmse(y, base) - 1)}
                if (sm, su) != (0.75, 0.75):
                    entry["months"][month]["day_interval"] = day_interval(f["day"].to_numpy(), y, values, base, rng)
            entry["mean_ratio_pct"] = float(np.mean([m["ratio_pct"] for m in entry["months"].values()]))
            entry["all_months_better"] = all(m["ratio_pct"] <= 0 for m in entry["months"].values())
            entry["intervals_below_zero"] = int(sum(m.get("day_interval", [0, 0])[1] < 0 for m in entry["months"].values()))
            report["pairs"][key] = entry
    best = sorted(report["pairs"].items(), key=lambda kv: kv[1]["mean_ratio_pct"])[:8]
    report["best_by_mean"] = [k for k, _ in best]
    write_json(ROOT / "docs/share_scope/screen.json", report)
    for k, e in best:
        print(k, round(e["mean_ratio_pct"], 4), {m: round(v["ratio_pct"], 3) for m, v in e["months"].items()}, "intervals<0:", e["intervals_below_zero"])


if __name__ == "__main__":
    main()
