"""Development table of the shipped model part, layer by layer (v46 -> v51m -> v52m -> v53m -> v54m -> v55m), on runs/lomo_dev/rows.parquet.

Months and footing are the rows table's: January and July 2025 eleven-month leave-one-month-out, November the forward fold, December the
January-November refit. No model is fitted and nothing is selected; every layer is the shipped arithmetic, checked against its round's
recorded readings before the table is written:

  v46   tools/run_share_gate_reread.full_stack with v46's committee mix            == docs/thinned_regime/reread_n85_nd1200.json
  v51m  + the combination member at w 0.40 in the network table (non-LIRF rows)   == docs/adsb_ctx3_combo/results_v1.json w40
  v52m  + the masked member at w 0.40 after it                                     == docs/adsb_ctx3_mask/results_v1.json w40
  v53m  + the LIRF masked member (w 0.55) on lirf_u rows                           route RMSE == docs/adsb_ctx3_lirf_mask/results_v1.json w55
  v54m  + the dark re-read (w 0.70, share 0.75) on lirf_u rows of dark months      route RMSE == docs/lirf_dark/results_v1.json w70_s750
        (LIRF non-population match share below 5%: January 2025 only)
  v55m  + the matched-row member (w 0.40) on lirf_m rows                           route RMSE == docs/lirf_combo/results_v1.json mw40
LIRF layers are added as deltas (candidate minus control arithmetic on the route), floored at -12 s, as the exports do.
Output: runs/dev_model_part/rows.parquet and runs/dev_model_part/build.json.
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
from prc2026_final import lirf_gate as lg, network_member, share_gate  # noqa: E402
import run_lirf_combo as rlc  # noqa: E402
import run_lirf_dark as rld  # noqa: E402
import run_lirf_member as rlm  # noqa: E402
import run_network_member as rnm  # noqa: E402

OUT = ROOT / "runs/dev_model_part"
COMBO, MASK = "configs/adsb_ctx3_combo_dev_v1.json", "configs/adsb_ctx3_mask_dev_v1.json"
LIRF_MASK = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
FLOOR, CAP, TOL = -12.0, 20000.0, 1e-6


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def lirf_member(month: str, ids: np.ndarray) -> np.ndarray:
    fits = read_json(ROOT / LIRF_MASK)["fits"]
    root = ROOT / fits["outputs"].split("<variant>")[0]
    fold, file = rlm.NATURAL[month]
    return np.mean([rlm.read(root / f"{fits['variants'][0]}_s{s}" / fold / file, ids) for s in fits["seeds"]], axis=0)


def main() -> None:
    rows = pd.read_parquet(ROOT / rnm.ROWS)
    cs = rnm.scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = rnm.offsets(), rnm.offsets_all()
    seg = rnm.segments(rnm.MONTHS)
    k = read_json(ROOT / rnm.STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    reread = read_json(ROOT / rnm.REREAD)["months"]
    combo, mask = read_json(ROOT / COMBO), read_json(ROOT / MASK)
    rec = {"v51m": read_json(ROOT / "docs/adsb_ctx3_combo/results_v1.json")["candidates"]["w40"]["months"],
           "v52m": read_json(ROOT / "docs/adsb_ctx3_mask/results_v1.json")["candidates"]["w40"]["months"],
           "v53m": read_json(ROOT / "docs/adsb_ctx3_lirf_mask/results_v1.json")["results"]["w55"],
           "v54m": read_json(ROOT / "docs/lirf_dark/results_v1.json")["results"]["w70_s750"],
           "v55m": read_json(ROOT / "docs/lirf_combo/results_v1.json")["results"]["mw40"]}
    frames, checks = [], {}
    for month in rnm.MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        ids, y = f[ID].to_numpy(), f["label"].to_numpy(dtype=float)
        lirf = f["airport"].to_numpy() == "LIRF"
        committee, nets = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)
        c = {}

        def stack(table: np.ndarray) -> np.ndarray:
            mix, _ = share_gate.mix(committee, table, network_member.V46_SHARE, network_member.V46_GATE)
            return rnm.full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)

        v46 = stack(nets)
        c["v46"] = (rmse(y, v46), reread[month]["candidate_rmse"])
        sources: dict = {}
        t51 = rnm.with_bases(nets, lirf, ids, month, [{"member": combo["member"], "weight": network_member.candidate_weight(combo, "w40")}], sources)
        v51 = stack(t51)
        c["v51m"] = (rmse(y, v51), rec["v51m"][month]["rmse"])
        m52, _ = rnm.member_values(mask["member"], month, ids)
        t52, _ = network_member.table_with_member(t51, np.where(lirf, np.nan, m52), network_member.candidate_weight(mask, "w40"))
        v52 = stack(t52)
        c["v52m"] = (rmse(y, v52), rec["v52m"][month]["rmse"])
        for name, (got, want) in c.items():
            if abs(got - want) > TOL:
                raise ValueError(f"{month} {name}: {got} against the recorded {want}.")
        v9, net, tree = (f[x].to_numpy(dtype=float) for x in ("v9", "lirf_net", "lirf_tree"))
        u, m = f["lirf_u"].to_numpy(dtype=bool), f["lirf_m"].to_numpy(dtype=bool)
        M = lirf_member(month, ids)
        key = f"natural:{month}"
        # v53m: LIRF masked member on unmatched non-population rows
        ctrl_u, _ = lg.gated(v9[u], net[u], tree[u], "unmatched", rlm.GATE)
        cand_u, _ = lg.gated(v9[u], 0.45 * net[u] + 0.55 * M[u], tree[u], "unmatched", rlm.GATE)
        keep = y[u] < CAP
        c["v53m_route"] = (rmse(y[u][keep], cand_u[keep]), rec["v53m"][key]["primary"]["rmse"])
        c["v46_lirf_u_vs_gated_control_max_abs_s"] = float(np.max(np.abs(v52[u] - ctrl_u))) if u.any() else 0.0
        v53 = v52.copy()
        v53[u] = np.maximum(v52[u] + (cand_u - ctrl_u), FLOOR)
        # v54m: the dark re-read in dark months (match share of LIRF non-population rows below 5%)
        share = m.sum() / max(m.sum() + u.sum(), 1)
        v54 = v53.copy()
        if share < 0.05:
            s = {"v9": v9[u], "net": net[u], "tree": tree[u], "member": M[u]}
            dark, _ = rld.values(s, 0.70, 0.75, 600.0)
            base, _ = rld.values(s, rld.CONTROL["member_weight"], rld.CONTROL["network_share"], 600.0)
            c["v54m_route"] = (rmse(y[u][keep], dark[keep]), rec["v54m"][key]["primary"]["rmse"])
            v54[u] = np.maximum(v53[u] + (dark - base), FLOOR)
        # v55m: matched-row member on matched non-population rows
        v55 = v54.copy()
        if m.any():
            ctrl_m, _ = rlc.values(v9[m], net[m], tree[m], None, 0.0)
            cand_m, _ = rlc.values(v9[m], net[m], tree[m], M[m], 0.40)
            keep_m = y[m] < CAP
            if key in rec["v55m"]:
                c["v55m_route"] = (rmse(y[m][keep_m], cand_m[keep_m]), rec["v55m"][key]["primary"]["rmse"])
            v55[m] = np.maximum(v54[m] + (cand_m - ctrl_m), FLOOR)
        for name, val in c.items():
            if isinstance(val, tuple) and abs(val[0] - val[1]) > TOL:
                raise ValueError(f"{month} {name}: {val[0]} against the recorded {val[1]}.")
        checks[month] = {**{n: (list(v) if isinstance(v, tuple) else v) for n, v in c.items()}, "lirf_match_share": float(share), "dark": bool(share < 0.05),
                         "rmse": {n: rmse(y, v) for n, v in (("v46", v46), ("v51m", v51), ("v52m", v52), ("v53m", v53), ("v54m", v54), ("v55m", v55))}}
        frames.append(pd.DataFrame({ID: ids, "month": month, "day": f["day"].to_numpy(), "airport": f["airport"].to_numpy(), "label": y,
                                    "matched": f["matched"].to_numpy(dtype=bool), "lirf_m": m, "lirf_u": u,
                                    "v46": v46, "v51m": v51, "v52m": v52, "v53m": v53, "v54m": v54, "v55m": v55}))
        print(month, {n: round(v, 4) for n, v in checks[month]["rmse"].items()}, "dark" if share < 0.05 else "", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.concat(frames, ignore_index=True)
    path = OUT / "rows.parquet"
    table.to_parquet(path, index=False)
    write_json(OUT / "build.json", {"created_at_utc": utc_now(), "rows": int(len(table)), "output": {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)},
                                     "inputs": {rnm.ROWS: sha256_file(ROOT / rnm.ROWS)}, "code": {"tools/build_dev_model_part.py": sha256_file(ROOT / "tools/build_dev_model_part.py")},
                                     "checks": checks})
    print("->", path)


if __name__ == "__main__":
    main()
