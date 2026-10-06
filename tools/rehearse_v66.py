"""Rehearsal of the v66 chain (docs/final_file/v66_chain.md): rerun every step that built runs/submissions/vigorous-whistle_v66.parquet
into runs/final_file/rehearsal_v66/ and compare each intermediate with the shipped dry run and build.

The chain is tools/export_ctx_ownpoints.chain, imported unchanged, with the module globals set exactly as
tools/export_ownpoints_weight_override.export sets them (CHAIN = runs/ctx_ownpoints_weight/chain, DECLARATION = the weight round's
declaration) and the candidate own_all_w100. Nothing is written under runs/submissions; nothing is uploaded; no official score enters.

  python tools/rehearse_v66.py            -> runs/final_file/rehearsal_v66/{vigorous-whistle_v66.parquet, steps/, steps.json, rehearsal.json}
"""
from __future__ import annotations

import json
import platform
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

OUT = ROOT / "runs/final_file/rehearsal_v66"
CANDIDATE = "own_all_w100"
VERSION = 66
V66 = ("runs/submissions/vigorous-whistle_v66.parquet", "b4ea727e1644b6add104f44832898bfecfd3d5607f7d6a9ce0471efe581322e4")
SHIPPED_STEPS = {"dry run": "runs/ctx_ownpoints_weight/rank/dry_v66/steps.json", "build": "runs/ctx_ownpoints_weight/rank/build_v66/steps.json"}
INPUTS = [
    "data/raw/ranking.parquet", "data/raw/submitting.parquet",
    "runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet",
    "runs/ctx_ownpoints_weight/chain/own_all_w100/ctl.parquet", "runs/ctx_ownpoints_weight/chain/own_all_w100/rows_all.parquet",
    "runs/ctx_ownpoints_weight/chain/own_all_w100/dmp.parquet",
    "configs/ctx_ownpoints_weight_v1.json", "configs/ctx_ownpoints_dev_v1.json", "configs/ctx_ownpoints_network_v1.json",
    "configs/adsb_ctx3_combo_dev_v1.json", "configs/adsb_ctx3_mask_dev_v1.json",
    "configs/rematch_nonm_v1.json", "configs/daily_totals_v1.json", "configs/scout_extreme_shift_v1.json", "configs/scout_var_alloc_v1.json",
    "configs/stack_resid_v2.json", "configs/stack_resid_v3.json", "configs/intruder_bound_v1.json", "configs/intruder_identity_v1.json",
    "configs/own_motion_v1.json", "configs/own_motion_v2.json",
    "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet",
    "docs/ctx_ownpoints_weight/override_v66.json",
]
CODE = ["tools/rehearse_v66.py", "tools/export_ownpoints_weight_override.py", "tools/run_ownpoints_weight.py", "tools/export_ctx_ownpoints.py",
        "tools/run_ctx_ownpoints_chain.py", "tools/run_network_member.py", "tools/export_network_member.py", "tools/export_disagreement_gate.py",
        "tools/run_rematch_nonm.py", "tools/export_daily_totals.py", "tools/scout_extreme_shift.py", "tools/scout_var_alloc.py",
        "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py", "tools/run_intruder_bound.py", "tools/run_intruder_identity.py",
        "tools/run_own_motion.py", "tools/run_own_motion_v2.py"]


class PeakRSS:
    """Samples this process's resident memory every 2 s (psutil when available)."""

    def __init__(self) -> None:
        self.peak, self._stop = 0, threading.Event()
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:  # noqa: BLE001
            self._proc = None

    def __enter__(self):
        if self._proc is not None:
            threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

    def __exit__(self, *exc) -> None:
        self._stop.set()


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["numpy", "pandas", "pyarrow", "lightgbm", "sklearn", "torch"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def main() -> None:
    import export_ctx_ownpoints as E
    import run_ownpoints_weight as W
    if OUT.exists():
        raise ValueError(f"{OUT} exists; never overwrite a rehearsal.")
    if sha256_file(ROOT / V66[0]) != V66[1]:
        raise ValueError("The uploaded v66 file changed.")
    inputs_before = {p: sha256_file(ROOT / p) for p in INPUTS}
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    start, t0 = utc_now(), time.time()
    with PeakRSS() as mem:
        result = E.chain(CANDIDATE, OUT, VERSION, False)
    wall = time.time() - t0
    mine = read_json(OUT / "steps.json")
    compared = {}
    for label, path in SHIPPED_STEPS.items():
        ref = read_json(ROOT / path)
        keys = sorted(set(ref["steps"]) | set(mine["steps"]))
        compared[label] = {k: {"shipped": ref["steps"].get(k), "rehearsal": mine["steps"].get(k), "equal": ref["steps"].get(k) == mine["steps"].get(k)}
                           for k in keys}
        compared[label]["model_part_equal"] = {"equal": ref["model_part"]["sha256"] == mine["model_part"]["sha256"]}
    final = sha256_file(result)
    report = {
        "created_at_utc": utc_now(), "started_at_utc": start, "wall_seconds": round(wall, 1),
        "peak_rss_gib": round(mem.peak / 2 ** 30, 2) if mem.peak else None,
        "final": {"file": result.relative_to(ROOT).as_posix(), "sha256": final, "uploaded_v66": V66[1], "byte_identical": final == V66[1]},
        "intermediates_equal": {label: all(v["equal"] for v in d.values()) for label, d in compared.items()},
        "steps": compared,
        "inputs": inputs_before,
        "inputs_unchanged": {p: sha256_file(ROOT / p) == h for p, h in inputs_before.items()},
        "code": {p: sha256_file(ROOT / p) for p in CODE},
        "environment": versions(),
        "official_scores_used": "none", "uploaded": False,
    }
    write_json(OUT / "rehearsal.json", report)
    print(json.dumps({"final": report["final"], "intermediates_equal": report["intermediates_equal"], "wall_seconds": report["wall_seconds"],
                      "peak_rss_gib": report["peak_rss_gib"], "inputs_unchanged": all(report["inputs_unchanged"].values())}, indent=1))
    if not report["final"]["byte_identical"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
