"""Collect adsb.lol match files from the Azure workers, learn stands locally, push them back and run pass 2.

Usage: adsb_sync.py collect            (pull matches_*.parquet from all workers)
       adsb_sync.py stands             (learn stands.parquet from all local 2025 pass-1 matches)
       adsb_sync.py pass2 HOST MONTH.. (push stands.parquet to HOST and run pass 2 there, blocking)
       adsb_sync.py collect2           (pull matches2_*.parquet from all workers)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_adsb.features import learn_stands

HOSTS = ["arcw1", "arcw2", "arcw3", "arcw4", "arcw5"]
REMOTE = "/home/azureuser/prc-flight-2026"
LOCAL = ROOT / "data/external/adsblol/matches"


def run(cmd: list[str], timeout: int = 3600) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def main():
    command = sys.argv[1]
    LOCAL.mkdir(parents=True, exist_ok=True)
    if command in ("collect", "collect2"):
        pattern = "matches2_*.parquet" if command == "collect2" else "matches_*.parquet"
        for host in HOSTS:
            names = run(["ssh", "-o", "BatchMode=yes", host, f"ls {REMOTE}/data/external/adsblol/matches/{pattern} 2>/dev/null"], 120).split()
            for name in names:
                subprocess.run(["scp", "-q", f"{host}:{name}", str(LOCAL / Path(name).name)], check=True, timeout=600)
            print(host, [Path(n).name for n in names])
    elif command == "stands":
        frames = [pd.read_parquet(p) for p in sorted(LOCAL.glob("matches_2025-*.parquet"))]
        stands = learn_stands(pd.concat(frames, ignore_index=True))
        stands.to_parquet(LOCAL / "stands.parquet", index=False)
        print("stands", len(stands), "from", [p.name for p in sorted(LOCAL.glob("matches_2025-*.parquet"))])
    elif command == "pass2":
        host, months = sys.argv[2], sys.argv[3:]
        subprocess.run(["scp", "-q", str(LOCAL / "stands.parquet"), f"{host}:{REMOTE}/data/external/adsblol/matches/stands.parquet"], check=True, timeout=600)
        out = run(["ssh", "-o", "BatchMode=yes", host, f"cd {REMOTE} && .venv/bin/python -u tools/build_adsb_features.py --months {' '.join(months)} --workers 12 --stands 2>&1 | grep -v Warning | tail -3"], 3600)
        print(host, out.strip())


if __name__ == "__main__":
    main()
