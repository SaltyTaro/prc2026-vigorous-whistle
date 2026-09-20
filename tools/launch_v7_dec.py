"""Run tools/run_v7_dec.py folds on Azure workers and mirror their outputs (adsb_v24).

Usage: launch_v7_dec.py launch HOST FOLD        start one fold (A|B|C|D|refit) under nohup on HOST (the runner is copied first, hash-checked)
       launch_v7_dec.py status HOST [HOST ...]  fold directories, log tails, uptime
       launch_v7_dec.py mirror HOST FOLD        copy predictions / nodes / run record to runs/adsb_v24/v7/<fold>/ (models stay on the worker)

The worker must already hold the frozen timeline bundle, its parents, the raw files and the December build (the v23 fan-out). Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/azureuser/prc-flight-2026"
OUTPUT_ROOT = "runs/adsb_v24/v7"
DECEMBER = "data/december/d198dea10625b0a75977"
FILES = ["tools/run_v7_dec.py", "src/prc2026_timeline/native.py", "src/prc2026_timeline/encoding.py", "src/prc2026_timeline/inference.py", "src/prc2026_timeline/features.py",
         "src/prc2026_surface_airports/inference.py", "src/prc2026_surface_refit/inference.py", "src/prc2026_surface_refit/guards.py", "src/prc2026_nmclock/features.py",
         "configs/surface_refit_v1.json", "configs/timeline_v1.json", "configs/surface_airports_v1.json"]


def ssh(host: str, command: str, timeout: int = 600) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True, timeout=timeout)
    return (result.stdout + result.stderr).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "status", "mirror"])
    parser.add_argument("host")
    parser.add_argument("fold", nargs="?")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.command == "status":
        for host in [args.host, *([args.fold] if args.fold else [])]:
            print("==", host)
            print(ssh(host, f"cd {REMOTE} && for d in {OUTPUT_ROOT}/*/; do n=$(basename $d); echo \"$n: $(ls $d 2>/dev/null | tr '\\n' ' ' | cut -c1-200)\"; done; tail -n 3 {OUTPUT_ROOT}/launch_*.log 2>/dev/null | cut -c1-200; uptime; free -g | head -2"))
        return
    if not args.fold:
        raise SystemExit("FOLD required")
    if args.command == "launch":
        subprocess.run(["scp", "-q", str(ROOT / "tools/run_v7_dec.py"), f"{args.host}:{REMOTE}/tools/run_v7_dec.py"], check=True, timeout=120)
        remote = ssh(args.host, f"cd {REMOTE} && sha256sum {' '.join(FILES)}").split()
        local = subprocess.run(["sha256sum", *FILES], capture_output=True, text=True, cwd=ROOT).stdout.split()
        if remote[0::2] != local[0::2]:
            raise SystemExit(f"Runner, frozen module or config differs on {args.host}:\n{remote}\n{local}")
        job = f"tools/run_v7_dec.py --fold {args.fold} --output {OUTPUT_ROOT} --december {DECEMBER} --threads {args.threads}"
        script = (f"cd {REMOTE} && mkdir -p {OUTPUT_ROOT} && (echo \"=== fold {args.fold} $(date -u +%FT%TZ)\" >> {OUTPUT_ROOT}/launch_{args.fold}.log && "
                  f".venv/bin/python -u {job} >> {OUTPUT_ROOT}/launch_{args.fold}.log 2>&1; echo \"=== fold {args.fold} done $(date -u +%FT%TZ)\" >> {OUTPUT_ROOT}/launch_{args.fold}.log) > /dev/null 2>&1\n")
        local_script = ROOT / "runs" / f"chain24_{args.host}_{args.fold}.sh"
        local_script.write_text(script, encoding="utf-8", newline="\n")
        ssh(args.host, f"mkdir -p {REMOTE}/{OUTPUT_ROOT}")
        subprocess.run(["scp", "-q", str(local_script), f"{args.host}:{REMOTE}/{OUTPUT_ROOT}/chain_{args.fold}.sh"], check=True, timeout=300)
        try:
            print(ssh(args.host, f"cd {REMOTE} && nohup setsid bash {OUTPUT_ROOT}/chain_{args.fold}.sh < /dev/null > /dev/null 2>&1 & sleep 2; ps -eo cmd | grep -c '[r]un_v7_dec.py'", timeout=30)[-200:])
        except subprocess.TimeoutExpired:
            print("launched (the detached session kept the ssh channel open; the fold is running)")
    else:
        target = ROOT / OUTPUT_ROOT / args.fold
        target.mkdir(parents=True, exist_ok=True)
        files = ["run.json", "nodes.json", "predictions.parquet" if args.fold != "refit" else "ranking_predictions.parquet"]
        for file in files:
            subprocess.run(["scp", "-q", f"{args.host}:{REMOTE}/{OUTPUT_ROOT}/{args.fold}/{file}", str(target / file)], check=True, timeout=1800)
        subprocess.run(["scp", "-q", f"{args.host}:{REMOTE}/{OUTPUT_ROOT}/launch_{args.fold}.log", str(target / "launch.log")], check=True, timeout=300)
        print("mirrored fold", args.fold, "from", args.host, "->", target)


if __name__ == "__main__":
    main()
