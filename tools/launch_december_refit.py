"""Run twelve-month refits of banked learners on Azure workers (tools/run_adsb_learner_dec.py) and mirror them.

Usage: launch_december_refit.py launch HOST LEAF [LEAF ...]     start one chain (sequential jobs) under nohup on HOST
       launch_december_refit.py status HOST [HOST ...]          which refits exist, log tails
       launch_december_refit.py mirror HOST LEAF [LEAF ...]     copy model, ranking predictions and run record to runs/adsb_v23/<name>/refit/

LEAF is a banked learner directory (e.g. runs/adsb_v15/lgb_s16); the refit is written to runs/adsb_v23/<dir>__<name>/refit/ on the
worker and mirrored to the same path locally. The worker must already hold the frozen bundles, the December build, the December
match table and v19 features, and the leaf's fold records (runs/december/source_records.tar.gz). Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json

REMOTE = "/home/azureuser/prc-flight-2026"
OUTPUT_ROOT = "runs/adsb_v23"
DECEMBER = "data/december/d198dea10625b0a75977"


def name_of(leaf: str) -> str:
    return leaf.replace("runs/", "").replace("/", "__")


def ssh(host: str, command: str, timeout: int = 600) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True, timeout=timeout)
    return (result.stdout + result.stderr).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "status", "mirror"])
    parser.add_argument("host")
    parser.add_argument("leaves", nargs="*")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.command == "status":
        for host in [args.host, *args.leaves]:
            print("==", host)
            print(ssh(host, f"cd {REMOTE} && for d in {OUTPUT_ROOT}/*/; do n=$(basename $d); echo \"$n: $(ls $d/refit 2>/dev/null | tr '\\n' ' ')\"; done; tail -n 2 {OUTPUT_ROOT}/*/launch.log 2>/dev/null | tail -n 24 | cut -c1-160; uptime"))
        return
    if args.command == "launch":
        remote_hash = ssh(args.host, f"cd {REMOTE} && sha256sum tools/run_adsb_learner_dec.py tools/run_adsb_learner.py src/prc2026_adsb/features.py").split()
        local = subprocess.run(["sha256sum", "tools/run_adsb_learner_dec.py", "tools/run_adsb_learner.py", "src/prc2026_adsb/features.py"], capture_output=True, text=True, cwd=ROOT).stdout.split()
        if remote_hash[0::2] != local[0::2]:
            raise SystemExit(f"Remote runner or feature code differs from the local copy on {args.host}.")
        for leaf in args.leaves:
            for fold in ["seasonal", "forward"]:
                if not (ROOT / leaf / fold / "run.json").exists():
                    raise SystemExit(f"Missing local fold record {leaf}/{fold}")
        chain = [f"mkdir -p {OUTPUT_ROOT}/_features"]
        for leaf in args.leaves:
            output = f"{OUTPUT_ROOT}/{name_of(leaf)}"
            job = ["tools/run_adsb_learner_dec.py", "--source-run", leaf, "--output", output, "--december", DECEMBER, "--threads", str(args.threads)]
            chain.append(f"mkdir -p {output} && ln -sfn {REMOTE}/{OUTPUT_ROOT}/_features {output}/features")
            chain.append(f"echo \"=== {leaf} refit12 $(date -u +%FT%TZ)\" >> {output}/launch.log && .venv/bin/python -u {' '.join(shlex.quote(p) for p in job)} >> {output}/launch.log 2>&1")
        script = f"cd {REMOTE} && mkdir -p {OUTPUT_ROOT} && ({' && '.join(chain)}; echo \"=== chain done $(date -u +%FT%TZ)\" >> {OUTPUT_ROOT}/chain_{args.host}.log) > /dev/null 2>&1\n"
        local_script = ROOT / "runs" / f"chain23_{args.host}.sh"
        local_script.write_text(script, encoding="utf-8", newline="\n")
        ssh(args.host, f"mkdir -p {REMOTE}/{OUTPUT_ROOT}")
        subprocess.run(["scp", "-q", str(local_script), f"{args.host}:{REMOTE}/{OUTPUT_ROOT}/chain_{args.host}.sh"], check=True, timeout=300)
        try:
            print(ssh(args.host, f"cd {REMOTE} && nohup setsid bash {OUTPUT_ROOT}/chain_{args.host}.sh < /dev/null > /dev/null 2>&1 & sleep 2; pgrep -f 'run_adsb_learner_dec.py' | wc -l", timeout=30)[-200:])
        except subprocess.TimeoutExpired:
            print("launched (the detached session kept the ssh channel open; the chain is running)")
    else:
        for leaf in args.leaves:
            target = ROOT / OUTPUT_ROOT / name_of(leaf) / "refit"
            target.mkdir(parents=True, exist_ok=True)
            kind = (read_json(ROOT / leaf / "seasonal" / "run.json")["inputs"].get("kind") or "lightgbm")
            for file in ["ranking_predictions.parquet", "run.json", "model.txt" if kind == "lightgbm" else "model.cbm"]:
                subprocess.run(["scp", "-q", f"{args.host}:{REMOTE}/{OUTPUT_ROOT}/{name_of(leaf)}/refit/{file}", str(target / file)], check=True, timeout=1800)
            subprocess.run(["scp", "-q", f"{args.host}:{REMOTE}/{OUTPUT_ROOT}/{name_of(leaf)}/launch.log", str(target.parent / "launch.log")], check=True, timeout=300)
            print("mirrored", leaf, "from", args.host)


if __name__ == "__main__":
    main()
