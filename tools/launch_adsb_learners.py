"""Run declared full-data ADS-B learner variants on Azure workers and mirror their outputs.

Usage: launch_adsb_learners.py launch HOST VARIANT [VARIANT ...]   sync the runner, start the chain (seasonal, forward, refit per variant) under nohup
       launch_adsb_learners.py status HOST [HOST ...]               show which fold records exist and the log tails
       launch_adsb_learners.py mirror HOST VARIANT [VARIANT ...]    copy predictions, models and run records to runs/adsb_v15/<variant>/

Variants come from the declaration's `learners` block (configs/adsb_v15_v1.json by default): kind, seed,
parameter overrides, round cap, patience and label cap. Every variant fits without LIRF on the timeline
bundle plus the ADS-B features of data/external/adsblol/matches. Nothing is uploaded.
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
FOLDS = ["seasonal", "forward", "refit"]
FILES = {"seasonal": ["predictions.parquet", "run.json"], "forward": ["predictions.parquet", "run.json"], "refit": ["ranking_predictions.parquet", "run.json"]}


def ssh(host: str, command: str, timeout: int = 600) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True, timeout=timeout)
    return (result.stdout + result.stderr).strip()


def job_command(spec: dict, output: str, fold: str) -> str:
    parts = ["tools/run_adsb_learner.py", "--fold", fold, "--kind", spec["kind"], "--seed", str(spec["seed"]), "--threads", "16",
             "--exclude-airports", "LIRF", "--output", output, "--matches-dir", "data/external/adsblol/matches"]
    for key, value in spec.get("parameters", {}).items():
        parts += ["--param", f"{key}={str(value).lower() if isinstance(value, bool) else value}"]
    if spec.get("target"):
        parts += ["--target", spec["target"]]
    if spec.get("rounds") is not None:
        parts += ["--rounds", str(spec["rounds"])]
    if spec.get("patience") is not None:
        parts += ["--patience", str(spec["patience"])]
    if spec.get("max_label") is not None:
        parts += ["--max-label", str(spec["max_label"])]
    return ".venv/bin/python -u " + " ".join(shlex.quote(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "status", "mirror"])
    parser.add_argument("host")
    parser.add_argument("names", nargs="*")
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v15_v1.json"))
    parser.add_argument("--feature-cache", default="runs/adsb_full_cat/features", help="Existing remote ADS-B feature cache to link (built from the same matches and stands)")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    learners = config["learners"]
    root = config["learner_output_root"]
    hosts = [args.host, *args.names] if args.command == "status" else [args.host]
    if args.command == "status":
        for host in hosts:
            print("==", host)
            print(ssh(host, f"cd {REMOTE} && for d in {root}/*/; do n=$(basename $d); echo \"$n: $(ls $d | tr '\\n' ' ')\"; done; tail -n 2 {root}/*/launch.log 2>/dev/null | tail -n 30; uptime"))
        return
    for name in args.names:
        if name not in learners:
            raise SystemExit(f"Unknown learner variant: {name}")
    if args.command == "launch":
        subprocess.run(["scp", "-q", str(ROOT / "tools/run_adsb_learner.py"), f"{args.host}:{REMOTE}/tools/run_adsb_learner.py"], check=True, timeout=300)
        remote_hash = ssh(args.host, f"sha256sum {REMOTE}/tools/run_adsb_learner.py {REMOTE}/src/prc2026_adsb/features.py").split()
        local = subprocess.run(["sha256sum", "tools/run_adsb_learner.py", "src/prc2026_adsb/features.py"], capture_output=True, text=True, cwd=ROOT).stdout.split()
        if remote_hash[0] != local[0] or remote_hash[2] != local[2]:
            raise SystemExit("Remote runner or feature code differs from the local copy.")
        chain = []
        for name in args.names:
            output = f"{root}/{name}"
            chain.append(f"mkdir -p {output} && ln -sfn {REMOTE}/{args.feature_cache} {output}/features")
            for fold in FOLDS:
                chain.append(f"echo \"=== {name} {fold} $(date -u +%FT%TZ)\" >> {output}/launch.log && {job_command(learners[name], output, fold)} >> {output}/launch.log 2>&1")
        script = f"cd {REMOTE} && ({' && '.join(chain)}; echo \"=== chain done $(date -u +%FT%TZ)\" >> {root}/chain_{args.host}.log) > /dev/null 2>&1 &"
        print(ssh(args.host, f"nohup bash -c {shlex.quote(script)} > /dev/null 2>&1; sleep 1; pgrep -f 'run_adsb_learner.py --fold' | wc -l")[-200:])
    else:
        for name in args.names:
            for fold, files in FILES.items():
                target = ROOT / root / name / fold
                target.mkdir(parents=True, exist_ok=True)
                for file in files + (["model.txt"] if learners[name]["kind"] == "lightgbm" else ["model.cbm"]):
                    subprocess.run(["scp", "-q", f"{args.host}:{REMOTE}/{root}/{name}/{fold}/{file}", str(target / file)], check=True, timeout=1800)
            subprocess.run(["scp", "-q", f"{args.host}:{REMOTE}/{root}/{name}/launch.log", str(ROOT / root / name / "launch.log")], check=True, timeout=300)
            print("mirrored", name, "from", args.host)


if __name__ == "__main__":
    main()
