"""Launch, watch and mirror the unmatched-row specialist design (configs/adsb_unm_design_v1.json) on the Azure workers.

Usage: launch_adsb_unm.py launch HOST       refuse while any training process runs there; sync and hash-check the code, declaration and
                                            inputs; start the host's declared chain (execution.assignment) under nohup, 16 threads, runs in sequence
       launch_adsb_unm.py status HOST [...]  which runs have a record, FAILED lines, training processes, log tails
       launch_adsb_unm.py mirror HOST       copy each assigned run's stop predictions, model and record (and its log) back
Each worker builds its own ADS-B table cache (runs/adsb_unm/features) from its hash-checked matches and stands on the first run.
Nothing is uploaded; the declaration must carry its declared_at_utc before anything launches.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file  # noqa: E402

REMOTE = "/home/azureuser/prc-flight-2026"
DECLARATION = "configs/adsb_unm_design_v1.json"
CODE = ["tools/run_adsb_unm.py", "tools/run_adsb_learner.py", DECLARATION]
CHECKED = ["src/prc2026_adsb/features.py", "configs/timeline_v1.json", "data/timeline/latest.json"]
DESIGN_MONTHS = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09", "2025-10"]
TRAINING = r"tools/run_adsb_(ctx|rctx|learner|unm|nn)\.py"


def ssh(host: str, command: str, timeout: int = 600) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command], capture_output=True, text=True, timeout=timeout)
    return (result.stdout + result.stderr).strip()


def remote_hashes(host: str, paths: list[str]) -> dict[str, str | None]:
    out = ssh(host, f"cd {REMOTE} && for f in {' '.join(shlex.quote(p) for p in paths)}; do if [ -f \"$f\" ]; then sha256sum \"$f\"; else echo \"missing $f\"; fi; done")
    hashes = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "missing":
            hashes[parts[1]] = None
        elif len(parts) == 2:
            hashes[parts[1].lstrip("*")] = parts[0]
    return {p: hashes.get(p) for p in paths}


def chain_script(decl: dict, host: str) -> str:
    root = decl["output_root"]
    lines = [f"cd {REMOTE}", f"mkdir -p {root} runs/adsb_unm/logs", f"log=runs/adsb_unm/logs/chain_{host}.log"]
    for run in decl["execution"]["assignment"][host]:
        command = f".venv/bin/python -u tools/run_adsb_unm.py --declaration {DECLARATION} --run {run} --threads 16"
        lines.append(f"echo \"=== {run} start $(date -u +%FT%TZ)\" >> $log")
        lines.append(f"if {command} >> runs/adsb_unm/logs/{run}.log 2>&1; then echo \"done {run} $(date -u +%FT%TZ)\" >> $log; "
                     f"else echo \"FAILED {run} $(date -u +%FT%TZ)\" >> $log; fi")
    lines.append("echo \"=== chain done $(date -u +%FT%TZ)\" >> $log")
    return "\n".join(lines) + "\n"


def launch(decl: dict, host: str) -> None:
    if not decl.get("declared_at_utc"):
        raise SystemExit("The declaration is a draft (declared_at_utc is null): finalise and commit it first.")
    if host not in decl["execution"]["assignment"]:
        raise SystemExit(f"{host} has no declared runs.")
    busy = ssh(host, f"pgrep -fc '{TRAINING}' || true").strip().splitlines()
    if not busy or busy[-1] != "0":
        raise SystemExit(f"{host} still runs a training process ({busy}); wait for it to finish.")
    existing = ssh(host, f"cd {REMOTE} && ls {decl['output_root']}/*/design/run.json 2>/dev/null | wc -l").strip()
    if existing not in ("", "0"):
        raise SystemExit(f"{host} already holds {existing} design records; refusing to launch a duplicate chain.")
    for path in CODE:
        ssh(host, f"mkdir -p {REMOTE}/{Path(path).parent.as_posix()}")
        subprocess.run(["scp", "-q", str(ROOT / path), f"{host}:{REMOTE}/{path}"], check=True, timeout=300)
    common = decl["learner_common"]
    inputs = [*CODE, *CHECKED, f"{common['matches_dir']}/stands.parquet", *[f"{common['matches_dir']}/matches2_{m}.parquet" for m in DESIGN_MONTHS],
              *[f"{common['extra_features']}/features_{common['extra_tag']}_{m}.parquet" for m in DESIGN_MONTHS]]
    remote = remote_hashes(host, inputs)
    for path in inputs:
        local = sha256_file(ROOT / path)
        if remote[path] is None and path.startswith((common["matches_dir"], common["extra_features"])):
            ssh(host, f"mkdir -p {REMOTE}/{Path(path).parent.as_posix()}")
            subprocess.run(["scp", "-q", str(ROOT / path), f"{host}:{REMOTE}/{path}"], check=True, timeout=1800)
            remote[path] = remote_hashes(host, [path])[path]
        if remote[path] != local:
            raise SystemExit(f"{host}:{path} differs from the local copy ({remote[path]} vs {local}).")
    bundle = read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest = remote_hashes(host, [f"{bundle}/manifest.json"])[f"{bundle}/manifest.json"]
    if manifest != sha256_file(ROOT / bundle / "manifest.json"):
        raise SystemExit(f"{host}: the timeline bundle manifest differs or is missing.")
    local_script = ROOT / "runs/adsb_unm" / f"chain_{host}.sh"
    local_script.parent.mkdir(parents=True, exist_ok=True)
    local_script.write_text(chain_script(decl, host), encoding="utf-8", newline="\n")
    ssh(host, f"mkdir -p {REMOTE}/runs/adsb_unm")
    subprocess.run(["scp", "-q", str(local_script), f"{host}:{REMOTE}/runs/adsb_unm/chain_{host}.sh"], check=True, timeout=300)
    # stdin detached, or the ssh session stays open for as long as the chain runs
    print(ssh(host, f"cd {REMOTE} && nohup setsid bash runs/adsb_unm/chain_{host}.sh < /dev/null > /dev/null 2>&1 & sleep 3; "
                    f"echo {host} launched {' '.join(decl['execution']['assignment'][host])}; pgrep -fc 'tools/run_adsb_unm.py'"))


def status(decl: dict, hosts: list[str]) -> None:
    for host in hosts:
        runs = decl["execution"]["assignment"].get(host, [])
        print("==", host, ssh(host, f"cd {REMOTE} && for r in {' '.join(runs)}; do [ -f {decl['output_root']}/$r/design/run.json ] && echo \"$r: done\" || echo \"$r: -\"; done; "
                                    f"grep -h FAILED runs/adsb_unm/logs/chain_{host}.log 2>/dev/null; tail -n 2 runs/adsb_unm/logs/chain_{host}.log 2>/dev/null; "
                                    f"echo procs $(pgrep -fc 'tools/run_adsb_unm.py'); uptime"))


def mirror(decl: dict, host: str) -> None:
    root = decl["output_root"]
    for run in decl["execution"]["assignment"][host]:
        spec = decl["runs"][run]
        target = ROOT / root / run / "design"
        target.mkdir(parents=True, exist_ok=True)
        for name in ["stop_predictions.parquet", "run.json", "model.txt" if spec["kind"] == "lightgbm" else "model.cbm"]:
            subprocess.run(["scp", "-q", f"{host}:{REMOTE}/{root}/{run}/design/{name}", str(target / name)], check=True, timeout=1800)
        subprocess.run(["scp", "-q", f"{host}:{REMOTE}/runs/adsb_unm/logs/{run}.log", str(target.parent / "run.log")], check=True, timeout=300)
        print("mirrored", run, "from", host)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["launch", "status", "mirror"])
    parser.add_argument("hosts", nargs="+")
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    if args.command == "status":
        status(decl, args.hosts)
        return
    for host in args.hosts:
        (launch if args.command == "launch" else mirror)(decl, host)


if __name__ == "__main__":
    main()
