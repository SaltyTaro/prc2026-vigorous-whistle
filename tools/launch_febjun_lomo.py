"""Launch, watch and mirror Tier B's tree half (docs/final_file/febjun_check.md, configs/febjun_tierb_v1.json): the 29 leaves of
ens_v20_bag_q_all / ens_v22_lirf_q refitted without February (resp. June) 2025 and scored on the held month, as configs/lomo_dev_v1.json
did for January / July (tools/launch_lomo_dev.py, unchanged, is the template).

Usage: launch_febjun_lomo.py plan HOST [HOST ...] [--speed HOST=f ...]   job list, longest first onto the least-loaded worker -> runs/febjun_lomo/plan.json
       launch_febjun_lomo.py launch [HOST ...]   sync code, check inputs, then start HOST's chain (sequential jobs) detached
       launch_febjun_lomo.py status [HOST ...]   chain markers, running jobs, load, free memory and disk
       launch_febjun_lomo.py mirror [HOST ...]   copy finished jobs' predictions, run records and models to the same local paths

Jobs: tools/run_adsb_learner_lomo.py --hold-out 2025-02 | 2025-06 at 16 threads, one at a time per worker (never two CatBoosts). Estimated
seconds: the leaf's twelve-month refit time x 11/12. Nothing is uploaded; nothing is scored here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, write_json  # noqa: E402
from prc2026_december.scoring import leaf_members  # noqa: E402

REMOTE = "/home/azureuser/prc-flight-2026"
TABLES = ["runs/adsb_v20/ens_v20_bag_q_all", "runs/adsb_v22/ens_v22_lirf_q"]
MONTHS = ["2025-02", "2025-06"]
CODE = ["tools/run_adsb_learner_lomo.py"]
FROZEN = ["tools/run_adsb_learner.py", "src/prc2026_adsb/features.py"]
CACHE = "runs/adsb_v23/_features"
BASE = "runs/febjun_lomo"


def name_of(leaf: str) -> str:
    return leaf.replace("runs/", "").replace("/", "__")


def ssh(host: str, command: str, timeout: int = 600) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", host, command],
                            capture_output=True, text=True, timeout=timeout)
    return (result.stdout + result.stderr).strip()


def copy(source: str, target: str, timeout: int = 600) -> None:
    for attempt in range(6):
        if subprocess.run(["scp", "-q", "-o", "ServerAliveInterval=30", source, target], timeout=timeout).returncode == 0:
            return
        time.sleep(15 * (attempt + 1))
    raise SystemExit(f"scp failed six times: {source} -> {target}")


def jobs() -> list[dict]:
    leaves = sorted({leaf for table in TABLES for leaf in leaf_members(ROOT, table)})
    out = []
    for leaf in leaves:
        refit = read_json(ROOT / "runs/adsb_v23" / name_of(leaf) / "refit" / "run.json")
        for month in MONTHS:
            out.append({"name": f"{name_of(leaf)}_{month}", "leaf": leaf, "month": month, "output": f"{BASE}/learners/{name_of(leaf)}",
                        "learner_kind": refit["inputs"]["kind"], "seconds": round(refit["seconds"] * 11 / 12)})
    return out


def plan(hosts: list[str], speed: dict[str, float]) -> dict:
    todo = jobs()
    load = {h: 0.0 for h in hosts}
    for job in sorted(todo, key=lambda j: -j["seconds"]):  # longest first onto the worker that finishes it first (speed-scaled seconds)
        host = min(hosts, key=lambda h: (load[h] + job["seconds"] / speed.get(h, 1.0), h))
        job["host"] = host
        load[host] += job["seconds"] / speed.get(host, 1.0)
    chains = {h: sorted([j for j in todo if j["host"] == h], key=lambda j: (j["learner_kind"] != "lightgbm", -j["seconds"])) for h in hosts}
    return {"hosts": hosts, "speed": speed, "estimated_seconds": load, "chains": chains}


def command(job: dict) -> str:
    return (f".venv/bin/python -u tools/run_adsb_learner_lomo.py --source-run {job['leaf']} --hold-out {job['month']} --output {job['output']} "
            f"--cache {CACHE} --threads 16")


def chain_script(host: str, chain: list[dict]) -> str:
    lines = ["#!/usr/bin/env bash", "set -u", f"cd {REMOTE}", f"mkdir -p {BASE}/logs", f"marker={BASE}/logs/chain_{host}.log",
             'echo "chain start $(date -u +%FT%TZ)" >> "$marker"']
    for job in chain:
        log = f"{BASE}/logs/{job['name']}.log"
        lines.append(f"if [ -e {job['output']}/lomo_{job['month']}/run.json ]; then echo \"skip {job['name']}\" >> \"$marker\"; else "
                     f"echo \"start {job['name']} $(date -u +%FT%TZ)\" >> \"$marker\"; "
                     f"if {command(job)} > {log} 2>&1; then echo \"done {job['name']} $(date -u +%FT%TZ)\" >> \"$marker\"; "
                     f"else echo \"FAILED {job['name']} $(date -u +%FT%TZ)\" >> \"$marker\"; fi; fi")
    lines.append('echo "chain done $(date -u +%FT%TZ)" >> "$marker"')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["plan", "launch", "status", "mirror"])
    parser.add_argument("hosts", nargs="*")
    parser.add_argument("--speed", nargs="*", default=[], help="HOST=factor (relative speed; default 1)")
    args = parser.parse_args()
    if args.action == "plan":
        speed = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.speed}
        p = plan(args.hosts, speed)
        write_json(ROOT / BASE / "plan.json", p)
        for h in p["hosts"]:
            print(h, round(p["estimated_seconds"][h] / 3600, 2), "h:", len(p["chains"][h]), "jobs")
        return
    p = read_json(ROOT / BASE / "plan.json")
    hosts = args.hosts or p["hosts"]
    if args.action == "launch":
        local = subprocess.run(["sha256sum", *CODE, *FROZEN], capture_output=True, text=True, cwd=ROOT).stdout.split()[0::2]
        for host in hosts:
            chain = p["chains"][host]
            ssh(host, f"mkdir -p {REMOTE}/{BASE}/logs")
            for path in CODE:
                copy(str(ROOT / path), f"{host}:{REMOTE}/{path}", 300)
            remote = ssh(host, f"cd {REMOTE} && sha256sum {' '.join(CODE + FROZEN)}").split()[0::2]
            if remote != local:
                raise SystemExit(f"{host}: code differs from the local copy.")
            missing = ssh(host, f"cd {REMOTE} && for p in {CACHE} data/december/d198dea10625b0a75977/manifest.json data/timeline/latest.json "
                                + " ".join(f"{j['leaf']}/seasonal/run.json {j['leaf']}/forward/run.json" for j in chain)
                                + "; do [ -e $p ] || echo MISSING $p; done")
            if "MISSING" in missing:
                raise SystemExit(f"{host}: {missing}")
            script = ROOT / BASE / f"chain_{host}.sh"
            script.write_text(chain_script(host, chain), encoding="utf-8", newline="\n")
            copy(str(script), f"{host}:{REMOTE}/{BASE}/chain_{host}.sh", 300)
            try:
                print(host, ssh(host, f"cd {REMOTE} && setsid -f bash {BASE}/chain_{host}.sh < /dev/null > /dev/null 2>&1; sleep 3; "
                                      f"pgrep -f 'chain_{host}.sh' | wc -l", timeout=40))
            except subprocess.TimeoutExpired:
                print(host, "launched (the detached session kept the ssh channel open)")
    elif args.action == "status":
        for host in hosts:
            print("==", host)
            print(ssh(host, f"cd {REMOTE} && tail -n 3 {BASE}/logs/chain_{host}.log 2>/dev/null; grep -c '^done' {BASE}/logs/chain_{host}.log 2>/dev/null; "
                            f"grep FAILED {BASE}/logs/chain_{host}.log 2>/dev/null; "
                            f"pgrep -af 'run_adsb_learner_lomo' | grep -v pgrep | cut -c1-160; uptime | sed 's/.*load/load/'; "
                            "free -g | awk '/Mem/{print \"avail \"$7\" GB\"}'; df -h / | tail -1"))
    else:
        for host in hosts:
            for job in p["chains"][host]:
                src = f"{job['output']}/lomo_{job['month']}"
                files = ["run.json", "predictions.parquet", "model.txt" if job["learner_kind"] == "lightgbm" else "model.cbm"]
                if (ROOT / src / "run.json").exists():
                    continue
                if "yes" not in ssh(host, f"[ -e {REMOTE}/{src}/run.json ] && echo yes"):
                    continue
                (ROOT / src).mkdir(parents=True, exist_ok=True)
                for f in files[::-1]:  # run.json last, so a local run.json means a complete mirror
                    copy(f"{host}:{REMOTE}/{src}/{f}", str(ROOT / src / f), 3600)
                print("mirrored", job["name"], "from", host)


if __name__ == "__main__":
    main()
