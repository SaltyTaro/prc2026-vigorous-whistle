"""Launch, watch and mirror the leave-one-month-out development fits of configs/lomo_dev_v1.json on the Azure workers.

Usage: launch_lomo_dev.py plan                      the job list with estimated seconds, assigned longest-first to the hosts -> runs/lomo_dev/plan.json
       launch_lomo_dev.py launch HOST [HOST ...]    sync code + reference files, then start HOST's chain (sequential jobs) under nohup
       launch_lomo_dev.py status [HOST ...]          chain markers, running jobs, uptime
       launch_lomo_dev.py mirror [HOST ...]          copy finished jobs' predictions, run records (and models) to the same local paths

Jobs: the 29 leaves of ens_v20_bag_q_all / ens_v22_lirf_q x {2025-01, 2025-07} (tools/run_adsb_learner_lomo.py, 16 threads), v7 L01 / L07
(tools/run_v7_lomo.py, arcw0 only: the only worker holding run_v7_dec.py and its runs) and the reproduction check (lgb_clean held out on
2025-12). One job at a time per worker, so never more than one CatBoost. Before its first job every chain stops round E's refit12 processes
(`--fold refit12`, allowed by the parallel session); on arcw4 it first waits until both of its round-E chains have finished their refit step.
Estimated seconds: the leaf's twelve-month refit time x 11/12 (v7: 1,800 s). Nothing is uploaded.
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
HOSTS = ["arcw0", "arcw1", "arcw3", "arcw4", "arcw5", "arcw6", "arcw7"]
TABLES = ["runs/adsb_v20/ens_v20_bag_q_all", "runs/adsb_v22/ens_v22_lirf_q"]
MONTHS = ["2025-01", "2025-07"]
LATE = {"arcw4": ["runs/adsb_ctx/logs/chain_ctx_e32c_s505.log", "runs/adsb_ctx/logs/chain_ctx_e32c_s513.log"]}  # round E refit still running at launch
CODE = ["tools/run_adsb_learner_lomo.py", "tools/run_v7_lomo.py", "configs/lomo_dev_v1.json"]
FROZEN = ["tools/run_adsb_learner.py", "src/prc2026_adsb/features.py"]
CACHE = "runs/adsb_v23/_features"


def name_of(leaf: str) -> str:
    return leaf.replace("runs/", "").replace("/", "__")


def ssh(host: str, command: str, timeout: int = 600) -> str:
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command], capture_output=True, text=True, timeout=timeout)
    return (result.stdout + result.stderr).strip()


def copy(source: str, target: str, timeout: int = 600) -> None:
    """scp with retries (arcw5 drops connections intermittently)."""
    for attempt in range(6):
        if subprocess.run(["scp", "-q", source, target], timeout=timeout).returncode == 0:
            return
        time.sleep(15 * (attempt + 1))
    raise SystemExit(f"scp failed six times: {source} -> {target}")


def jobs() -> list[dict]:
    leaves = sorted({leaf for table in TABLES for leaf in leaf_members(ROOT, table)})
    out = [{"name": "repro_lgb_clean_2025-12", "kind": "learner", "leaf": "runs/adsb_v15/lgb_clean", "month": "2025-12",
            "output": "runs/lomo_dev/repro/adsb_v15__lgb_clean", "seconds": 300, "host": None}]
    for leaf in leaves:
        refit = read_json(ROOT / "runs/adsb_v23" / name_of(leaf) / "refit" / "run.json")
        for month in MONTHS:
            out.append({"name": f"{name_of(leaf)}_{month}", "kind": "learner", "leaf": leaf, "month": month, "output": f"runs/lomo_dev/learners/{name_of(leaf)}",
                        "learner_kind": refit["inputs"]["kind"], "seconds": round(refit["seconds"] * 11 / 12), "host": None})
    for fold in ["L01", "L07"]:
        out.append({"name": f"v7_{fold}", "kind": "v7", "fold": fold, "output": "runs/lomo_dev/v7", "seconds": 1800, "host": "arcw0"})
    return out


def plan() -> dict:
    todo = jobs()
    load = {h: (1800 if h in LATE else 0) for h in HOSTS}  # arcw4 starts after round E's refit
    for job in sorted(todo, key=lambda j: (j["host"] is None, -j["seconds"])):  # pinned first, then longest first
        host = job["host"] or min(HOSTS, key=lambda h: (load[h], h))
        job["host"] = host
        load[host] += job["seconds"]
    chains = {h: sorted([j for j in todo if j["host"] == h], key=lambda j: (j["kind"] != "learner" or not j["name"].startswith("repro"), j["kind"] != "v7",
                                                                              j.get("learner_kind") != "lightgbm", -j["seconds"])) for h in HOSTS}
    return {"hosts": HOSTS, "estimated_seconds": load, "chains": chains}


def command(job: dict) -> str:
    if job["kind"] == "v7":
        return f".venv/bin/python -u tools/run_v7_lomo.py --fold {job['fold']} --output {job['output']}"
    return (f".venv/bin/python -u tools/run_adsb_learner_lomo.py --source-run {job['leaf']} --hold-out {job['month']} --output {job['output']} "
            f"--cache {CACHE} --threads 16")


def chain_script(host: str, chain: list[dict]) -> str:
    lines = ["#!/usr/bin/env bash", "set -u", f"cd {REMOTE}", "mkdir -p runs/lomo_dev/logs", f"marker=runs/lomo_dev/logs/chain_{host}.log",
             'echo "chain start $(date -u +%FT%TZ)" >> "$marker"']
    if host in LATE:
        cond = " && ".join(f"grep -qE 'done refit |FAILED' {m}" for m in LATE[host])  # a failed round-E step must not hang the chain
        lines += [f"until {cond}; do sleep 60; done", 'echo "round E refit done $(date -u +%FT%TZ)" >> "$marker"', "sleep 30"]
    lines += ["for i in 1 2 3 4 5 6; do pkill -f -- '--fold refit12' && echo \"stopped round E refit12 $(date -u +%FT%TZ)\" >> \"$marker\"; sleep 10; done"]
    for job in chain:
        log = f"runs/lomo_dev/logs/{job['name']}.log"
        lines.append(f"echo \"start {job['name']} $(date -u +%FT%TZ)\" >> \"$marker\"")
        lines.append(f"if {command(job)} > {log} 2>&1; then echo \"done {job['name']} $(date -u +%FT%TZ)\" >> \"$marker\"; "
                     f"else echo \"FAILED {job['name']} $(date -u +%FT%TZ)\" >> \"$marker\"; fi")
    lines.append('echo "chain done $(date -u +%FT%TZ)" >> "$marker"')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["plan", "launch", "status", "mirror"])
    parser.add_argument("hosts", nargs="*")
    args = parser.parse_args()
    if args.action == "plan":
        p = plan()
        write_json(ROOT / "runs/lomo_dev/plan.json", p)
        for h in HOSTS:
            print(h, round(p["estimated_seconds"][h] / 3600, 2), "h:", ", ".join(j["name"] for j in p["chains"][h]))
        return
    p = read_json(ROOT / "runs/lomo_dev/plan.json")
    hosts = args.hosts or HOSTS
    if args.action == "launch":
        local = subprocess.run(["sha256sum", *CODE, *FROZEN], capture_output=True, text=True, cwd=ROOT).stdout.split()[0::2]
        for host in hosts:
            chain = p["chains"][host]
            ssh(host, f"mkdir -p {REMOTE}/runs/lomo_dev/logs {REMOTE}/configs {REMOTE}/tools")
            for path in CODE:
                copy(str(ROOT / path), f"{host}:{REMOTE}/{path}", 300)
            remote = ssh(host, f"cd {REMOTE} && sha256sum {' '.join(CODE + FROZEN)}").split()[0::2]
            if remote != local:
                raise SystemExit(f"{host}: code differs from the local copy.")
            for job in chain:  # the held-row reference: the leaf's seasonal predictions (and fold records, already there)
                if job["kind"] == "learner" and job["month"] != "2025-12":
                    ssh(host, f"mkdir -p {REMOTE}/{job['leaf']}/seasonal")
                    copy(str(ROOT / job["leaf"] / "seasonal/predictions.parquet"), f"{host}:{REMOTE}/{job['leaf']}/seasonal/predictions.parquet")
            missing = ssh(host, f"cd {REMOTE} && for p in {CACHE} data/december/d198dea10625b0a75977/manifest.json "
                                + " ".join(f"{j['leaf']}/seasonal/run.json {j['leaf']}/forward/run.json" for j in chain if j["kind"] == "learner")
                                + (" tools/run_v7_dec.py" if any(j["kind"] == "v7" for j in chain) else "") + "; do [ -e $p ] || echo MISSING $p; done")
            if "MISSING" in missing:
                raise SystemExit(f"{host}: {missing}")
            script = ROOT / "runs/lomo_dev" / f"chain_{host}.sh"
            script.write_text(chain_script(host, chain), encoding="utf-8", newline="\n")
            copy(str(script), f"{host}:{REMOTE}/runs/lomo_dev/chain_{host}.sh", 300)
            try:
                print(host, ssh(host, f"cd {REMOTE} && nohup setsid bash runs/lomo_dev/chain_{host}.sh < /dev/null > /dev/null 2>&1 & sleep 3; pgrep -f 'chain_{host}.sh' | wc -l", timeout=40))
            except subprocess.TimeoutExpired:
                print(host, "launched (the detached session kept the ssh channel open)")
    elif args.action == "status":
        for host in hosts:
            print("==", host)
            print(ssh(host, f"cd {REMOTE} && tail -n 4 runs/lomo_dev/logs/chain_{host}.log 2>/dev/null; pgrep -af 'run_adsb_learner_lomo|run_v7_lomo' | grep -v pgrep | cut -c1-150; "
                            f"uptime | sed 's/.*load/load/'; free -g | awk '/Mem/{{print \"avail \"$7\" GB\"}}'"))
    else:
        for host in hosts:
            for job in p["chains"][host]:
                if job["kind"] == "v7":
                    src = f"{job['output']}/{job['fold']}"
                    files = ["run.json", "lomo.json", "predictions.parquet", "nodes.json"]
                else:
                    src = f"{job['output']}/lomo_{job['month']}"
                    files = ["run.json", "predictions.parquet", "model.txt" if job.get("learner_kind", "lightgbm") == "lightgbm" else "model.cbm"]
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
