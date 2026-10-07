"""stack_queue_v1 driver on an Azure VM (configs/stack_queue_v1.json; docs/stack_queue/HANDOFF.md): runs tools/run_stack_queue.py's steps as
child processes with their dependencies, at most MAXJ at a time and only while MemAvailable > MIN_FREE_GB. Markers
runs/stack_queue/state/<job>.done|.failed, logs runs/stack_queue/logs/<job>.log; a rerun skips finished jobs. On the first failure no new
job starts; running jobs finish. The children run with the lazy-fetch hook (tools/ctx_own2_vmfetch, file server via reverse tunnel).
At the end: runs/stack_queue/mirror/out.tar (+ .sha256) and state/all.done.

Usage (on the VM, workspace root): setsid nohup .venv/bin/python -u tools/stack_queue_remote.py > runs/stack_queue/remote.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

W = Path(__file__).resolve().parents[1]
ST, LG = W / "runs/stack_queue/state", W / "runs/stack_queue/logs"
MAXJ, MIN_FREE_GB = 6, 10.0
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
TOOL = [str(W / ".venv/bin/python"), "-X", "utf8", "-B", "-u", "tools/run_stack_queue.py"]
JOBS: list[tuple[str, list[str], list[str]]] = [("warm_fit", ["--warm", "fit"], [])]
JOBS += [(f"dev_off_{m}", ["--dev-month", m, "--queue", "off"], ["warm_fit"]) for m in MONTHS]
JOBS += [("rank_off", ["--rank-fit", "--queue", "off"], ["warm_fit"]), ("warm_chain", ["--warm", "chain"], ["warm_fit"])]
JOBS += [(f"dev_on_{m}", ["--dev-month", m, "--queue", "on"], ["warm_fit"]) for m in MONTHS]
JOBS += [("rank_on", ["--rank-fit", "--queue", "on"], ["warm_fit"]),
         ("rank_identity", ["--rank-identity"], ["rank_off", "warm_chain"]),
         ("dev_score", ["--dev-score"], [f"dev_{q}_{m}" for q in ["off", "on"] for m in MONTHS] + ["warm_chain"]),
         ("rank_check", ["--rank-check"], ["dev_score", "rank_identity", "rank_on"]),
         ("rank_dry", ["--rank-dry"], ["rank_check"])]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mem_free_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    return 0.0


def selected() -> bool:
    p = W / "docs/stack_queue/results_v1.json"
    return p.exists() and json.loads(p.read_text()).get("selected") == "q65_bag10"


def main() -> None:
    ST.mkdir(parents=True, exist_ok=True)
    LG.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PRC_ROOT": str(W), "PRC_FETCH_URL": "http://127.0.0.1:8768", "PRC_FETCH_LOG": str(W / "runs/stack_queue/vm_fetch.jsonl"),
           "PYTHONPATH": str(W / "tools/ctx_own2_vmfetch"), "OMP_NUM_THREADS": "4"}
    done = {j for j, _, _ in JOBS if (ST / f"{j}.done").exists()}
    running: dict[str, subprocess.Popen] = {}
    failed = False
    print(now(), "start; done already:", sorted(done), flush=True)
    while True:
        for j, p in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[j]
            if rc == 0:
                (ST / f"{j}.done").write_text(now() + "\n")
                done.add(j)
                print(now(), "done", j, flush=True)
            else:
                (ST / f"{j}.failed").write_text(f"{now()} rc {rc}\n")
                failed = True
                print(now(), "FAILED", j, "rc", rc, flush=True)
        if not failed:
            for j, args, deps in JOBS:
                if j in done or j in running or not all(d in done for d in deps):
                    continue
                if j == "rank_dry" and not selected():
                    continue
                if len(running) >= MAXJ or (running and mem_free_gb() < MIN_FREE_GB):
                    break
                log = open(LG / f"{j}.log", "w")
                running[j] = subprocess.Popen(TOOL + args, cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(now(), "start", j, "free GB", round(mem_free_gb(), 1), flush=True)
                time.sleep(20)   # let the new job load its tables before the memory check of the next one
        if not running:
            break
        time.sleep(15)
    if failed:
        print(now(), "stopped after a failure; nothing mirrored", flush=True)
        sys.exit(1)
    if "rank_check" in done and not selected():
        print(now(), "no candidate selected; no ranking file", flush=True)
    out = W / "runs/stack_queue/mirror/out.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w") as tf:
        for rel in ["runs/stack_queue/dev", "runs/stack_queue/rank", "runs/stack_queue/logs", "runs/stack_queue/state", "docs/stack_queue",
                    "runs/stack_queue/vm_fetch.jsonl", "runs/stack_queue/remote.log"]:
            if (W / rel).exists():
                tf.add(W / rel, arcname=rel)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.parent / "out.tar.sha256").write_text(h + "\n")
    (ST / "all.done").write_text(now() + "\n")
    print(now(), "ALL DONE", h, flush=True)


if __name__ == "__main__":
    main()
