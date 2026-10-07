"""Lever 9 (fork J; a copy of tools/ctx_opdi_dispatch.py for clusters prc-ctx5a-l): dispatch the fits of a ctx_opdi stage over whatever L40S nodes (on-demand or spot) are up, mirror every finished fit at
once with a hash check, and re-run from scratch any fit whose node was pre-empted. Runs in WSL from the project root with the sky venv's python
(ssh aliases and `sky` from ~/.sky). Never touches clusters other than prc-ctx5a-l.

Jobs: runs/ctx_opdi/jobs_<stage>/all.txt (tools/ctx_opdi_jobs.py; lines: runner config variant seed fold output extra_features_dir).
State: runs/ctx_opdi/dispatch_<stage>.json; events (launches, mirrors, failures, pre-emptions, node preparation): runs/ctx_opdi/dispatch_<stage>.log.
Each cycle (every --interval seconds):
  1. nodes: reachable by ssh? ready (~/prc/.opdi_ready, set after the token / own-table / test checks)? A reachable node that is not ready and
     has finished its sky setup is prepared in the background: `check` if it already holds the token build (a head), else `fanout` from a
     ready node. A node holding running fits that is unreachable for --lost-cycles cycles while `sky status` does not show it UP is counted as
     pre-empted: its fits go back to pending (attempt not charged), it is taken down and relaunched with the capacity retry (spot nodes only;
     on-demand nodes are reported).
  2. running fits on reachable nodes: run.json present -> mirror the fit directory (rsync), compare every file's SHA-256 with the node's ->
     done (with hashes); .failed -> failed attempt (re-queued until --max-attempts); not started 15 min after dispatch -> re-queued.
  3. pending fits go to free lanes (--lanes per ready node) via `sky exec <node> --gpus L40S:<1/lanes> -d tools/ctx_opdi_run1.sh ...`.
Stops when every fit is done (or failed for good). Partial fits are never mirrored or counted.
Usage: ~/.venv-sky/bin/python tools/ctx_opdi_dispatch.py --stage design [--interval 60] [--lanes 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKY = str(Path.home() / ".venv-sky/bin/sky")
ALL_NODES = [f"prc-ctx5{c}" for c in "abcdefghijkl"]
SPOT = set(f"prc-ctx5{c}" for c in "ijkl")
HEADS = {"prc-ctx5a", "prc-ctx5e", "prc-ctx5g", "prc-ctx5i", "prc-ctx5k"}  # launched with a head yaml; all others get a fanout


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def ssh(node: str, command: str, timeout: int = 40) -> tuple[int, str]:
    return run(["ssh", "-o", "ConnectTimeout=12", "-o", "BatchMode=yes", node, command], timeout=timeout)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Dispatcher:
    def __init__(self, args):
        self.args = args
        self.base = ROOT / "runs/ctx_opdi"
        self.state_path = self.base / f"dispatch_{args.stage}.json"
        self.log_path = self.base / f"dispatch_{args.stage}.log"
        lines = [l.split() for l in (self.base / f"jobs_{args.stage}/all.txt").read_text().splitlines() if l.strip()]
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"jobs": {}, "nodes": {}, "preemptions": []}
        for f in lines:
            tag = f"{Path(f[5]).name}_{f[4]}"
            self.state["jobs"].setdefault(tag, {"fields": f, "status": "pending", "node": None, "attempts": 0, "preempted": 0})
        self.preparing: dict[str, subprocess.Popen] = {}
        self.relaunching: dict[str, subprocess.Popen] = {}
        self.unreachable: dict[str, int] = {}
        self.synced: set[str] = set()

    def event(self, text: str) -> None:
        line = f"{now()} {text}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_path)

    def sky_up(self, node: str) -> bool:
        code, out = run([SKY, "status", node], timeout=120)
        return code == 0 and any(l.split()[:1] == [node] and " UP " in f" {l} " for l in out.splitlines())

    def nodes(self) -> tuple[list[str], set[str]]:
        ready, reachable, waiting = [], set(), {}
        for n in ALL_NODES:
            if not (Path.home() / ".sky/generated/ssh" / n).exists():
                continue
            code, out = ssh(n, "test -f ~/prc/.opdi_ready && echo READY; "
                               "ls -d ~/prc/.venv/lib/python3*/site-packages/catboost ~/prc/.venv/lib/python3*/site-packages/torch > /dev/null 2>&1 && echo PKGS; "
                               "test -f ~/prc/data/ctx/tokens_v1/manifest.json && echo TOKENS; true")  # exit 0 = reachable, whatever the tests
            if code != 0:
                self.unreachable[n] = self.unreachable.get(n, 0) + 1
                continue
            self.unreachable[n] = 0
            reachable.add(n)
            if "READY" in out:
                ready.append(n)
            elif "PKGS" in out and n not in self.preparing:
                waiting[n] = "TOKENS" in out
        for n, has_tokens in waiting.items():
            self.prepare(n, has_tokens, ready)
        return ready, reachable

    def prepare(self, node: str, has_tokens: bool, ready: list[str]) -> None:
        """A head checks its own upload and token build (once its setup has built the tokens); a worker copies the data from a ready node."""
        env = f"NODES={node}"
        if node in HEADS:
            if not has_tokens:
                return  # its sky setup is still building the tokens
            cmd = f"{env} bash tools/ctx_opdi_nodes.sh sync && {env} bash tools/ctx_opdi_nodes.sh check"
        else:
            sources = [r for r in ready if r != node]
            if not sources:
                return
            cmd = f"NODES='{sources[0]} {node}' bash tools/ctx_opdi_nodes.sh sync && NODES='{sources[0]} {node}' FROM={sources[0]} bash tools/ctx_opdi_nodes.sh fanout"
        log = open(self.base / f"logs/prepare_{node}_{datetime.now(timezone.utc).strftime('%H%M%S')}.log", "w")
        self.preparing[node] = subprocess.Popen(["bash", "-c", cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        self.event(f"prepare {node} ({'check' if node in HEADS else 'fanout'})")

    def reap(self) -> None:
        for d in (self.preparing, self.relaunching):
            for n, p in list(d.items()):
                if p.poll() is not None:
                    self.event(f"{'prepared' if d is self.preparing else 'relaunch ended'} {n} exit {p.returncode}")
                    del d[n]

    def check_running(self, reachable: set[str]) -> None:
        by_node: dict[str, list[str]] = {}
        for tag, j in self.state["jobs"].items():
            if j["status"] == "running":
                by_node.setdefault(j["node"], []).append(tag)
        for node, tags in by_node.items():
            if node not in reachable:
                if self.unreachable.get(node, 0) >= self.args.lost_cycles and not self.sky_up(node):
                    for tag in tags:
                        j = self.state["jobs"][tag]
                        j.update(status="pending", node=None, preempted=j["preempted"] + 1)
                    self.state["preemptions"].append({"node": node, "at": now(), "fits_requeued": tags})
                    self.event(f"PRE-EMPTED {node}: re-queued {tags}")
                    self.relaunch(node)
                continue
            checks = "; ".join(
                f"o={shlex.quote(self.state['jobs'][t]['fields'][5] + '/' + self.state['jobs'][t]['fields'][4])}; "
                f"if [ -f $o/run.json ]; then echo {t} DONE; elif [ -f $o.failed ]; then echo {t} FAILED; "
                f"elif [ -f $o.started ]; then echo {t} RUNNING; else echo {t} NOTSTARTED; fi" for t in tags)
            code, out = ssh(node, f"cd ~/prc; {checks}")
            if code != 0:
                continue
            for line in out.splitlines():
                parts = line.split()
                if len(parts) != 2 or parts[0] not in self.state["jobs"]:
                    continue
                tag, status = parts
                j = self.state["jobs"][tag]
                if status == "DONE":
                    self.mirror(tag, j, node)
                elif status == "FAILED":
                    j["attempts"] += 1
                    j.update(status="pending" if j["attempts"] < self.args.max_attempts else "failed", node=None)
                    self.event(f"FAILED {tag} on {node} (attempt {j['attempts']}) -> {j['status']}")
                elif status == "NOTSTARTED" and time.time() - j.get("dispatched_ts", time.time()) > 900:
                    j.update(status="pending", node=None)
                    self.event(f"not started 15 min after dispatch: {tag} on {node} -> pending")

    def mirror(self, tag: str, j: dict, node: str) -> None:
        rel = f"{j['fields'][5]}/{j['fields'][4]}"
        local = ROOT / rel
        local.mkdir(parents=True, exist_ok=True)
        code, out = run(["rsync", "-a", "-e", "ssh", f"{node}:prc/{rel}/", f"{local}/"], timeout=600)
        if code != 0:
            self.event(f"mirror failed {tag} from {node}: {out[-200:]}")
            return
        code, out = ssh(node, f"cd ~/prc/{shlex.quote(rel)} && sha256sum *", timeout=120)
        if code != 0:
            return
        remote = {l.split()[1].lstrip("*"): l.split()[0] for l in out.splitlines() if len(l.split()) == 2}
        mine = {p.name: sha256(p) for p in local.iterdir() if p.is_file()}
        if not remote or remote != mine or "run.json" not in mine:
            self.event(f"hash mismatch {tag} from {node}; will retry the mirror")
            return
        j.update(status="done", node_done=node, finished=now(), files=mine)
        self.event(f"done+mirrored {tag} from {node} ({len(mine)} files, hashes equal)")

    def relaunch(self, node: str) -> None:
        if node not in SPOT or node in self.relaunching:
            self.event(f"{node} lost; not relaunched automatically (on-demand)" if node not in SPOT else f"{node} relaunch already running")
            return
        run([SKY, "down", node, "-y"], timeout=600)
        log = open(self.base / f"logs/relaunch_{node}_{datetime.now(timezone.utc).strftime('%H%M%S')}.log", "w")
        self.relaunching[node] = subprocess.Popen(["bash", "-c", f"NODES={node} RETRY_MIN=60 bash tools/ctx_opdi_nodes.sh up"], cwd=ROOT,
                                                  stdout=log, stderr=subprocess.STDOUT)
        self.event(f"relaunching spot node {node}")

    def dispatch(self, ready: list[str]) -> None:
        load = {n: 0 for n in ready}
        for j in self.state["jobs"].values():
            if j["status"] == "running" and j["node"] in load:
                load[j["node"]] += 1
        pending = [t for t, j in self.state["jobs"].items() if j["status"] == "pending"]
        for node in ready:
            if pending and node not in self.synced:  # this run's code and configs reach every node before its first fit
                code, out = run(["bash", "-c", f"NODES={node} bash tools/ctx_opdi_nodes.sh sync"], timeout=600)
                if code != 0 or "synced" not in out:
                    self.event(f"sync failed for {node}: {out[-200:]}")
                    continue
                self.synced.add(node)
                self.event(f"synced code and configs to {node}")
            while load[node] < self.args.lanes and pending:
                tag = pending.pop(0)
                j = self.state["jobs"][tag]
                args = " ".join(shlex.quote(x) for x in j["fields"])
                code, out = run([SKY, "exec", node, "--gpus", f"L40S:{1 / self.args.lanes:g}", "-d", "--name", f"opdi-{tag}"[:60],
                                 f"cd ~/prc && bash tools/ctx_opdi_run1.sh {args}"], timeout=300)
                if code != 0:
                    self.event(f"sky exec failed for {tag} on {node}: {out[-300:]}")
                    pending.insert(0, tag)
                    break
                j.update(status="running", node=node, dispatched=now(), dispatched_ts=time.time())
                load[node] += 1
                self.event(f"dispatched {tag} -> {node}")

    def loop(self) -> None:
        self.event(f"dispatcher start: {len(self.state['jobs'])} fits")
        while True:
            self.reap()
            ready, reachable = self.nodes()
            self.check_running(reachable)
            self.dispatch(ready)
            self.save()
            counts = {s: sum(j["status"] == s for j in self.state["jobs"].values()) for s in ("pending", "running", "done", "failed")}
            self.state["summary"] = {"at": now(), "ready_nodes": ready, **counts}
            self.save()
            if counts["pending"] == 0 and counts["running"] == 0:
                self.event(f"all fits finished: {counts}")
                return
            time.sleep(self.args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["design", "dev"], required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--lost-cycles", type=int, default=3)
    args = parser.parse_args()
    lock = ROOT / f"runs/ctx_opdi/dispatch_{args.stage}.pid"
    import os
    if lock.exists():
        pid = int(lock.read_text().strip() or 0)
        if pid and Path(f"/proc/{pid}").exists():
            raise SystemExit(f"dispatcher already running (pid {pid})")
    lock.write_text(str(os.getpid()))
    try:
        Dispatcher(args).loop()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
