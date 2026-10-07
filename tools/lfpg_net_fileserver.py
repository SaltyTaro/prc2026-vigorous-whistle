"""Lever 12b (fork N; a copy of tools/ctx_own2_fileserver.py with its own log): read-only file server for the project tree, bound to 127.0.0.1 in WSL and reached by an Azure VM only
through an SSH reverse tunnel (ssh -R). The VM's lazy-fetch hook (tools/ctx_own2_vmfetch/sitecustomize.py) asks for exactly the files the
chain opens, so only the chain's real inputs leave the workstation.

  GET /f/<relpath>   the file's bytes (header X-SHA256 with its hash)
  GET /l/<relpath>   JSON {"dirs": [...], "files": [...]} of a directory (non-recursive)
  GET /s/<relpath>   JSON {"type": "file" | "dir" | "missing"}

Only paths under data/, runs/, configs/, docs/, src/, tools/ are served; .env, .git, infra/, any path with '..' and any name starting with
'.env' are refused. Every request is logged to runs/lfpg_net/fileserver.log. Usage (WSL, project root):
  python3 tools/ctx_own2_fileserver.py --port 8765
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = ("data/", "runs/", "configs/", "docs/", "src/", "tools/")
LOG = ROOT / "runs/lfpg_net/fileserver.log"


def safe(rel: str) -> Path | None:
    rel = unquote(rel).lstrip("/")
    parts = rel.split("/")
    if not rel.startswith(ALLOWED) or ".." in parts or any(p.startswith(".env") or p == ".git" for p in parts):
        return None
    return ROOT / rel


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: N802
        with open(LOG, "a") as f:
            f.write(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {fmt % args}\n")

    def send_json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        kind, _, rel = self.path.partition("/")[2].partition("/")
        p = safe(rel)
        if p is None:
            self.send_error(403)
            return
        if kind == "s":
            self.send_json({"type": "file" if p.is_file() else "dir" if p.is_dir() else "missing"})
        elif kind == "l":
            if not p.is_dir():
                self.send_error(404)
                return
            entries = list(os.scandir(p))
            self.send_json({"dirs": sorted(e.name for e in entries if e.is_dir() and not e.name.startswith(".env")),
                            "files": sorted(e.name for e in entries if e.is_file() and not e.name.startswith(".env"))})
        elif kind == "f":
            if not p.is_file():
                self.send_error(404)
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-SHA256", hashlib.sha256(data).hexdigest())
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
