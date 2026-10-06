"""Lever 5 (fork E), v67 on Azure: lazy-fetch hook loaded by every Python process of the chain on the VM (PYTHONPATH points here).

When the chain stats, opens or lists a path under the project root (PRC_ROOT) that is not on the VM yet, the file or the directory listing is
fetched from the workstation's read-only file server (tools/ctx_own2_fileserver.py) through the SSH reverse tunnel (PRC_FETCH_URL), written
in place with its SHA-256 checked, and recorded in PRC_FETCH_LOG (one JSON line per file: path, size, sha256). Paths the chain writes are
never fetched (a missing file opened for writing is simply created). Nothing outside the project root is touched.
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
import threading
import urllib.error
import urllib.request

ROOT = os.environ.get("PRC_ROOT")
URL = os.environ.get("PRC_FETCH_URL")
LOG = os.environ.get("PRC_FETCH_LOG")
_lock = threading.RLock()
_seen_dirs: set[str] = set()
_missing: set[str] = set()
_busy = threading.local()
_real_stat, _real_open, _real_listdir, _real_scandir = os.stat, builtins.open, os.listdir, os.scandir


def _rel(path) -> str | None:
    if not ROOT or not URL:
        return None
    try:
        p = os.fspath(path)
    except TypeError:
        return None
    if isinstance(p, bytes) or isinstance(p, int):
        return None
    p = os.path.abspath(p)
    if not p.startswith(ROOT + os.sep):
        return None
    rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
    return rel if rel.split("/")[0] in ("data", "runs", "configs", "docs", "src", "tools") else None


def _get(kind: str, rel: str):
    with urllib.request.urlopen(f"{URL}/{kind}/{urllib.request.quote(rel)}", timeout=600) as r:
        return r.read(), r.headers.get("X-SHA256")


def _fetch_file(rel: str) -> bool:
    if rel in _missing:
        return False
    try:
        data, sha = _get("f", rel)
    except urllib.error.HTTPError:
        _missing.add(rel)
        return False
    if hashlib.sha256(data).hexdigest() != sha:
        raise OSError(f"fetch hash mismatch for {rel}")
    dest = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".fetchtmp"
    with _real_open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    if LOG:
        with _real_open(LOG, "a") as f:
            f.write(json.dumps({"path": rel, "size": len(data), "sha256": sha}) + "\n")
    return True


def _ensure_dir(rel: str) -> bool:
    """Materialise a remote directory's listing: sub-directories are created, files fetched (only when listed by the chain)."""
    if rel in _seen_dirs:
        return True
    try:
        listing = json.loads(_get("l", rel)[0])
    except urllib.error.HTTPError:
        return False
    base = os.path.join(ROOT, rel)
    os.makedirs(base, exist_ok=True)
    for d in listing["dirs"]:
        os.makedirs(os.path.join(base, d), exist_ok=True)
    for name in listing["files"]:
        if not os.path.exists(os.path.join(base, name)):
            _fetch_file(f"{rel}/{name}")
    _seen_dirs.add(rel)
    return True


def _ensure(path, listing: bool = False) -> None:
    if getattr(_busy, "on", False):
        return
    rel = _rel(path)
    if rel is None:
        return
    _busy.on = True
    try:
        with _lock:
            local = os.path.join(ROOT, rel)
            if listing:
                _ensure_dir(rel)
                return
            if os.path.lexists(local):
                return
            try:
                kind = json.loads(_get("s", rel)[0])["type"]
            except Exception:  # noqa: BLE001
                return
            if kind == "file":
                _fetch_file(rel)
            elif kind == "dir":
                os.makedirs(local, exist_ok=True)
    finally:
        _busy.on = False


def _stat(path, *args, **kwargs):
    try:
        return _real_stat(path, *args, **kwargs)
    except FileNotFoundError:
        _ensure(path)
        return _real_stat(path, *args, **kwargs)


def _open(file, mode="r", *args, **kwargs):
    if isinstance(mode, str) and not any(c in mode for c in "wax+"):
        try:
            if not os.path.lexists(file):
                _ensure(file)
        except Exception:  # noqa: BLE001
            pass
    return _real_open(file, mode, *args, **kwargs)


def _listdir(path="."):
    _ensure(path, listing=True)
    return _real_listdir(path)


def _scandir(path="."):
    _ensure(path, listing=True)
    return _real_scandir(path)


if ROOT and URL:
    os.stat = _stat
    builtins.open = _open
    os.listdir = _listdir
    os.scandir = _scandir
    import io
    io.open = _open

    class _ParquetHook:
        """pyarrow opens parquet files in C++: make sure the file is here first."""

        @staticmethod
        def install():
            try:
                import pandas as pd
                import pyarrow.parquet as pq
            except Exception:  # noqa: BLE001
                return
            real_pd, real_table, real_file = pd.read_parquet, pq.read_table, pq.ParquetFile

            def read_parquet(path, *a, **k):
                if isinstance(path, (str, os.PathLike)):
                    _ensure(path)
                return real_pd(path, *a, **k)

            def read_table(source, *a, **k):
                if isinstance(source, (str, os.PathLike)):
                    _ensure(source)
                return real_table(source, *a, **k)

            class ParquetFile(real_file):  # noqa: D401
                def __init__(self, source, *a, **k):
                    if isinstance(source, (str, os.PathLike)):
                        _ensure(source)
                    super().__init__(source, *a, **k)

            pd.read_parquet, pq.read_table, pq.ParquetFile = read_parquet, read_table, ParquetFile

    _ParquetHook.install()
