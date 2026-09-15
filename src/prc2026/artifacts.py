"""Small, explicit helpers for provenance and process resource measurements."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def versions() -> dict:
    return {name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "psutil", "tzdata"]}


def code_receipt(root: Path) -> dict:
    paths = [*sorted((root / "src" / "prc2026").glob("*.py")),
             root / "tools" / "run_phase1.py", root / "pyproject.toml",
             root / "requirements-phase1.txt"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def verify_receipts(base: Path, receipts: list[dict]) -> None:
    """Reject missing, stale, truncated, or escaping artifact paths."""
    resolved_base = base.resolve()
    for receipt in receipts:
        path = (base / receipt["path"]).resolve()
        if not path.is_relative_to(resolved_base):
            raise ValueError("Artifact path escapes its bundle.")
        if not path.is_file() or path.stat().st_size != receipt["size_bytes"]:
            raise ValueError(f"Missing or wrong-size artifact: {receipt['path']}")
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {receipt['path']}")


def environment(root: Path) -> dict:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus_reported": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "disk_available_bytes": psutil.disk_usage(str(root)).free, "packages": versions(),
    }


class ResourceMonitor:
    """Sample this process every 50 ms; includes native-library thread CPU time."""

    def __enter__(self):
        self.process = psutil.Process()
        self.start_wall = time.perf_counter()
        cpu = self.process.cpu_times()
        self.start_cpu = cpu.user + cpu.system
        self.peak_rss = self.process.memory_info().rss
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(0.05):
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        wall = time.perf_counter() - self.start_wall
        cpu = self.process.cpu_times()
        cpu_seconds = cpu.user + cpu.system - self.start_cpu
        self.stats = {
            "wall_seconds": wall, "cpu_seconds": cpu_seconds,
            "mean_active_cpu_cores": cpu_seconds / wall,
            "peak_process_rss_bytes_sampled": self.peak_rss,
            "rss_sample_interval_seconds": 0.05,
        }
