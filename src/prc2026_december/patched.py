"""Load the frozen feature modules with their December guard lifted, without touching the frozen files."""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARDS = {  # package -> the single guard statement that refuses December (frozen sources, unchanged on disk)
    "prc2026_phase4": 'raise ValueError("Reserved audit features cannot be materialized in Phase 4.")',
    "prc2026_phase5r": 'raise ValueError("Reserved audit context cannot enter this campaign.")',
    "prc2026_weather": 'raise ValueError("Reserved audit or undeclared months cannot enter weather features.")',
    "prc2026_nmclock": 'raise ValueError("Reserved audit or undeclared movement period.")',
    "prc2026_timeline": 'raise ValueError("Reserved or undeclared movement period.")',
}
REPLACEMENT = "pass  # December opened on 2026-09-19 (docs/december/authorization_20260919.json); guard lifted in this loaded copy only"
_cache: dict[str, types.ModuleType] = {}


def source_hashes() -> dict[str, str]:
    return {f"src/{package}/features.py": hashlib.sha256((ROOT / "src" / package / "features.py").read_bytes()).hexdigest() for package in GUARDS}


def load(package: str) -> types.ModuleType:
    """The package's features module executed from its source with the guard replaced; siblings resolve to the frozen packages."""
    if package in _cache:
        return _cache[package]
    path = ROOT / "src" / package / "features.py"
    source = path.read_text(encoding="utf-8")
    guard = GUARDS[package]
    if source.count(guard) != 1:
        raise ValueError(f"Expected exactly one December guard in {path}.")
    name = f"{package}.features_december"
    module = types.ModuleType(name)
    module.__package__ = package
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source.replace(guard, REPLACEMENT), f"{path} [december]", "exec"), module.__dict__)
    _cache[package] = module
    return module
