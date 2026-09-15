"""Match every current test identity to passing Windows and Azure records."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json


def main():
    output = ROOT / "docs/next_lever/tests_combined.json"
    if output.exists():
        raise ValueError("Preserve the existing combined test receipt.")
    command = [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", "-o", "addopts="]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    required = {line.strip() for line in result.stdout.splitlines() if line.startswith("tests/") and "::" in line}
    if not required:
        raise ValueError("No test identities were collected.")
    passed, sources = {"windows": {}, "azure": {}}, {}
    for path in sorted((ROOT / "docs").rglob("*.xml")):
        try:
            tree = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for suite in tree.iter("testsuite"):
            host = suite.attrib.get("hostname", "").lower()
            platform = "azure" if host == "arcw0" else "windows" if host == "" else None
            if platform is None:
                continue
            for case in suite.findall("testcase"):
                identity = case.attrib.get("classname", "").replace(".", "/") + ".py::" + case.attrib.get("name", "")
                if identity not in required or any(case.find(tag) is not None for tag in ["failure", "error", "skipped"]):
                    continue
                relative = path.relative_to(ROOT).as_posix()
                # Prefer canonical reports over copies in immutable mirror snapshots.
                old = passed[platform].get(identity)
                if old is None or (len(relative), relative) < (len(old), old):
                    passed[platform][identity] = relative
    missing = {platform: sorted(required - set(items)) for platform, items in passed.items()}
    if any(missing.values()):
        raise ValueError("Current tests lack passing platform evidence: " + repr(missing))
    for relative in sorted({name for items in passed.values() for name in items.values()}):
        sources[relative] = sha256_file(ROOT / relative)
    report = {"created_at_utc": utc_now(), "passed": True, "distinct_current_tests": len(required),
              "current_test_identities": sorted(required), "platform_test_evidence": passed,
              "evidence_sha256": sources, "test_source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted((ROOT / "tests").rglob("*.py"))},
              "collection_command": [".venv Python", *command[1:]], "code_sha256": sha256_file(Path(__file__)),
              "scope": "Matching passing test identities across recorded runs on both platforms. No competition training or new development score."}
    write_json(output, report)
    print({"passed": True, "distinct_current_tests": len(required), "platform_counts": {key: len(value) for key, value in passed.items()}, "evidence_files": len(sources)})


if __name__ == "__main__":
    main()
