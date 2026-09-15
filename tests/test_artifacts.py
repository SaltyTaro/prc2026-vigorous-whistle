import pytest

from prc2026.artifacts import environment, sha256_file, verify_receipts


def test_environment_report_accepts_path_objects_on_windows_and_linux(tmp_path):
    report = environment(tmp_path)
    assert report["logical_cpus"] >= 1
    assert report["memory_total_bytes"] > 0
    assert report["disk_available_bytes"] > 0


def test_same_size_corruption_is_detected(tmp_path):
    path = tmp_path / "feature.parquet"
    path.write_bytes(b"original")
    receipt = {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    verify_receipts(tmp_path, [receipt])
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_receipts(tmp_path, [receipt])


def test_artifact_paths_cannot_escape_bundle(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        verify_receipts(tmp_path, [{"path": "../outside", "size_bytes": 0, "sha256": ""}])
