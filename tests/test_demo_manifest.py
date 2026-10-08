import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_demo_manifest import ManifestError, verify_manifest


def _manifest(path: str, digest: str, distributable: bool = False) -> dict:
    roles = sorted({
        "application", "environment_lock", "feature_config", "feature_metadata",
        "feature_split", "feature_scalers", "baseline_metadata", "baseline_models",
        "baseline_thresholds", "lstm_metadata", "lstm_checkpoint", "scoring_metadata",
        "scoring_scores", "scoring_thresholds", "evaluation_report",
    })
    return {
        "schema_version": 1,
        "demo_id": "test",
        "data_policy": {
            "classification": "local_only_real_data",
            "source_rights_status": "pending",
            "distributable": distributable,
        },
        "files": [{"role": role, "path": path, "sha256": digest} for role in roles],
    }


def test_manifest_verifies_hashes_and_local_only_policy(tmp_path: Path) -> None:
    payload = b"demo"
    (tmp_path / "file.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest("file.bin", digest)), encoding="utf-8")
    result = verify_manifest(tmp_path, manifest_path)
    assert result["status"] == "passed"
    assert result["criteria_met"] is False


def test_manifest_rejects_distribution_with_pending_rights(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest("missing", "0" * 64, True)), encoding="utf-8")
    with pytest.raises(ManifestError, match="distributable=false"):
        verify_manifest(tmp_path, manifest_path)


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest("../outside", "0" * 64)), encoding="utf-8")
    with pytest.raises(ManifestError, match="escapes project root"):
        verify_manifest(tmp_path, manifest_path)
