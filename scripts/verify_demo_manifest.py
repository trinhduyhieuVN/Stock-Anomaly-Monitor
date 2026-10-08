"""Verify the integrity and distribution policy of a prompt-12 demo manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any


REQUIRED_ROLES = {
    "application",
    "environment_lock",
    "feature_config",
    "feature_metadata",
    "feature_split",
    "feature_scalers",
    "baseline_metadata",
    "baseline_models",
    "baseline_thresholds",
    "lstm_metadata",
    "lstm_checkpoint",
    "scoring_metadata",
    "scoring_scores",
    "scoring_thresholds",
    "evaluation_report",
}


class ManifestError(ValueError):
    """Raised when the demo manifest is unsafe or internally inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_member(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ManifestError(f"absolute path is forbidden: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ManifestError(f"path escapes project root: {relative}") from exc
    return resolved


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")

    policy = manifest.get("data_policy", {})
    rights = policy.get("source_rights_status")
    distributable = policy.get("distributable")
    if rights != "confirmed" and distributable is not False:
        raise ManifestError("unconfirmed source rights require distributable=false")
    if policy.get("classification") == "local_only_real_data" and distributable is not False:
        raise ManifestError("local-only real data cannot be distributable")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError("files must be a non-empty list")
    roles = [item.get("role") for item in files]
    missing_roles = sorted(REQUIRED_ROLES.difference(roles))
    if missing_roles:
        raise ManifestError(f"missing required roles: {', '.join(missing_roles)}")
    if len(roles) != len(set(roles)):
        raise ManifestError("file roles must be unique")

    verified: list[dict[str, Any]] = []
    metadata_runs = manifest.get("metadata_runs", {})
    for item in files:
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ManifestError("each file needs string path and sha256")
        path = resolve_member(root, relative)
        if not path.is_file():
            raise ManifestError(f"missing file: {relative}")
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ManifestError(f"sha256 mismatch: {relative}")
        expected_run = metadata_runs.get(item["role"])
        if expected_run:
            actual_run = json.loads(path.read_text(encoding="utf-8")).get("run_id")
            if actual_run != expected_run:
                raise ManifestError(f"run_id mismatch: {relative}")
        verified.append({"role": item["role"], "path": relative, "sha256": actual})

    return {
        "criteria_met": False,
        "check": "prompt12_local_only_manifest_integrity",
        "demo_id": manifest.get("demo_id"),
        "data_policy": policy,
        "verified_file_count": len(verified),
        "verified_files": verified,
        "status": "passed",
        "note": "Partial prompt 12 evidence only; this does not establish a distributable demo or clean environment application run.",
    }


def clean_copy_check(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_copy_parent = root / "reports"
    clean_copy_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prompt12_clean_copy_", dir=clean_copy_parent) as temp_name:
        clean_root = Path(temp_name)
        copied_manifest = clean_root / "demo" / "manifest.local.json"
        copied_manifest.parent.mkdir(parents=True)
        shutil.copy2(manifest_path, copied_manifest)
        for item in manifest["files"]:
            source = resolve_member(root, item["path"])
            destination = clean_root / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        copied_script = clean_root / "scripts" / Path(__file__).name
        copied_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, copied_script)
        completed = subprocess.run(
            [sys.executable, str(copied_script), "--root", str(clean_root), "--manifest", str(copied_manifest)],
            cwd=clean_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ManifestError(f"clean-copy verifier failed: {completed.stderr.strip()}")
        return {
            "performed": True,
            "temporary_copy_removed": True,
            "command": [sys.executable, "scripts/verify_demo_manifest.py", "--root", ".", "--manifest", "demo/manifest.local.json"],
            "exit_code": completed.returncode,
            "verified_file_count": json.loads(completed.stdout)["verified_file_count"],
        }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("demo/manifest.local.json"))
    parser.add_argument("--clean-copy", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    try:
        result = verify_manifest(root, manifest_path)
        if args.clean_copy:
            result["clean_copy"] = clean_copy_check(root, manifest_path)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    if args.report:
        report_path = args.report if args.report.is_absolute() else root / args.report
        write_json_atomic(report_path, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
