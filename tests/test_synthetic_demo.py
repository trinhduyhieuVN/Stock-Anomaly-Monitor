import csv
import json
from pathlib import Path

import pytest

from scripts.run_synthetic_demo import SyntheticDemoError, run, validate_fixture


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_synthetic_bundle_passes() -> None:
    result = run(ROOT, ROOT / "demo/manifest.synthetic.json")
    assert result["status"] == "passed"
    assert result["criteria_met"] is False
    assert result["fixture"]["rows"] == 18
    assert result["fixture"]["case_counts"] == {"baseline": 15, "seeded_event": 3}


def test_manifest_has_no_real_or_local_artifact_paths() -> None:
    manifest = json.loads((ROOT / "demo/manifest.synthetic.json").read_text(encoding="utf-8"))
    assert manifest["data_policy"]["contains_real_source_data"] is False
    assert manifest["data_policy"]["distributable"] is True
    paths = [item["path"].replace("\\", "/").lower() for item in manifest["files"]]
    assert all(not path.startswith(("artifacts/", "data/raw/", "data/processed/")) for path in paths)


def test_fixture_rejects_real_source_marker(tmp_path: Path) -> None:
    source = ROOT / "demo/synthetic_ohlcv.csv"
    rows = list(csv.DictReader(source.open("r", encoding="utf-8", newline="")))
    rows[0]["source"] = "VCI"
    changed = tmp_path / "changed.csv"
    with changed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = json.loads((ROOT / "demo/manifest.synthetic.json").read_text(encoding="utf-8"))
    with pytest.raises(SyntheticDemoError, match="source must be SYNTHETIC"):
        validate_fixture(changed, manifest)
