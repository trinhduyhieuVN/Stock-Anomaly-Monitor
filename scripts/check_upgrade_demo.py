"""Portable synthetic bundle check; independent of local real artifacts."""
import json
from pathlib import Path

from stock_monitor.upgrade.ui_service import load_ui_bundle


def main():
    bundle = load_ui_bundle(Path(__file__).resolve().parents[1])
    assert bundle["manifest"]["contains_real_data"] is False
    print(json.dumps({"status": "PASS", "mode": "SYNTHETIC_NOT_LIVE", "manifest_sha256": bundle["key"],
        "sessions": bundle["sessions"], "observations": len(bundle["rows"]), "news": len(bundle["news"]),
        "real_artifacts_required": False, "provider_calls": 0, "training_invoked": False}))


if __name__ == "__main__":
    main()
