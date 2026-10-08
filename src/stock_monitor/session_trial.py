"""Connect verified delay observations, immutable history and frozen live scoring."""
from dataclasses import replace
import json
from pathlib import Path
import time
import uuid

import pandas as pd

from .config import load_config
from .io import DataContractError, sha256
from .live_trial import execute_vci_one_cycle, preflight_live_trial
from .pipeline import load_dataset_bundle
from .realtime import closed_bar_readiness, score_ready_closed_bars, verify_frozen_scoring_manifest
from .session_sampling import session_gate, utc_now, verified_delay_basis
from .validation import load_profile


def trial_history(project_root, manifest_path, expected_sha256):
    root = Path(project_root).resolve()
    path = (root / manifest_path).resolve()
    if not path.is_relative_to(root / "reports/trials") or sha256(path) != expected_sha256:
        raise DataContractError("History manifest path/checksum mismatch")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if (bundle.get("kind") != "local_research_trial_history" or bundle.get("source_rights_status") != "pending"
            or bundle.get("distributable") is not False or len(bundle.get("symbol_scope", [])) != 1):
        raise DataContractError("Expected one-symbol local-only research history")
    pins = {item["path"]: item["sha256"] for item in bundle["files"] + bundle["inputs"]}
    if bundle["config_path"] not in pins or bundle["history_path"] not in pins:
        raise DataContractError("History/config must be hash-pinned in the manifest")
    for relative, digest in pins.items():
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root) or sha256(artifact) != digest:
            raise DataContractError("Trial history artifact/input checksum mismatch")
    from .providers import load_snapshot
    load_snapshot(root / bundle["inputs"][0]["path"], root)
    frozen = verify_frozen_scoring_manifest(root, "demo/manifest.local.json")
    manifest = json.loads((root / frozen["manifest_path"]).read_text(encoding="utf-8"))
    pin = next(p for p in manifest["files"] if p["role"] == "feature_metadata")
    if sha256(root / pin["path"]) != pin["sha256"]:
        raise DataContractError("Feature metadata checksum mismatch")
    feature = json.loads((root / pin["path"]).read_text(encoding="utf-8"))
    original, metadata, _ = load_dataset_bundle(load_config(root / "configs/research.yaml"), feature["input_dataset_dir"])
    if metadata["validation"]["dataset_sha256"] != feature["input_dataset_sha256"]:
        raise DataContractError("History differs from frozen feature provenance")
    symbol = bundle["symbol_scope"][0]
    extension = pd.read_parquet(root / bundle["history_path"])
    combined = pd.concat([original.loc[original.symbol.eq(symbol)], extension], ignore_index=True)
    if not combined.symbol.eq(symbol).all() or combined.duplicated(["symbol", "timestamp"]).any():
        raise DataContractError("Trial history scope or duplicate keys are invalid")
    profile = load_profile(load_config(root / bundle["config_path"]))
    return bundle, profile, combined


def run_measured_trial(project_root, history_manifest, history_sha256, delay_path, delay_sha256, *,
                       transport, clock=utc_now, sleep=time.sleep):
    root = Path(project_root).resolve()
    bundle, profile, historical = trial_history(root, history_manifest, history_sha256)
    observed = clock()
    state, day = session_gate(profile, observed)
    if state != "SESSION_OPEN":
        raise DataContractError(state)
    symbol = bundle["symbol_scope"][0]
    evidence = verified_delay_basis(root, delay_path, delay_sha256, config_path=bundle["config_path"],
                                    symbol=symbol, now=observed)
    preflight = preflight_live_trial(root, config_path=bundle["config_path"], observed_at_utc=observed,
        provider_delay_seconds=evidence["empirical_delay_seconds"], provider_delay_evidence=f"{delay_path} sha256={delay_sha256}",
        max_polls=1, max_attempts=1, max_wait_seconds=0)
    preflight["symbols"] = [symbol]
    cycle_id = "live_" + uuid.uuid4().hex[:24]
    cutoff = pd.Timestamp(day, tz="Asia/Ho_Chi_Minh").tz_convert("UTC")
    if historical.timestamp.ge(cutoff).any():
        raise DataContractError("Trial historical bars must precede the current session")
    def score(readiness, received):
        scored = score_ready_closed_bars(readiness, historical, profile, project_root=root, received_at_utc=received)
        timing = dict(scored.timing)
        start, end = pd.Timestamp(timing["score_started_at_utc"]), pd.Timestamp(timing["score_finished_at_utc"])
        if not pd.Timestamp(received) <= start <= end:
            raise DataContractError("Live receipt/score clock order is invalid")
        timing.update(receive_to_score_start_ms=(start - pd.Timestamp(received)).total_seconds() * 1000,
                      receive_to_score_end_ms=(end - pd.Timestamp(received)).total_seconds() * 1000,
                      latency_classification="actual_receipt_to_process_score_candidate")
        return replace(scored, timing=timing)
    result = execute_vci_one_cycle(preflight, store_root=root / "data/raw/live_trial" / cycle_id,
        cycle_id=cycle_id, profile=profile, project_root=root, transport=transport, sleep=sleep,
        readiness_fn=closed_bar_readiness, score_fn=score, timeout_seconds=10, clock=clock,
        delay_observations={"path": str(delay_path), "sha256": delay_sha256})
    readiness = result.readiness.status if result.readiness else {}
    timing = result.score_result.timing if result.score_result else {}
    report = {"schema_version": 1, "kind": "measured_session_trial", "state": result.state,
              "criteria_met": False, "live_evidence": result.live_evidence, "preflight": preflight,
              "delay_evidence": {"path": str(delay_path), "sha256": delay_sha256, "analysis": evidence},
              "history_evidence": {"path": str(history_manifest), "sha256": history_sha256},
              "attempts": list(result.attempts), "waited_seconds": result.waited_seconds,
              "readiness": readiness, "timing": timing, "source_rights_status": "pending",
              "cycle_id": cycle_id,
              "store_reports": [s.report for s in result.store_results],
              "scores": [] if result.score_result is None else json.loads(result.score_result.scores.to_json(orient="records", date_format="iso"))}
    operator = {"schema_version": 1, "kind": "operator_status_snapshot", "state": result.state,
                "live_evidence": result.live_evidence, "source_rights_status": "pending",
                "latest_closed_bar_utc": readiness.get("latest_bar_end_utc"),
                "gap_count": readiness.get("gap_count", 0), "stale": readiness.get("stale", False),
                "received_at_utc": timing.get("received_at_utc"), "score_duration_ms": timing.get("score_duration_ms"),
                "calendar_status": readiness.get("calendar_status"),
                "artifact_status": "verified" if result.readiness else "not_verified_after_poll",
                "provider_delay_seconds": evidence["empirical_delay_seconds"],
                "operator_message": (
                    f"Độ trễ nguồn thực nghiệm: {evidence['empirical_delay_seconds']} giây (mẫu hữu hạn). "
                    f"Nhận → bắt đầu score: {timing.get('receive_to_score_start_ms', 'chưa đo')} ms; "
                    f"nhận → kết thúc score: {timing.get('receive_to_score_end_ms', 'chưa đo')} ms. "
                    f"Snapshot một chu kỳ {symbol}, quan sát lúc {pd.Timestamp(observed).isoformat()}.")}
    return report, operator
