"""Read-only P02 historical adapter; saved v1 scores, never synthetic calibration."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from scripts.verify_demo_manifest import resolve_member, verify_manifest
from stock_monitor.io import BAR_COLUMNS, DataContractError, resolve_project_reference, sha256
from stock_monitor.services import checked_metadata, dashboard_artifact_key, filtered_scores, load_monitor_data, MODELS
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.upgrade.ui_service import event_csv

KEY_COLUMNS = ["symbol", "timestamp"]
OHLCV = ["open", "high", "low", "close", "volume"]
AVAILABILITY_BASIS = "retrospective_bar_end_assumption_no_per_bar_receipts"


def verify_history_binding(root, manifest_name, expected_manifest_sha256):
    """No arbitrary uploaded model or changed manifest may replace the pinned run."""
    root = Path(root).resolve()
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise DataContractError("Historical adapter requires an explicit pinned manifest SHA256")
    path = resolve_member(root, manifest_name)
    if not path.is_file() or sha256(path) != expected_manifest_sha256:
        raise DataContractError("Historical manifest missing or checksum mismatch")
    verified = verify_manifest(root, path)
    policy = verified["data_policy"]
    if policy.get("classification") != "local_only_real_data" or policy.get("source_rights_status") != "pending" or policy.get("distributable") is not False:
        raise DataContractError("Historical source must remain local-only with pending rights")
    roles = {row["role"]: resolve_member(root, row["path"]) for row in verified["verified_files"]}
    baseline_directory, scoring_directory = roles["baseline_metadata"].parent, roles["scoring_metadata"].parent
    baseline = checked_metadata(baseline_directory, "baseline_bundle", ["scores.parquet"])
    feature_directory = resolve_project_reference(baseline["input_feature_dir"], root)
    if feature_directory != roles["feature_metadata"].parent:
        raise DataContractError("Historical manifest and baseline feature directories differ")
    feature = checked_metadata(feature_directory, "feature_window_bundle", ["features.parquet"])
    scoring = checked_metadata(scoring_directory, "lstm_scoring_bundle", ["scores.parquet"])
    if (resolve_project_reference(scoring["input_lstm_dir"], root) != roles["lstm_metadata"].parent or
            resolve_project_reference(scoring["input_feature_dir"], root) != feature_directory):
        raise DataContractError("Historical scoring paths differ from pinned manifest")
    dataset_directory = resolve_project_reference(feature["input_dataset_dir"], root)
    if not dataset_directory.is_relative_to(root):
        raise DataContractError("Historical dataset escapes project")
    dataset = checked_metadata(dataset_directory, None, ["bars.parquet"])
    if dataset.get("timestamp_convention") != "bar_end" or dataset.get("interval") != "5m" or dataset.get("timestamp_timezone") != "UTC":
        raise DataContractError("Historical adapter requires UTC five-minute bar-end data")
    confirmations = feature.get("source_confirmations", {})
    if len(confirmations) != 6 or any(row.get("status") != "pending" for row in confirmations.values()):
        raise DataContractError("Historical source confirmations must remain pending")
    service_key = dashboard_artifact_key(baseline_directory, scoring_directory)
    # Bind the transitive files checked by v1, not just the 15 demo manifest roles.
    pins = {row["path"]: row["sha256"] for row in verified["verified_files"]}
    pins[manifest_name] = expected_manifest_sha256
    for directory in (baseline_directory, scoring_directory, roles["lstm_metadata"].parent, feature_directory, dataset_directory):
        if not directory.is_relative_to(root):
            raise DataContractError("Historical artifact directory escapes project")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        for name in ["metadata.json", *metadata["artifact_sha256"]]:
            member = resolve_member(root, (directory.relative_to(root) / name).as_posix())
            pins[member.relative_to(root).as_posix()] = sha256(member)
    key = hashlib.sha256(json.dumps({"source": "HISTORICAL_REPLAY", "manifest": expected_manifest_sha256,
                                    "v1": service_key, "pins": pins}, sort_keys=True).encode()).hexdigest()
    return {"root": root, "manifest_name": manifest_name, "manifest_sha256": expected_manifest_sha256,
            "key": key, "pins": pins, "baseline_directory": baseline_directory,
            "scoring_directory": scoring_directory, "dataset_directory": dataset_directory,
            "feature_directory": feature_directory, "data_policy": deepcopy(policy), "dataset_metadata": dataset}


def _verify_loaded(bundle):
    for name, digest in bundle["pins"].items():
        path = resolve_member(bundle["root"], name)
        if not path.is_file() or sha256(path) != digest:
            raise DataContractError("Historical artifact missing or checksum mismatch: " + name)


def load_historical_bundle(root, *, expected_manifest_sha256, manifest_name="demo/manifest.local.json"):
    binding = verify_history_binding(root, manifest_name, expected_manifest_sha256)
    frame, metadata, _ = load_monitor_data(binding["baseline_directory"], binding["scoring_directory"])
    bars = pd.read_parquet(binding["dataset_directory"] / "bars.parquet")
    if bars.duplicated(KEY_COLUMNS).any() or frame.duplicated(KEY_COLUMNS).any():
        raise DataContractError("Historical bars must have unique symbol/timestamp keys")
    if set(map(tuple, frame[KEY_COLUMNS].to_numpy())) != set(map(tuple, bars[KEY_COLUMNS].to_numpy())):
        raise DataContractError("Historical v1 join lost or added dataset bars")
    left = bars.set_index(KEY_COLUMNS).sort_index()
    right = frame.set_index(KEY_COLUMNS).sort_index()
    if not left[OHLCV].equals(right[OHLCV]):
        raise DataContractError("Historical v1 OHLCV differs from pinned processed bars")
    if str(frame.timestamp.dt.tz) != "UTC" or not frame.timestamp.dt.second.eq(0).all() or not frame.timestamp.dt.minute.mod(5).eq(0).all():
        raise DataContractError("Historical timestamps must be closed UTC five-minute boundaries")
    # Existing service keeps every valid bar; no score.dropna filtering here.
    binding.update(frame=frame.sort_values(KEY_COLUMNS).reset_index(drop=True), metadata=metadata,
                   sessions=sorted(frame.session_id.unique()), symbols=sorted(frame.symbol.unique()),
                   capabilities={"candles": True, "frozen_v1_scores": True, "v2_real_calibration": False,
                                 "news_receipts": False, "live": False, "per_bar_receive_time": False})
    _verify_loaded(binding)  # Reject a disk revision during loading, before exposing cacheable data.
    return binding


def historical_session(bundle, *, symbol, session_id, model, as_of=None):
    """Prefix by bar_end only: an explicit retrospective assumption, not a receipt."""
    _verify_loaded(bundle)  # Runs even when the caller retains/caches a loaded bundle.
    if symbol not in bundle["symbols"] or session_id not in bundle["sessions"] or model not in MODELS.values():
        raise DataContractError("Unknown historical symbol/session/model")
    selected = filtered_scores(bundle["frame"], symbol, session_id, model)
    cutoff = selected.timestamp.max() if as_of is None else pd.Timestamp(utc_time(as_of, "historical as_of"))
    if cutoff.tz_convert("Asia/Ho_Chi_Minh").date().isoformat() != session_id:
        raise DataContractError("Historical replay cutoff must be in the selected session")
    selected = selected.loc[selected.timestamp.le(cutoff)].copy().reset_index(drop=True)
    selected["score_eligible"] = selected.score.notna()
    selected["score_status"] = selected.score.notna().map({True: "frozen_v1_score", False: "frozen_v1_window_unavailable"})
    bars = selected[list(BAR_COLUMNS[:7]) + ["session_id", "segment_id", "quality_status", "feature_eligible", "feature_reasons", "score_eligible", "score_status"]].copy()
    return {"bars": bars, "scores": selected, "source_mode": "HISTORICAL_REPLAY", "bundle_key": bundle["key"],
            "symbol": symbol, "session_id": session_id, "model": model, "as_of": cutoff.isoformat(),
            "processed_at_utc": datetime.now(timezone.utc).isoformat(), "received_at_utc": None,
            "availability_basis": AVAILABILITY_BASIS, "exploratory": True,
            "data_policy": deepcopy(bundle["data_policy"]), "capabilities": deepcopy(bundle["capabilities"]),
            "provenance": {"manifest_sha256": bundle["manifest_sha256"],
                           "model_run": bundle["metadata"]["lstm" if model == "lstm" else "baseline"]["run_id"],
                           "feature_order": list(bundle["metadata"]["feature"]["feature_order"]),
                           "source_confirmations": deepcopy(bundle["metadata"]["feature"]["source_confirmations"]),
                           "effective_profile": deepcopy(bundle["metadata"]["feature"]["source_effective_profile"])}}


def historical_csv(snapshot):
    """Export exactly the selected prefix; no file-time receive-time substitution."""
    fields = ["symbol", "timestamp", "session_id", "segment_id", *OHLCV, "quality_status", "feature_eligible",
              "feature_reasons", "window_start", "split", "score", "threshold", "alert", "reason", "score_eligible", "score_status"]
    frame = snapshot["scores"][fields].copy()
    cutoff = pd.Timestamp(utc_time(snapshot["as_of"], "historical CSV as_of"))
    if (snapshot["source_mode"] != "HISTORICAL_REPLAY" or snapshot["model"] not in MODELS.values() or
            not frame.symbol.eq(snapshot["symbol"]).all() or not frame.session_id.eq(snapshot["session_id"]).all()):
        raise DataContractError("Historical CSV source/symbol/session/model scope differs")
    if not frame.timestamp.le(cutoff).all():
        raise DataContractError("Historical CSV contains a future bar")
    if not (frame.window_start.isna() | frame.window_start.le(frame.timestamp)).all():
        raise DataContractError("Historical CSV window starts in the future")
    for name in ("source_mode", "bundle_key", "model", "as_of", "processed_at_utc", "availability_basis", "exploratory"):
        frame.insert(len(frame.columns), name, snapshot[name])
    frame["source_rights_status"] = snapshot["data_policy"]["source_rights_status"]
    frame["distributable"] = snapshot["data_policy"]["distributable"]
    frame["received_at_utc"] = ""  # Unknown, not zero or a reconstructed observation time.
    frame["model_run"] = snapshot["provenance"]["model_run"]
    if snapshot["model"] == "lstm":
        for prefix in ("error", "share"):
            for name in snapshot["provenance"]["feature_order"]:
                frame[f"{prefix}_{name}"] = snapshot["scores"][f"{prefix}_{name}"]
    return event_csv(frame.to_dict("records")) if len(frame) else frame.to_csv(index=False).encode("utf-8-sig")


def historical_journal_path(root, bundle_key, model):
    """Namespace contract only; no write or synthetic-journal reuse in this slice."""
    if not isinstance(bundle_key, str) or not re.fullmatch(r"[0-9a-f]{64}", bundle_key) or model not in MODELS.values():
        raise DataContractError("Invalid historical journal binding")
    return Path(root).resolve() / ".local/research_ui" / bundle_key / "historical" / model / "journal.sqlite"
