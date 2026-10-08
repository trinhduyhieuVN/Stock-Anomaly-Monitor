"""Checked frozen artifacts, filtering, explanation and exports for the UI."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .alerts import group_consecutive_alerts
from .dashboard import dashboard_slice, load_dashboard_bundle
from .io import DataContractError, resolve_project_reference, sha256

MODELS = {"Robust z-score": "robust_zscore", "Isolation Forest": "isolation_forest", "LSTM Autoencoder": "lstm"}


def checked_metadata(directory, kind, required):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if kind is not None and (metadata.get("kind") != kind or metadata.get("run_id") != directory.name):
        raise DataContractError("Artifact không khớp loại/run_id đã khai báo")
    hashes = metadata.get("artifact_sha256", {})
    if not set(required).issubset(hashes):
        raise DataContractError("Artifact thiếu file bắt buộc: " + ", ".join(sorted(set(required) - set(hashes))))
    for filename, digest in hashes.items():
        file = (directory / filename).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or sha256(file) != digest:
            raise DataContractError(f"Artifact sai checksum hoặc thiếu file: {filename}")
    return metadata


def dashboard_artifact_key(baseline_directory, scoring_directory=None):
    baseline_directory = Path(baseline_directory).resolve()
    baseline = checked_metadata(baseline_directory, "baseline_bundle", ["scores.parquet", "models.joblib", "thresholds.json"])
    project_root = baseline_directory.parents[2]
    feature_directory = resolve_project_reference(baseline["input_feature_dir"], project_root)
    feature = checked_metadata(feature_directory, "feature_window_bundle", ["features.parquet", "windows.npz", "scalers.joblib"])
    feature_hash = sha256(feature_directory / "metadata.json")
    if feature_hash != baseline["input_feature_metadata_sha256"]:
        raise DataContractError("Baseline và feature provenance không khớp")
    dataset_directory = resolve_project_reference(feature["input_dataset_dir"], project_root)
    checked_metadata(dataset_directory, None, ["bars.parquet"])
    if sha256(dataset_directory / "metadata.json") != feature["input_metadata_sha256"]:
        raise DataContractError("Dataset provenance không khớp feature bundle")
    key_parts = [sha256(baseline_directory / "metadata.json"), feature_hash,
                 sha256(dataset_directory / "metadata.json")]
    if scoring_directory:
        scoring_directory = Path(scoring_directory).resolve()
        scoring = checked_metadata(scoring_directory, "lstm_scoring_bundle", ["scores.parquet", "thresholds.json", "alert_segments.parquet"])
        if (scoring["input_feature_metadata_sha256"] != feature_hash or
                resolve_project_reference(scoring["input_feature_dir"], project_root) != feature_directory or
                scoring["feature_order"] != feature["feature_order"]):
            raise DataContractError("LSTM và baseline phải dùng cùng feature bundle và feature order")
        lstm_directory = resolve_project_reference(scoring["input_lstm_dir"], project_root)
        lstm = checked_metadata(lstm_directory, "lstm_training_bundle", ["best_checkpoint.pt", "scalers.joblib"])
        if (sha256(lstm_directory / "metadata.json") != scoring["input_lstm_metadata_sha256"] or
                sha256(lstm_directory / "best_checkpoint.pt") != scoring["input_checkpoint_sha256"] or
                lstm["input_feature_metadata_sha256"] != feature_hash):
            raise DataContractError("LSTM model/scaler provenance không khớp scores")
        key_parts.extend([sha256(scoring_directory / "metadata.json"), sha256(lstm_directory / "metadata.json")])
    return hashlib.sha256(json.dumps(key_parts).encode("utf-8")).hexdigest()


def latest_scoring_bundle(root, feature_directory):
    candidates = sorted((Path(root) / "artifacts/scoring").glob("lstm_scores_*"), reverse=True)
    for directory in candidates:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if resolve_project_reference(metadata["input_feature_dir"], root) == Path(feature_directory).resolve():
            return directory.resolve()
    return None


def load_monitor_data(baseline_directory, scoring_directory=None):
    dashboard_artifact_key(baseline_directory, scoring_directory)
    frame, baseline, feature_metadata = load_dashboard_bundle(baseline_directory)
    project_root = Path(baseline_directory).resolve().parents[2]
    feature_directory = resolve_project_reference(baseline["input_feature_dir"], project_root)
    features = pd.read_parquet(feature_directory / "features.parquet")
    extra_columns = ["symbol", "timestamp", "quality_status", "feature_eligible", "feature_reasons"]
    frame = frame.merge(features[extra_columns], on=["symbol", "timestamp"], validate="one_to_one")
    baseline_scores = pd.read_parquet(Path(baseline_directory) / "scores.parquet")
    frame = frame.merge(baseline_scores[["symbol", "timestamp", "window_start"]],
                        on=["symbol", "timestamp"], how="left", validate="one_to_one")
    metadata = {"baseline": baseline, "feature": feature_metadata, "lstm": None}
    if scoring_directory:
        scoring_directory = Path(scoring_directory)
        scoring = json.loads((scoring_directory / "metadata.json").read_text(encoding="utf-8"))
        scores = pd.read_parquet(scoring_directory / "scores.parquet")
        if (scores.duplicated(["symbol", "timestamp"]).any() or
                set(map(tuple, scores[["symbol", "timestamp"]].to_numpy())) !=
                set(map(tuple, baseline_scores[["symbol", "timestamp"]].to_numpy()))):
            raise DataContractError("Các model không chấm cùng tập mã/timestamp")
        columns = ["symbol", "timestamp", "score", "threshold", "alert", "reason", "zero_total_error"]
        columns += [f"{prefix}_{name}" for prefix in ("error", "share") for name in feature_metadata["feature_order"]]
        values = scores[columns].rename(columns={"score": "lstm", "threshold": "lstm_threshold", "alert": "lstm_alert"})
        frame = frame.merge(values, on=["symbol", "timestamp"], how="left", validate="one_to_one")
        frame["lstm_alert"] = frame["lstm_alert"].fillna(False).astype(bool)
        metadata["lstm"] = scoring
    return frame, metadata, features


def frozen_model_resources(baseline_directory, scoring_directory=None):
    import joblib
    from .config import load_config
    from .pipeline import load_lstm_bundle

    dashboard_artifact_key(baseline_directory, scoring_directory)
    baseline_directory = Path(baseline_directory).resolve()
    project_root = baseline_directory.parents[2]
    baseline = json.loads((baseline_directory / "metadata.json").read_text(encoding="utf-8"))
    feature_directory = resolve_project_reference(baseline["input_feature_dir"], project_root)
    models = {"baseline": joblib.load(baseline_directory / "models.joblib")}
    models["scalers"] = joblib.load(feature_directory / "scalers.joblib")
    if scoring_directory:
        scoring = json.loads((Path(scoring_directory) / "metadata.json").read_text(encoding="utf-8"))
        feature_directory = resolve_project_reference(scoring["input_feature_dir"], project_root)
        feature_metadata = json.loads((feature_directory / "metadata.json").read_text(encoding="utf-8"))
        root = project_root
        config_name = "research.yaml" if baseline["source_profile_mode"] == "research" else "default.yaml"
        lstm_directory = resolve_project_reference(scoring["input_lstm_dir"], project_root)
        model, _, _, _ = load_lstm_bundle(load_config(root / "configs" / config_name), lstm_directory,
                                         feature_metadata, feature_directory / "metadata.json")
        models["lstm"] = model.eval()
    return models


def filtered_scores(frame, symbol, session_id, model):
    if model != "lstm":
        selected = dashboard_slice(frame, symbol, session_id, model)
        selected["reason"] = selected["alert"].map({True: "score_above_validation_q98", False: ""})
        return selected
    if "lstm" not in frame:
        raise DataContractError("Chưa có LSTM scores/model. Chạy train-lstm và score-lstm với cùng feature bundle.")
    selected = frame.loc[frame.symbol.eq(symbol) & frame.session_id.eq(session_id)].sort_values("timestamp").copy()
    if selected.empty:
        raise DataContractError("Không có dữ liệu cho mã/ngày đã chọn")
    selected["score"] = selected["lstm"]
    selected["threshold"] = selected["lstm_threshold"]
    selected["alert"] = selected["lstm_alert"]
    return selected


def daily_overview(frame, session_id, model):
    rows = []
    for symbol in sorted(frame.loc[frame.session_id.eq(session_id), "symbol"].unique()):
        selected = filtered_scores(frame, symbol, session_id, model)
        segments = group_consecutive_alerts(selected)
        rows.append({"Mã": symbol, "Số nến": len(selected), "Đã chấm": int(selected.score.notna().sum()),
                     "Điểm cảnh báo": int(selected.alert.sum()), "Đoạn cảnh báo": len(segments),
                     "Nến mới nhất": selected.timestamp.max().tz_convert("Asia/Ho_Chi_Minh").isoformat(),
                     "Chất lượng": ", ".join(sorted(selected.quality_status.unique()))})
    return pd.DataFrame(rows)


def observation_window(features, selected_bar):
    if pd.isna(selected_bar["window_start"]) or pd.isna(selected_bar["score"]):
        raise DataContractError("Nến này chưa đủ cửa sổ để chấm điểm")
    window = features.loc[features.symbol.eq(selected_bar["symbol"]) &
                          features.segment_id.eq(selected_bar["segment_id"]) &
                          features.timestamp.between(selected_bar["window_start"], selected_bar["timestamp"])].copy()
    if len(window) != 6 or not window.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all():
        raise DataContractError("Cửa sổ quan sát không khớp sáu nến liên tiếp")
    return window


def filtered_csv(selected, model, run_id, feature_order):
    columns = ["symbol", "timestamp", "window_start", "session_id", "segment_id", "open", "high", "low",
               "close", "volume", "quality_status", "split", "score", "threshold", "alert", "reason"]
    if model == "lstm":
        columns += [f"{prefix}_{name}" for prefix in ("error", "share") for name in feature_order]
    exported = selected[columns].copy()
    exported.insert(0, "run_id", run_id)
    exported.insert(1, "model", model)
    return exported.to_csv(index=False).encode("utf-8-sig")


OPERATOR_STATE_LABELS = {
    "OUTSIDE_SESSION": "Đang ngoài phiên giao dịch đã cấu hình",
    "BLOCKED_PROVIDER_ACCESS": "Bị chặn: quyền truy cập nguồn",
    "BLOCKED_PROVIDER_QUOTA": "Bị chặn: hạn mức nguồn",
    "BLOCKED_PROVIDER_PAYLOAD": "Bị chặn: phản hồi nguồn không hợp lệ",
    "BLOCKED_PROVIDER_HTTP": "Bị chặn: nguồn trả lỗi HTTP",
    "BLOCKED_ARTIFACT": "Bị chặn: artifact không hợp lệ",
    "BLOCKED_DATA_CONTRACT": "Bị chặn: dữ liệu không đúng hợp đồng",
    "BLOCKED_GAPS": "Bị chặn: thiếu nến đã đóng",
    "BLOCKED_READINESS_CONTRACT": "Bị chặn: readiness không hợp lệ",
    "BLOCKED_SCORING_CONTRACT": "Bị chặn: scoring không hợp lệ",
    "BLOCKED_UNKNOWN_CALENDAR": "Bị chặn: ngày chưa có bằng chứng lịch",
    "RETRY_EXHAUSTED": "Dừng: đã hết số lần thử",
    "RETRY_WAIT_BUDGET_EXHAUSTED": "Dừng: đã hết ngân sách chờ",
    "STALE": "Dữ liệu stale",
    "WAITING_FIRST_CLOSED_BAR": "Đang chờ nến 5 phút đầu tiên đóng",
    "WAITING_SYMBOLS": "Đang chờ đủ mã",
    "READY_FOR_SCORE": "Nến đã sẵn sàng để chấm",
    "SCORED_OFFLINE": "Đã chấm offline",
    "SCORED_LIVE_CANDIDATE": "Đã chấm với delay được đánh dấu live-candidate",
}


def load_operator_snapshot(project_root, path):
    root = Path(project_root).resolve()
    source = Path(path)
    source = source.resolve() if source.is_absolute() else (root / source).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise DataContractError("Operator status snapshot must be a file inside the project")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataContractError("Operator status snapshot is unreadable") from exc
    if value.get("schema_version") != 1 or value.get("kind") != "operator_status_snapshot":
        raise DataContractError("Operator status snapshot contract is incompatible")
    return value


def _operator_time(value):
    if value is None:
        return "Chưa đo trong phiên thật"
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("Operator timestamp is invalid") from exc
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise DataContractError("Operator timestamp must include a timezone")
    return stamp.tz_convert("Asia/Ho_Chi_Minh").isoformat()


def operator_status_view(snapshot):
    """Return fail-closed Vietnamese display values without asserting live status."""
    if not isinstance(snapshot, dict):
        raise DataContractError("Operator snapshot must be an object")
    state = snapshot.get("state")
    if not isinstance(state, str) or not state:
        raise DataContractError("Operator snapshot requires a state")
    live_evidence = snapshot.get("live_evidence") is True
    rights = snapshot.get("source_rights_status")
    if rights not in {"pending", "confirmed"}:
        raise DataContractError("Operator snapshot requires explicit source-rights status")
    gap_count = snapshot.get("gap_count", 0)
    if isinstance(gap_count, bool) or not isinstance(gap_count, int) or gap_count < 0:
        raise DataContractError("Operator gap_count must be a nonnegative integer")
    duration = snapshot.get("score_duration_ms")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0):
        raise DataContractError("Operator score_duration_ms must be nonnegative")
    offline = snapshot.get("offline_audit", {})
    offline_duration = offline.get("score_duration_ms")
    offline_receive = offline.get("received_at_utc")
    mode = ("LIVE-CANDIDATE — CHƯA PHẢI XÁC NHẬN HOÀN TẤT" if live_evidence
            else "OFFLINE — KHÔNG PHẢI LIVE")
    return {
        "state": state,
        "state_label": OPERATOR_STATE_LABELS.get(state, "Bị chặn: trạng thái chưa được hỗ trợ"),
        "severity": "info" if state in {"READY_FOR_SCORE", "SCORED_OFFLINE", "SCORED_LIVE_CANDIDATE"}
                    else "warning",
        "mode_label": mode,
        "latest_closed_bar": _operator_time(snapshot.get("latest_closed_bar_utc")),
        "source_rights_status": rights,
        "gap_count": gap_count,
        "stale": bool(snapshot.get("stale", False)),
        "receive_time": _operator_time(snapshot.get("received_at_utc")),
        "score_duration": (f"{float(duration):.3f} ms" if duration is not None
                           else "Chưa đo trong phiên thật"),
        "offline_audit": ("Không có" if not offline else
            f"receive={_operator_time(offline_receive)}; score={float(offline_duration):.3f} ms; "
            "chỉ là historical prefix"),
        "message": str(snapshot.get("operator_message", "")),
    }
