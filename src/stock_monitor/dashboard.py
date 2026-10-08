"""Read-only data service for the local Streamlit dashboard."""

import json
from pathlib import Path

import pandas as pd

from .io import DataContractError, resolve_project_reference, sha256

MODELS = {
    "Robust z-score": "robust_zscore",
    "Isolation Forest": "isolation_forest",
}


def latest_baseline_bundle(root):
    directories = sorted((Path(root) / "artifacts" / "baselines").glob("baselines_*"))
    directories = [path for path in directories if path.is_dir()]
    if not directories:
        raise DataContractError("Chưa có baseline bundle. Hãy chạy CLI baselines trước.")
    return directories[-1].resolve()


def load_dashboard_bundle(baseline_directory):
    baseline_directory = Path(baseline_directory).resolve()
    metadata_path = baseline_directory / "metadata.json"
    if not metadata_path.is_file():
        raise DataContractError("Baseline bundle thiếu metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "baseline_bundle" or metadata.get("run_id") != baseline_directory.name:
        raise DataContractError("Baseline bundle không khớp metadata")
    for filename, digest in metadata.get("artifact_sha256", {}).items():
        file = (baseline_directory / filename).resolve()
        if not file.is_relative_to(baseline_directory) or not file.is_file() or sha256(file) != digest:
            raise DataContractError(f"Baseline artifact sai checksum: {filename}")
    project_root = baseline_directory.parents[2]
    feature_directory = resolve_project_reference(metadata["input_feature_dir"], project_root)
    feature_metadata_path = feature_directory / "metadata.json"
    if not feature_metadata_path.is_file() or sha256(feature_metadata_path) != metadata["input_feature_metadata_sha256"]:
        raise DataContractError("Feature bundle provenance không còn khớp")
    feature_metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
    features_path = feature_directory / "features.parquet"
    expected_features_hash = metadata["input_feature_artifact_sha256"].get("features.parquet")
    if expected_features_hash is None or sha256(features_path) != expected_features_hash:
        raise DataContractError("Feature parquet sai checksum")
    scores = pd.read_parquet(baseline_directory / "scores.parquet")
    bars = pd.read_parquet(features_path, columns=[
        "symbol", "timestamp", "session_id", "segment_id", "open", "high", "low", "close", "volume"
    ])
    if bars.duplicated(["symbol", "timestamp"]).any() or scores.duplicated(["symbol", "timestamp"]).any():
        raise DataContractError("Dashboard yêu cầu timestamp duy nhất theo mã")
    score_columns = ["symbol", "timestamp", "split", "robust_zscore", "isolation_forest",
                     "robust_zscore_threshold", "robust_zscore_alert",
                     "isolation_forest_threshold", "isolation_forest_alert"]
    frame = bars.merge(scores[score_columns], on=["symbol", "timestamp"], how="left", validate="one_to_one")
    for model in MODELS.values():
        frame[f"{model}_alert"] = frame[f"{model}_alert"].fillna(False).astype(bool)
    if len(frame) != len(bars) or int(frame.robust_zscore.notna().sum()) != len(scores):
        raise DataContractError("Dashboard join làm thay đổi hợp đồng bar/window")
    return frame, metadata, feature_metadata


def dashboard_slice(frame, symbol, session_id, model):
    if model not in MODELS.values():
        raise DataContractError("Baseline không được hỗ trợ")
    selected = frame.loc[frame.symbol.eq(symbol) & frame.session_id.eq(session_id)].sort_values("timestamp").copy()
    if selected.empty:
        raise DataContractError("Không có dữ liệu cho mã/ngày đã chọn")
    selected["alert"] = selected[f"{model}_alert"]
    selected["score"] = selected[model]
    selected["threshold"] = selected[f"{model}_threshold"]
    return selected
