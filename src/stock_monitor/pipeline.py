"""Connect raw acquisition/validation to immutable Parquet bundles and reports."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import pending_confirmations, project_path
from .io import (DataContractError, _write_manifest_once, collect_probe_snapshot, read_bars,
                 read_table, resolve_project_reference, sha256)
from .providers import ADAPTER_VERSION, fetch_vci_raw, load_snapshot
from .validation import SourceNotConfirmed, audit_raw_minutes, load_profile, normalize_minutes, validate_standard_bars


def fetch_network_snapshot(settings):
    data = settings["data"]
    files = []
    http = data["http"]
    for symbol in data["symbols"]:
        raw, meta = fetch_vci_raw(project_path(settings, "raw_dir"), symbol, data["start"], data["end"], http["count_back"],
            max_attempts=http["max_attempts"], backoff_seconds=http["backoff_seconds"],
            max_wait_seconds=http["max_wait_seconds"], timeout=http["timeout"])
        files.append({"symbol": symbol, "raw_path": raw.relative_to(settings["_root"]).as_posix(),
            "metadata_path": meta.relative_to(settings["_root"]).as_posix(),
            "raw_sha256": sha256(raw), "metadata_sha256": sha256(meta)})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return _write_manifest_once(project_path(settings, "raw_dir") / "snapshots" / f"vci-network-{digest[:16]}",
        {"schema_version": 1, "kind": "raw_source_snapshot", "source": "VCI", "source_interval": "1m",
         "target_interval": "5m", "requested_start": data["start"], "requested_end": data["end"],
         "symbols": data["symbols"], "files": files, "normalized": False})


def preserve_input(settings, path):
    path = Path(path).resolve()
    digest = sha256(path)
    directory = project_path(settings, "raw_dir") / "file_inputs" / digest
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"input{path.suffix.lower()}"
    if destination.exists():
        if sha256(destination) != digest:
            raise DataContractError("Immutable input copy checksum mismatch")
    else:
        with path.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if sha256(destination) != digest:
            raise DataContractError("Source changed during input acquisition")
    return destination, {"path": destination.relative_to(settings["_root"]).as_posix(), "sha256": digest}


def write_bundle(settings, result, profile, provenance, *, kind):
    base = project_path(settings, "processed_dir") / ("diagnostic" if profile["diagnostic"] else
        "research" if profile.get("research", False) else "validated")
    name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:12]}"
    directory = base / f".{name}.pending"
    destination = base / name
    directory.mkdir(parents=True, exist_ok=False)
    data_file = "minutes.parquet" if kind == "raw_minutes_diagnostic" else "bars.parquet"
    result.bars.to_parquet(directory / data_file, index=False)
    result.rejected_rows.to_parquet(directory / "rejected_rows.parquet", index=False)
    result.rejected_bars.to_parquet(directory / "rejected_bars.parquet", index=False)
    files = {p.name: sha256(p) for p in directory.glob("*.parquet")}
    report = dict(result.report)
    report.update(output_dir=str(destination), dataset_sha256=files[data_file], data_file=data_file,
                  full_market_validation_implemented=True, market_checks_executed=True,
                  full_market_validation_executed=not profile["diagnostic"],
                  structural_status="PASS", scope=kind)
    metadata = {"schema_version": 1, "package_version": __version__, "adapter_version": ADAPTER_VERSION,
        "python": platform.python_version(), "libraries": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "pyarrow", "PyYAML"]},
        "source": "VCI", "source_interval": "5m" if kind == "standard_file" else "1m",
        "interval": "1m" if kind == "raw_minutes_diagnostic" else "5m", "timestamp_timezone": "UTC",
        "timestamp_convention": "unchanged_source_label_epoch_utc" if kind == "raw_minutes_diagnostic" else "bar_end",
        "target_price_unit": "VND", "target_volume_unit": "shares", "effective_profile": profile,
        "source_confirmations": settings["data"]["confirmations"],
        "quality_policy": settings["data"]["quality_policy"], "config_sha256": sha256(settings["_config_path"]),
        "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ["providers.py", "validation.py", "pipeline.py"]},
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance,
        "artifact_sha256": files, "validation": report,
        "note": ("Unverified raw/declared input; no source-label shifts or unit conversions, not approved for model use"
            if profile["diagnostic"] else "Local research only under documented assumptions; source confirmations remain pending"
            if profile.get("research", False) else "Approved source profile applied; see rejected rows and coverage")}
    for name, value in [("metadata.json", metadata), ("validation.json", report)]:
        with (directory / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    if not directory.resolve().is_relative_to(project_path(settings, "processed_dir")) or not destination.resolve().is_relative_to(project_path(settings, "processed_dir")):
        raise DataContractError("Bundle path escapes processed storage")
    directory.rename(destination)
    return report


def run_validation(settings, *, input_path=None, snapshot_path=None, diagnostic=False, as_of=None):
    data = settings["data"]
    if input_path:
        preserved, provenance = preserve_input(settings, input_path)
        frame = read_table(preserved)
        kind = "standard_file"
    else:
        snapshot_path = Path(snapshot_path) if snapshot_path else collect_probe_snapshot(settings)
        frame, manifest, provenance = load_snapshot(snapshot_path, settings["_root"], project_path(settings, "raw_dir"))
        if (manifest["requested_start"], manifest["requested_end"], set(manifest["symbols"])) != (data["start"], data["end"], set(data["symbols"])):
            raise DataContractError("Snapshot scope differs from config")
        kind = "raw_snapshot"
    try:
        profile = load_profile(settings, diagnostic)
    except SourceNotConfirmed as exc:
        return {"status": "BLOCKED_SOURCE_CONFIRMATIONS", "error": str(exc), "scope": kind,
            "structural_status": "PASS" if input_path else "PASS_RAW_HASHES",
            "received_rows": len(frame), "blocked_rows": len(frame), "accepted_bars": 0,
            "rejected_rows": 0, "blocked_reasons": pending_confirmations(settings),
            "source_confirmations_pending": pending_confirmations(settings), "training_ready": False,
            "full_market_validation_implemented": True, "market_checks_executed": False, "output_dir": None}
    if input_path:
        result = validate_standard_bars(frame, profile, start=data["start"], end=data["end"], symbols=data["symbols"], as_of=as_of)
    else:
        if diagnostic:
            result = audit_raw_minutes(frame, profile, start=data["start"], end=data["end"], symbols=data["symbols"], as_of=as_of)
            kind = "raw_minutes_diagnostic"
        else:
            result = normalize_minutes(frame, profile, start=data["start"], end=data["end"], symbols=data["symbols"], as_of=as_of)
    return write_bundle(settings, result, profile, provenance, kind=kind)


def load_dataset_bundle(settings, directory):
    """Require our checked raw-derived bundle; reject diagnostics/declared imports."""
    directory = Path(directory).resolve()
    if not directory.is_relative_to(project_path(settings, "processed_dir")):
        raise DataContractError("Feature input must be a dataset bundle inside processed storage")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = metadata["validation"]
    mode = settings["data"].get("normalization_mode", "strict")
    allowed = "RESEARCH_VALIDATED" if mode == "research" else "VALIDATED"
    if validation["status"] != allowed or validation["scope"] != "raw_snapshot":
        raise DataContractError("Feature input must be a checked raw-source 5m bundle in the selected mode")
    if metadata["interval"] != "5m" or metadata["timestamp_convention"] != "bar_end":
        raise DataContractError("Diagnostic minutes are not model bars")
    expected = load_profile(settings)
    actual = metadata["effective_profile"]
    for key in ("bar_label", "price", "volume", "calendar_sha256", "research"):
        if actual.get(key) != expected.get(key):
            raise DataContractError(f"Dataset source profile differs from feature config: {key}")
    if (validation["requested_start"], validation["requested_end"]) != (settings["data"]["start"], settings["data"]["end"]):
        raise DataContractError("Feature dataset dates differ from config")
    if validation["data_file"] != "bars.parquet" or metadata["artifact_sha256"].get("bars.parquet") != validation["dataset_sha256"]:
        raise DataContractError("Dataset checksum declarations disagree")
    for filename, digest in metadata["artifact_sha256"].items():
        file = (directory / filename).resolve()
        if not file.is_relative_to(directory) or sha256(file) != digest:
            raise DataContractError(f"Dataset bundle checksum mismatch: {filename}")
    provenance = metadata["provenance"]
    if not isinstance(provenance, list) or {item["symbol"] for item in provenance} != set(settings["data"]["symbols"]):
        raise DataContractError("Dataset requires raw provenance for every configured symbol")
    for item in provenance:
        raw = (settings["_root"] / item["raw_path"]).resolve()
        raw_metadata = raw.with_name("metadata.json")
        if not raw.is_relative_to(project_path(settings, "raw_dir")) or (
                sha256(raw) != item["raw_sha256"] or sha256(raw_metadata) != item["metadata_sha256"]):
            raise DataContractError("Feature input raw provenance checksum mismatch")
    frame = read_bars(directory / "bars.parquet", symbols=settings["data"]["symbols"])
    if len(frame) != validation["accepted_bars"] or set(frame.symbol) != set(settings["data"]["symbols"]):
        raise DataContractError("Dataset counts/symbols differ from the validation report")
    return frame, metadata, metadata_path


def run_features(settings, dataset_directory):
    """Create a reproducible feature/scaler/window bundle; no model training."""
    import joblib
    import numpy as np
    from .features import FEATURE_ORDER, compute_features
    from .windows import fit_scalers, make_windows, split_by_common_dates, transform_features

    bars, input_metadata, input_metadata_path = load_dataset_bundle(settings, dataset_directory)
    feature_config = settings["features"]
    if feature_config["order"] != list(FEATURE_ORDER):
        raise DataContractError("Feature order differs from the four-feature contract")
    if settings["split"]["by"] != "common_trading_date":
        raise DataContractError("Only common trading-date splits are supported")
    computed = compute_features(bars, input_metadata["effective_profile"]["calendar"],
        history_sessions=feature_config["volume_history_sessions"],
        min_history_sessions=feature_config["volume_min_history_sessions"],
        warmup_sessions=feature_config["warmup_sessions"])
    frame, splits = split_by_common_dates(computed.frame, settings["data"]["symbols"],
        tuple(settings["split"][name] for name in ("train", "validation", "test")))
    scalers, scaler_report = fit_scalers(frame, settings["data"]["symbols"])
    frame = transform_features(frame, scalers)
    windows = make_windows(frame, window_length=feature_config["window_length"])
    if any(count == 0 for per_symbol in windows.report["by_symbol"].values() for count in per_symbol.values()):
        raise DataContractError("Every symbol/split requires at least one complete window")
    base = (settings["_root"] / feature_config.get("artifact_dir", "artifacts/features")).resolve()
    if not base.is_relative_to(settings["_root"]) or any(base.is_relative_to(project_path(settings, key))
            for key in ("raw_dir", "processed_dir")):
        raise DataContractError("Feature artifacts must stay in a separate project directory")
    run_id = f"features_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:12]}"
    directory, destination = base / f".{run_id}.pending", base / run_id
    directory.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(directory / "features.parquet", index=False)
    windows.index.to_parquet(directory / "window_index.parquet", index=False)
    np.savez_compressed(directory / "windows.npz", values=windows.values, row_indices=windows.row_indices)
    joblib.dump(scalers, directory / "scalers.joblib")
    shutil.copyfile(settings["_config_path"], directory / "config.yaml")
    for filename, value in (("split.json", splits), ("scaler_fit.json", scaler_report)):
        (directory / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    research = settings["data"].get("normalization_mode", "strict") == "research"
    report = {"status": "FEATURES_READY_RESEARCH" if research else "FEATURES_READY", "run_id": run_id,
        "output_dir": str(destination), "dataset_dir": str(Path(dataset_directory).resolve()),
        "dataset_sha256": input_metadata["validation"]["dataset_sha256"],
        "source_profile_mode": "research" if research else "strict", "source_units_verified": not research,
        "source_confirmations_pending": pending_confirmations(settings),
        "research_ready": research, "training_ready": not research, "model_trained": False,
        "feature_order": list(FEATURE_ORDER), "features": computed.report, "split": splits,
        "scaler_fit": scaler_report, "windows": windows.report}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = {file.name: sha256(file) for file in directory.iterdir() if file.is_file()}
    metadata = {"schema_version": 1, "kind": "feature_window_bundle", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "feature_order": list(FEATURE_ORDER),
        "window_length": feature_config["window_length"], "source_profile_mode": report["source_profile_mode"],
        "source_effective_profile": input_metadata["effective_profile"], "source_confirmations": input_metadata["source_confirmations"],
        "input_dataset_dir": str(Path(dataset_directory).resolve()), "input_dataset_sha256": report["dataset_sha256"],
        "input_metadata_sha256": sha256(input_metadata_path), "input_provenance": input_metadata["provenance"],
        "config_sha256": sha256(settings["_config_path"]), "artifact_sha256": files,
        "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
            ("features.py", "windows.py", "pipeline.py", "config.py")},
        "libraries": {name: importlib.metadata.version(name) for name in
            ("pandas", "numpy", "pyarrow", "scikit-learn", "joblib", "PyYAML")}, "validation": report,
        "note": "No model trained. Research assumptions and source permissions remain separate from technical checks."}
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    directory.rename(destination)
    return report


def load_feature_bundle(settings, directory):
    """Load a checked feature/window bundle without regenerating its data."""
    import numpy as np
    import pandas as pd

    directory = Path(directory).resolve()
    base = (settings["_root"] / settings["features"].get("artifact_dir", "artifacts/features")).resolve()
    if not directory.is_relative_to(base):
        raise DataContractError("Baseline input must be a feature bundle inside feature artifact storage")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "feature_window_bundle" or metadata.get("run_id") != directory.name:
        raise DataContractError("Baseline input is not a matching feature/window bundle")
    expected_mode = "research" if settings["data"].get("normalization_mode", "strict") == "research" else "strict"
    if metadata.get("source_profile_mode") != expected_mode:
        raise DataContractError("Feature bundle source profile differs from baseline config")
    feature_order = settings["features"]["order"]
    if metadata.get("feature_order") != feature_order or metadata.get("window_length") != settings["features"]["window_length"]:
        raise DataContractError("Feature bundle shape contract differs from baseline config")
    if metadata.get("config_sha256") != sha256(settings["_config_path"]):
        raise DataContractError("Feature bundle config checksum differs from baseline config")
    for filename, digest in metadata.get("artifact_sha256", {}).items():
        file = (directory / filename).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or sha256(file) != digest:
            raise DataContractError(f"Feature bundle checksum mismatch: {filename}")
    required = {"windows.npz", "window_index.parquet", "features.parquet", "scalers.joblib"}
    if not required.issubset(metadata.get("artifact_sha256", {})):
        raise DataContractError("Feature bundle is missing required checked artifacts")
    input_metadata = Path(metadata["input_dataset_dir"]) / "metadata.json"
    if not input_metadata.is_file() or sha256(input_metadata) != metadata.get("input_metadata_sha256"):
        raise DataContractError("Feature bundle input metadata checksum mismatch")
    with np.load(directory / "windows.npz") as archive:
        if set(archive.files) != {"values", "row_indices"}:
            raise DataContractError("Feature bundle NPZ members differ from contract")
        windows = archive["values"]
    index = pd.read_parquet(directory / "window_index.parquet")
    return windows, index, metadata, metadata_path


def run_baselines(settings, feature_directory):
    """Fit and persist real train-only baselines for one checked feature bundle."""
    import joblib
    import numpy as np
    from .baselines import calibrate_thresholds, fit_baselines, score_baselines

    windows, index, input_metadata, input_metadata_path = load_feature_bundle(settings, feature_directory)
    feature_order = tuple(input_metadata["feature_order"])
    seed = settings["model"]["seed"]
    quantile = settings["model"]["threshold_quantile"]
    bundle = fit_baselines(windows, index, feature_order, random_state=seed)
    scores = score_baselines(bundle, windows, index)
    thresholds = calibrate_thresholds(scores, quantile=quantile)
    for model in ("robust_zscore", "isolation_forest"):
        threshold_column, alert_column = f"{model}_threshold", f"{model}_alert"
        scores[threshold_column] = scores.symbol.map({symbol: values[model]["threshold"]
                                                       for symbol, values in thresholds.items()})
        scores[alert_column] = scores[model].gt(scores[threshold_column])
    base = (settings["_root"] / "artifacts/baselines").resolve()
    if not base.is_relative_to(settings["_root"]) or any(base.is_relative_to(project_path(settings, key))
            for key in ("raw_dir", "processed_dir")):
        raise DataContractError("Baseline artifacts must stay in a separate project directory")
    run_id = f"baselines_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:12]}"
    directory, destination = base / f".{run_id}.pending", base / run_id
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(bundle, directory / "models.joblib")
    scores.to_parquet(directory / "scores.parquet", index=False)
    shutil.copyfile(settings["_config_path"], directory / "config.yaml")
    for filename, value in (("thresholds.json", thresholds), ("fit_report.json", bundle.fit_report)):
        (directory / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loaded = joblib.load(directory / "models.joblib")
    reloaded_scores = score_baselines(loaded, windows, index)
    try:
        np.testing.assert_allclose(reloaded_scores[["robust_zscore", "isolation_forest"]],
                                   scores[["robust_zscore", "isolation_forest"]], rtol=0, atol=0)
    except AssertionError as exc:
        raise DataContractError("Reloaded baseline scores differ from persisted run") from exc
    counts = {symbol: {split: {model: int(scores.loc[scores.symbol.eq(symbol) & scores.split.eq(split),
            f"{model}_alert"].sum()) for model in ("robust_zscore", "isolation_forest")}
            for split in ("train", "validation", "test")} for symbol in sorted(scores.symbol.unique())}
    research = input_metadata["source_profile_mode"] == "research"
    report = {"status": "BASELINES_READY_RESEARCH" if research else "BASELINES_READY", "run_id": run_id,
        "output_dir": str(destination), "feature_bundle_dir": str(Path(feature_directory).resolve()),
        "feature_bundle_sha256": sha256(input_metadata_path), "source_profile_mode": input_metadata["source_profile_mode"],
        "source_confirmations_pending": pending_confirmations(settings), "research_ready": research,
        "training_ready": not research, "model_trained": True, "feature_order": list(feature_order),
        "window_shape": list(windows.shape), "fit": bundle.fit_report, "thresholds": thresholds,
        "alert_counts": counts, "reload_score_match": True,
        "score_direction": "larger means more noteworthy; scores are not probabilities"}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = {file.name: sha256(file) for file in directory.iterdir() if file.is_file()}
    metadata = {"schema_version": 1, "kind": "baseline_bundle", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "feature_order": list(feature_order),
        "window_length": windows.shape[1], "source_profile_mode": input_metadata["source_profile_mode"],
        "source_confirmations": input_metadata["source_confirmations"],
        "input_feature_dir": str(Path(feature_directory).resolve()),
        "input_feature_metadata_sha256": sha256(input_metadata_path),
        "input_feature_artifact_sha256": input_metadata["artifact_sha256"],
        "input_dataset_dir": input_metadata["input_dataset_dir"],
        "input_dataset_sha256": input_metadata["input_dataset_sha256"],
        "input_provenance": input_metadata["input_provenance"], "config_sha256": sha256(settings["_config_path"]),
        "artifact_sha256": files, "implementation_sha256": {name: sha256(Path(__file__).with_name(name))
            for name in ("baselines.py", "pipeline.py", "config.py")},
        "libraries": {name: importlib.metadata.version(name) for name in
            ("pandas", "numpy", "pyarrow", "scikit-learn", "joblib", "PyYAML")},
        "validation": report,
        "note": "Local research baseline; thresholds are validation quantiles, not anomaly probabilities."}
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    directory.rename(destination)
    return report


def run_lstm_training(settings, feature_directory):
    """Train and persist one real LSTM autoencoder using train/validation only."""
    import time
    import torch
    from .model import LSTMAutoencoder
    from .training import fit_autoencoder, reconstruct

    windows, index, input_metadata, input_metadata_path = load_feature_bundle(settings, feature_directory)
    model_settings = settings["model"]
    if model_settings.get("architecture") != "lstm_autoencoder":
        raise DataContractError("Prompt 7 requires architecture: lstm_autoencoder")
    expected = {"hidden_size": 32, "latent_size": 8, "num_layers": 1, "batch_size": 64,
                "max_epochs": 50, "learning_rate": 0.001, "patience": 5, "seed": 42, "device": "cpu"}
    if any(model_settings.get(key) != value for key, value in expected.items()):
        raise DataContractError("Prompt 7 model configuration differs from the approved contract")
    split_counts = index.split.value_counts().to_dict()
    if set(split_counts) != {"train", "validation", "test"} or any(split_counts[name] < 1 for name in split_counts):
        raise DataContractError("LSTM training requires non-empty train, validation and untouched test splits")
    train_mask = index.split.eq("train").to_numpy()
    validation_mask = index.split.eq("validation").to_numpy()
    train_values, validation_values = windows[train_mask], windows[validation_mask]
    training_config = {**expected, "input_size": windows.shape[2]}
    started = time.perf_counter()
    model, result = fit_autoencoder(train_values, validation_values, training_config)
    elapsed = time.perf_counter() - started
    base = (settings["_root"] / "artifacts/lstm").resolve()
    if not base.is_relative_to(settings["_root"]) or any(base.is_relative_to(project_path(settings, key))
            for key in ("raw_dir", "processed_dir")):
        raise DataContractError("LSTM artifacts must stay in a separate project directory")
    run_id = f"lstm_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:12]}"
    directory, destination = base / f".{run_id}.pending", base / run_id
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint = {"schema_version": 1, "kind": "lstm_autoencoder_checkpoint", "run_id": run_id,
        "model_config": training_config, "state_dict": result.best_state,
        "best_epoch": result.best_epoch, "best_validation_loss": result.best_validation_loss,
        "feature_order": input_metadata["feature_order"], "window_length": input_metadata["window_length"],
        "input_feature_run_id": input_metadata["run_id"],
        "input_feature_metadata_sha256": sha256(input_metadata_path), "fit_split": "train",
        "early_stopping_split": "validation", "test_used_for_fit_or_selection": False}
    torch.save(checkpoint, directory / "best_checkpoint.pt")
    (directory / "history.json").write_text(json.dumps(result.history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    feature_directory = Path(feature_directory).resolve()
    for source, destination_name in ((settings["_config_path"], "config.yaml"),
                                     (feature_directory / "scalers.joblib", "scalers.joblib"),
                                     (feature_directory / "split.json", "split.json"),
                                     (feature_directory / "scaler_fit.json", "scaler_fit.json")):
        shutil.copyfile(source, directory / destination_name)
    loaded = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    reloaded = LSTMAutoencoder(input_size=loaded["model_config"]["input_size"],
        hidden_size=loaded["model_config"]["hidden_size"], latent_size=loaded["model_config"]["latent_size"],
        num_layers=loaded["model_config"]["num_layers"])
    reloaded.load_state_dict(loaded["state_dict"])
    sample = validation_values[:min(256, len(validation_values))]
    reference = reconstruct(model, sample)
    reloaded_output = reconstruct(reloaded, sample)
    reload_max_abs_diff = float(abs(reference - reloaded_output).max())
    if reload_max_abs_diff != 0:
        raise DataContractError("Reloaded LSTM inference differs from best in-memory checkpoint")
    research = input_metadata["source_profile_mode"] == "research"
    report = {"status": "LSTM_READY_RESEARCH" if research else "LSTM_READY", "run_id": run_id,
        "output_dir": str(destination), "feature_bundle_dir": str(feature_directory),
        "feature_bundle_sha256": sha256(input_metadata_path), "source_profile_mode": input_metadata["source_profile_mode"],
        "source_confirmations_pending": pending_confirmations(settings), "research_ready": research,
        "training_ready": not research, "model_trained": True, "feature_order": input_metadata["feature_order"],
        "window_shape": list(windows.shape), "split_windows": {name: int(split_counts[name])
            for name in ("train", "validation", "test")}, "fit_split": "train",
        "early_stopping_split": "validation", "test_used_for_fit_or_selection": False,
        "model_config": training_config, "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": result.history, "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss, "epochs_ran": result.epochs_ran,
        "stopped_early": result.stopped_early, "training_seconds": elapsed,
        "reload_inference_samples": len(sample), "reload_max_abs_diff": reload_max_abs_diff}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = {file.name: sha256(file) for file in directory.iterdir() if file.is_file()}
    metadata = {"schema_version": 1, "kind": "lstm_training_bundle", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "feature_order": input_metadata["feature_order"],
        "window_length": input_metadata["window_length"], "source_profile_mode": input_metadata["source_profile_mode"],
        "source_confirmations": input_metadata["source_confirmations"],
        "input_feature_dir": str(feature_directory), "input_feature_metadata_sha256": sha256(input_metadata_path),
        "input_feature_artifact_sha256": input_metadata["artifact_sha256"],
        "input_dataset_dir": input_metadata["input_dataset_dir"],
        "input_dataset_sha256": input_metadata["input_dataset_sha256"],
        "input_provenance": input_metadata["input_provenance"], "config_sha256": sha256(settings["_config_path"]),
        "artifact_sha256": files, "implementation_sha256": {name: sha256(Path(__file__).with_name(name))
            for name in ("model.py", "training.py", "pipeline.py", "config.py")},
        "libraries": {name: importlib.metadata.version(name) for name in
            ("torch", "pandas", "numpy", "pyarrow", "scikit-learn", "joblib", "PyYAML")},
        "validation": report,
        "note": "Local research training; test split was untouched and loss does not establish detection quality."}
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    directory.rename(destination)
    return report


def load_lstm_bundle(settings, directory, input_metadata, input_metadata_path):
    """Load a checked LSTM checkpoint tied to exactly one feature bundle."""
    import torch
    from .model import LSTMAutoencoder

    directory = Path(directory).resolve()
    base = (settings["_root"] / "artifacts/lstm").resolve()
    if not directory.is_relative_to(base):
        raise DataContractError("LSTM input must be inside LSTM artifact storage")
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "lstm_training_bundle" or metadata.get("run_id") != directory.name:
        raise DataContractError("LSTM input is not a matching training bundle")
    for filename, digest in metadata.get("artifact_sha256", {}).items():
        file = (directory / filename).resolve()
        if not file.is_relative_to(directory) or not file.is_file() or sha256(file) != digest:
            raise DataContractError(f"LSTM bundle checksum mismatch: {filename}")
    checkpoint_path = directory / "best_checkpoint.pt"
    if "best_checkpoint.pt" not in metadata.get("artifact_sha256", {}):
        raise DataContractError("LSTM bundle is missing its checked best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_feature_hash = sha256(input_metadata_path)
    if (metadata.get("input_feature_metadata_sha256") != expected_feature_hash or
            checkpoint.get("input_feature_metadata_sha256") != expected_feature_hash or
            resolve_project_reference(metadata.get("input_feature_dir", ""), settings["_root"]) !=
            Path(input_metadata_path).parent.resolve()):
        raise DataContractError("LSTM checkpoint was not trained from this feature bundle")
    feature_order = input_metadata["feature_order"]
    if (metadata.get("feature_order") != feature_order or checkpoint.get("feature_order") != feature_order or
            feature_order != settings["features"]["order"]):
        raise DataContractError("LSTM feature order differs from the scoring feature contract")
    if (metadata.get("window_length") != input_metadata["window_length"] or
            checkpoint.get("window_length") != input_metadata["window_length"]):
        raise DataContractError("LSTM window length differs from the feature bundle")
    config = checkpoint["model_config"]
    if config.get("input_size") != len(feature_order) or config.get("device") != "cpu":
        raise DataContractError("LSTM checkpoint model shape or device differs from the scoring contract")
    model = LSTMAutoencoder(input_size=config["input_size"], hidden_size=config["hidden_size"],
                            latent_size=config["latent_size"], num_layers=config["num_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    return model, metadata, metadata_path, checkpoint_path


def run_lstm_scoring(settings, feature_directory, lstm_directory):
    """Score all frozen windows and persist calibrated LSTM alerts."""
    import numpy as np
    import pandas as pd
    from .alerts import apply_thresholds, group_consecutive_alerts
    from .scoring import calibrate_thresholds, score_reconstructions
    from .training import reconstruct

    windows, index, input_metadata, input_metadata_path = load_feature_bundle(settings, feature_directory)
    model, lstm_metadata, lstm_metadata_path, checkpoint_path = load_lstm_bundle(
        settings, lstm_directory, input_metadata, input_metadata_path)
    feature_order = input_metadata["feature_order"]
    reconstructed = reconstruct(model, windows)
    scores = score_reconstructions(windows, reconstructed, index, feature_order,
                                    expected_feature_order=lstm_metadata["feature_order"])
    features = pd.read_parquet(Path(feature_directory).resolve() / "features.parquet")
    end_rows = scores["end_row"].to_numpy(dtype=np.int64)
    if len(end_rows) and (end_rows.min() < 0 or end_rows.max() >= len(features)):
        raise DataContractError("Window end-row pointers escape the feature table")
    closed_rows = features.iloc[end_rows].reset_index(drop=True)
    if (not closed_rows["symbol"].equals(scores["symbol"].reset_index(drop=True)) or
            not closed_rows["timestamp"].equals(scores["timestamp"].reset_index(drop=True))):
        raise DataContractError("Window timestamps do not match their feature-row pointers")
    scores["quality_status"] = closed_rows["quality_status"]
    quantile = settings["model"]["threshold_quantile"]
    thresholds = calibrate_thresholds(scores, quantile=quantile)
    scores = apply_thresholds(scores, thresholds)

    base = (settings["_root"] / "artifacts/scoring").resolve()
    if not base.is_relative_to(settings["_root"]) or any(base.is_relative_to(project_path(settings, key))
            for key in ("raw_dir", "processed_dir")):
        raise DataContractError("Scoring artifacts must stay in a separate project directory")
    run_id = f"lstm_scores_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:12]}"
    directory, destination = base / f".{run_id}.pending", base / run_id
    scores.insert(0, "run_id", run_id)
    alert_segments = group_consecutive_alerts(scores)

    reloaded, _, _, _ = load_lstm_bundle(settings, lstm_directory, input_metadata, input_metadata_path)
    reload_scores = score_reconstructions(windows, reconstruct(reloaded, windows), index, feature_order,
                                          expected_feature_order=lstm_metadata["feature_order"])
    reload_max_abs_diff = float(np.max(np.abs(reload_scores["score"].to_numpy() - scores["score"].to_numpy())))
    if reload_max_abs_diff != 0:
        raise DataContractError("Reloaded LSTM scores differ from the persisted scoring run")
    cutoff = sorted(scores["timestamp"].unique())[len(scores["timestamp"].unique()) // 2]
    prefix_mask = scores["timestamp"].le(cutoff).to_numpy()
    prefix_scores = score_reconstructions(windows[prefix_mask], reconstruct(reloaded, windows[prefix_mask]),
                                          index.loc[prefix_mask].reset_index(drop=True), feature_order,
                                          expected_feature_order=lstm_metadata["feature_order"])
    prefix_max_abs_diff = float(np.max(np.abs(prefix_scores["score"].to_numpy() -
                                                   scores.loc[prefix_mask, "score"].to_numpy())))
    prefix_tolerance = 1e-7
    if prefix_max_abs_diff > prefix_tolerance:
        raise DataContractError("Appending future windows changed scores at or before the prefix cutoff")

    directory.mkdir(parents=True, exist_ok=False)
    scores.to_parquet(directory / "scores.parquet", index=False)
    alert_segments.to_parquet(directory / "alert_segments.parquet", index=False)
    shutil.copyfile(settings["_config_path"], directory / "config.yaml")
    (directory / "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    research = input_metadata["source_profile_mode"] == "research"
    split_counts = scores["split"].value_counts().to_dict()
    alert_counts = {symbol: {split: int(scores.loc[scores.symbol.eq(symbol) & scores.split.eq(split), "alert"].sum())
                             for split in ("train", "validation", "test")}
                    for symbol in sorted(scores.symbol.unique())}
    report = {"status": "LSTM_SCORES_READY_RESEARCH" if research else "LSTM_SCORES_READY", "run_id": run_id,
        "output_dir": str(destination), "feature_bundle_dir": str(Path(feature_directory).resolve()),
        "feature_bundle_sha256": sha256(input_metadata_path), "lstm_bundle_dir": str(Path(lstm_directory).resolve()),
        "lstm_bundle_sha256": sha256(lstm_metadata_path), "checkpoint_sha256": sha256(checkpoint_path),
        "source_profile_mode": input_metadata["source_profile_mode"],
        "source_confirmations_pending": pending_confirmations(settings), "research_ready": research,
        "training_ready": not research, "feature_order": feature_order, "window_shape": list(windows.shape),
        "split_windows": {name: int(split_counts[name]) for name in ("train", "validation", "test")},
        "score_definition": "mean squared reconstruction error across features at the final bar",
        "contribution_definition": "feature squared error divided by total feature squared error; not a market cause",
        "zero_total_error_policy": "all shares are zero and zero_total_error is true",
        "score_is_probability": False, "thresholds": thresholds, "alert_counts": alert_counts,
        "alert_segment_count": len(alert_segments), "reload_score_match": True,
        "reload_max_abs_diff": reload_max_abs_diff, "prefix_cutoff": pd.Timestamp(cutoff).isoformat(),
        "prefix_samples": int(prefix_mask.sum()), "prefix_score_match": True,
        "prefix_max_abs_diff": prefix_max_abs_diff, "prefix_absolute_tolerance": prefix_tolerance}
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = {file.name: sha256(file) for file in directory.iterdir() if file.is_file()}
    metadata = {"schema_version": 1, "kind": "lstm_scoring_bundle", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "feature_order": feature_order,
        "window_length": input_metadata["window_length"], "source_profile_mode": input_metadata["source_profile_mode"],
        "source_confirmations": input_metadata["source_confirmations"],
        "input_feature_dir": str(Path(feature_directory).resolve()),
        "input_feature_metadata_sha256": sha256(input_metadata_path),
        "input_lstm_dir": str(Path(lstm_directory).resolve()), "input_lstm_metadata_sha256": sha256(lstm_metadata_path),
        "input_checkpoint_sha256": sha256(checkpoint_path), "input_dataset_dir": input_metadata["input_dataset_dir"],
        "input_dataset_sha256": input_metadata["input_dataset_sha256"],
        "input_provenance": input_metadata["input_provenance"], "config_sha256": sha256(settings["_config_path"]),
        "artifact_sha256": files, "implementation_sha256": {name: sha256(Path(__file__).with_name(name))
            for name in ("scoring.py", "alerts.py", "model.py", "training.py", "pipeline.py", "config.py")},
        "libraries": {name: importlib.metadata.version(name) for name in
            ("torch", "pandas", "numpy", "pyarrow", "PyYAML")}, "validation": report,
        "note": "Local research scores; reconstruction error is not a probability and feature shares are not causal explanations."}
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    directory.rename(destination)
    return report
