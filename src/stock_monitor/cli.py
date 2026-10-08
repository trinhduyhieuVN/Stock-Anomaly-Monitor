"""Acquire immutable data and run evidence-aware OHLCV market validation."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config, pending_confirmations, project_path
from .io import DataContractError, collect_probe_snapshot, import_bar_file
from .pipeline import (fetch_network_snapshot, run_baselines, run_features, run_lstm_scoring,
                       run_lstm_training, run_validation)


def make_parser():
    parser = argparse.ArgumentParser(prog="python -m stock_monitor.cli",
        description="OHLCV VCI 1m -> 5m: raw bất biến, validation, features và baseline nghiên cứu.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Thu nhận raw: cache mặc định, API với --network, hoặc CSV/Parquet chuẩn")
    fetch.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    acquisition = fetch.add_mutually_exclusive_group()
    acquisition.add_argument("--input", type=Path, help="File CSV/Parquet schema chuẩn để lưu bản sao bất biến")
    acquisition.add_argument("--network", action="store_true", help="Gọi VCI có giới hạn retry/quota, lưu raw mới; không đổi nguồn")
    validate = commands.add_parser("validate", help="Kiểm tra thị trường, phiên/gap và xuất Parquet kèm metadata")
    validate.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    inputs = validate.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="CSV/Parquet schema chuẩn để kiểm tra từng hàng")
    inputs.add_argument("--snapshot", type=Path, help="Manifest raw VCI; mặc định cache đã audit")
    validate.add_argument("--diagnostic", action="store_true", help="Audit raw 1m, không đổi đơn vị/nhãn; xuất riêng, unverified")
    validate.add_argument("--as-of", help="Mốc đóng nến ISO-8601 có timezone; mặc định giờ UTC hiện tại")
    validate.add_argument("--report", type=Path, help="Đường dẫn JSON báo cáo; mặc định reports/contract_validation.json")
    features = commands.add_parser("features", help="Đặc trưng nhân quả, chia ngày, scaler train-only và cửa sổ")
    features.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    features.add_argument("--dataset", type=Path, required=True, help="Thư mục bundle 5m raw-derived đã validation")
    features.add_argument("--report", type=Path, help="JSON báo cáo; mặc định reports/feature_preparation.json")
    baselines = commands.add_parser("baselines", help="Fit robust z-score và Isolation Forest từ feature bundle")
    baselines.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    baselines.add_argument("--features", type=Path, required=True, help="Thư mục bundle feature/window đã kiểm tra")
    baselines.add_argument("--report", type=Path, help="JSON báo cáo; mặc định reports/baseline_run.json")
    train_lstm = commands.add_parser("train-lstm", help="Train LSTM Autoencoder CPU từ feature bundle")
    train_lstm.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    train_lstm.add_argument("--features", type=Path, required=True, help="Thư mục bundle feature/window đã kiểm tra")
    train_lstm.add_argument("--report", type=Path, help="JSON báo cáo; mặc định reports/lstm_training.json")
    score_lstm = commands.add_parser("score-lstm", help="Chấm LSTM, hiệu chỉnh threshold và tạo alert từ artifact bất biến")
    score_lstm.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    score_lstm.add_argument("--features", type=Path, required=True, help="Thư mục bundle feature/window đã kiểm tra")
    score_lstm.add_argument("--lstm", type=Path, required=True, help="Thư mục bundle LSTM đã train")
    score_lstm.add_argument("--report", type=Path, help="JSON báo cáo; mặc định reports/lstm_scoring.json")
    help_command = commands.add_parser("help", help="Hiển thị trợ giúp cho CLI hoặc một command")
    help_command.add_argument("topic", nargs="?", choices=["fetch", "validate", "features", "baselines", "train-lstm", "score-lstm"])
    return parser


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.command == "help":
        if args.topic:
            parser.parse_args([args.topic, "--help"])
        else:
            parser.print_help()
        return 0
    try:
        settings = load_config(args.config)
        if args.command == "fetch":
            if args.input:
                path = import_bar_file(settings, args.input)
            elif args.network:
                path = fetch_network_snapshot(settings)
            else:
                path = collect_probe_snapshot(settings)
            print(json.dumps({"status": "RAW_SNAPSHOT_STORED", "manifest": str(path),
                "acquisition": "network" if args.network else "local",
                "source_confirmations_pending": pending_confirmations(settings),
                "normalized_by_this_command": False, "training_ready": False}, ensure_ascii=False, indent=2))
            return 0
        default_report = {"features": "feature_preparation.json", "baselines": "baseline_run.json",
                          "train-lstm": "lstm_training.json", "score-lstm": "lstm_scoring.json",
                          "validate": "contract_validation.json"}[args.command]
        destination = args.report.resolve() if args.report else project_path(settings, "report_dir") / default_report
        protected = [settings["_config_path"]]
        if getattr(args, "input", None):
            protected.append(args.input.resolve())
        if getattr(args, "snapshot", None):
            protected.append(args.snapshot.resolve())
        if getattr(args, "features", None):
            protected.append(args.features.resolve())
        if getattr(args, "lstm", None):
            protected.append(args.lstm.resolve())
        if destination in protected or any(destination.is_relative_to(project_path(settings, key)) for key in ["raw_dir", "processed_dir"]):
            raise DataContractError("A mutable report cannot overwrite config, input, raw or processed artifacts")
        artifact_root = (settings["_root"] / settings["features"].get("artifact_dir", "artifacts/features")).resolve()
        if destination.is_relative_to(artifact_root):
            raise DataContractError("A mutable report cannot overwrite feature artifacts")
        baseline_root = (settings["_root"] / "artifacts/baselines").resolve()
        if destination.is_relative_to(baseline_root):
            raise DataContractError("A mutable report cannot overwrite baseline artifacts")
        lstm_root = (settings["_root"] / "artifacts/lstm").resolve()
        if destination.is_relative_to(lstm_root):
            raise DataContractError("A mutable report cannot overwrite LSTM artifacts")
        scoring_root = (settings["_root"] / "artifacts/scoring").resolve()
        if destination.is_relative_to(scoring_root):
            raise DataContractError("A mutable report cannot overwrite scoring artifacts")
        try:
            if args.command == "features":
                report = run_features(settings, args.dataset)
            elif args.command == "baselines":
                report = run_baselines(settings, args.features)
            elif args.command == "train-lstm":
                report = run_lstm_training(settings, args.features)
            elif args.command == "score-lstm":
                report = run_lstm_scoring(settings, args.features, args.lstm)
            else:
                report = run_validation(settings, input_path=args.input, snapshot_path=args.snapshot,
                                        diagnostic=args.diagnostic, as_of=args.as_of)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report = {"status": "INVALID_INPUT", "structural_status": "FAIL", "error": str(exc),
                "accepted_bars": 0, "training_ready": False,
                "source_confirmations_pending": pending_confirmations(settings), "output_dir": None}
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        destination.write_text(content, encoding="utf-8")
        print(content, end="")
        return 0 if report["status"] in {"VALIDATED", "VALIDATED_FILE", "RESEARCH_VALIDATED", "FEATURES_READY",
                                         "FEATURES_READY_RESEARCH", "BASELINES_READY", "BASELINES_READY_RESEARCH",
                                         "LSTM_READY", "LSTM_READY_RESEARCH", "LSTM_SCORES_READY",
                                         "LSTM_SCORES_READY_RESEARCH"} else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Windows pipes can default to cp1252 even when the console is UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
