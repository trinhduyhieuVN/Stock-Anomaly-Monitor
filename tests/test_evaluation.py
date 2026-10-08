"""Evaluation tests use explicit labels and never derive truth from scores."""

import numpy as np
import pandas as pd
import pytest

from stock_monitor.evaluation import (MODELS, REVIEW_DATES, alert_summary, align_predictions, binary_metrics,
                                       descriptive_tables,
                                       evaluate_reviewed, inject_synthetic_anomalies,
                                      make_review_context, make_review_packet, make_review_template,
                                      validate_review_batch)
from stock_monitor.io import DataContractError


def score_frames():
    timestamps = pd.to_datetime([f"{date}T02:30:00Z" for date in REVIEW_DATES])
    keys = {"window_id": range(5), "symbol": ["FPT"] * 5, "session_id": REVIEW_DATES,
            "segment_id": ["morning"] * 5, "split": ["test"] * 5,
            "timestamp": timestamps, "window_start": timestamps - pd.Timedelta(minutes=25)}
    baseline = pd.DataFrame(keys)
    baseline["robust_zscore"] = [0., 3., 0., 3., 0.]
    baseline["isolation_forest"] = [0., 0., 3., 3., 0.]
    for model in MODELS[:2]:
        baseline[f"{model}_threshold"] = 2.
        baseline[f"{model}_alert"] = baseline[model].gt(2.)
    lstm = pd.DataFrame(keys)
    lstm["score"] = [0., 3., 3., 0., 0.]
    lstm["threshold"] = 2.
    lstm["alert"] = lstm.score.gt(2.)
    return baseline, lstm


def features():
    baseline, _ = score_frames()
    rows = []
    for endpoint in baseline.timestamp:
        for timestamp in pd.date_range(endpoint - pd.Timedelta(minutes=25), endpoint, freq="5min"):
            rows.append({"symbol": "FPT", "timestamp": timestamp, "segment_id": "morning"})
    frame = pd.DataFrame(rows)
    for column, value in {"open": 100., "high": 102., "low": 99., "close": 101., "volume": 1000,
                          "slot": "morning:09:25", "quality_status": "ok"}.items():
        frame[column] = value
    for column in ("log_return", "range_ratio", "body_ratio", "volume_surprise"):
        frame[column] = 0.
    frame["session_id"] = frame.timestamp.dt.strftime("%Y-%m-%d")
    frame["feature_eligible"] = True
    frame["feature_reasons"] = ""
    return frame


def test_alignment_requires_identical_test_timestamps_and_dates():
    baseline, lstm = score_frames()
    aligned = align_predictions(baseline, lstm)
    assert len(aligned) == 5 and aligned.split.eq("test").all()
    shifted = lstm.copy()
    shifted.loc[0, "timestamp"] += pd.Timedelta(minutes=5)
    with pytest.raises(DataContractError, match="identical"):
        align_predictions(baseline, shifted)
    changed = lstm.copy()
    changed.loc[0, "split"] = "train"
    with pytest.raises(DataContractError, match="test windows"):
        align_predictions(baseline, changed)


def test_review_template_is_full_and_blinded():
    baseline, lstm = score_frames()
    template = make_review_template(align_predictions(baseline, lstm), features())
    assert len(template) == 5 and set(template.session_id) == set(REVIEW_DATES)
    assert template.review_id.is_unique
    assert not any(token in column for column in template for token in ("model", "score", "threshold", "alert"))
    assert template.label.eq("").all() and template.reviewer_kind.eq("").all()


def test_review_context_has_six_causal_contiguous_bars_without_outputs():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    context = make_review_context(predictions, features())
    assert len(context) == 30 and context.groupby("review_id").size().eq(6).all()
    assert context.groupby("review_id").focus_bar.sum().eq(1).all()
    assert not any(token in column for column in context for token in ("model", "score", "threshold", "alert"))
    with pytest.raises(DataContractError, match="six consecutive"):
        make_review_context(predictions, features().drop(index=2))


def test_review_packet_is_one_row_per_endpoint_and_raw_ohlcv_only():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    template = make_review_template(predictions, features())
    context = make_review_context(predictions, features())
    packet = make_review_packet(template, context)
    assert len(packet) == len(template) == 5 and packet.review_id.is_unique
    assert packet.bar_times.map(lambda value: len(value.split("|"))).eq(6).all()
    assert packet.close_path.map(lambda value: len(value.split("|"))).eq(6).all()
    assert not any(token in column for column in packet for token in
                   ("model", "score", "threshold", "alert", "log_return", "volume_surprise"))
    with pytest.raises(DataContractError, match="six-bar"):
        make_review_packet(template, context.iloc[:-1])


def test_review_batch_validation_requires_exact_blinded_packet_rows():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    template = make_review_template(predictions, features())
    packet = make_review_packet(template, make_review_context(predictions, features()))
    reviewed = template[["review_id", "symbol", "session_id", "timestamp"]].iloc[[0]].copy()
    reviewed[["label", "reason", "reviewer", "reviewer_kind", "reference_url",
              "reference_published_at", "reviewed_at"]] = [
        "ordinary", "explicit OHLCV review", "tester", "ai_assisted", "", "", "2026-10-04T00:00:00Z"]
    validated = validate_review_batch(packet, reviewed, REVIEW_DATES[0], "FPT")
    assert len(validated) == 1 and validated.label.eq("ordinary").all()
    with pytest.raises(DataContractError, match="exactly once"):
        validate_review_batch(packet, pd.concat([reviewed, reviewed]), REVIEW_DATES[0], "FPT")


def test_review_batch_validation_rejects_model_leaks_and_bad_identity():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    template = make_review_template(predictions, features())
    packet = make_review_packet(template, make_review_context(predictions, features()))
    reviewed = template[["review_id", "symbol", "session_id", "timestamp"]].iloc[[0]].copy()
    reviewed[["label", "reason", "reviewer", "reviewer_kind", "reference_url",
              "reference_published_at", "reviewed_at"]] = [
        "ordinary", "explicit OHLCV review", "tester", "ai_assisted", "", "", "2026-10-04T00:00:00Z"]
    leaked = reviewed.assign(model_score=1.0)
    with pytest.raises(DataContractError, match="leaked"):
        validate_review_batch(packet, leaked, REVIEW_DATES[0], "FPT")
    changed = reviewed.assign(symbol="VCB")
    with pytest.raises(DataContractError, match="identity"):
        validate_review_batch(packet, changed, REVIEW_DATES[0], "FPT")


def test_incomplete_review_returns_unavailable_without_metrics():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    reviewed = make_review_template(predictions, features()).iloc[:2].copy()
    reviewed.loc[:, ["label", "reason", "reviewer", "reviewer_kind", "reviewed_at"]] = [
        "noteworthy", "explicit fixture", "tester", "human", "2026-10-04T00:00:00Z"]
    summary, metrics = evaluate_reviewed(predictions, reviewed)
    assert summary["status"] == "METRICS_UNAVAILABLE" and metrics.empty


def test_complete_review_reports_counts_coverage_and_zero_denominators():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    reviewed = make_review_template(predictions, features())
    reviewed["label"] = ["ordinary", "noteworthy", "uncertain", "data_issue", "ordinary"]
    reviewed[["reason", "reviewer", "reviewer_kind", "reviewed_at"]] = [
        "explicit fixture", "tester", "human", "2026-10-04T00:00:00Z"]
    summary, metrics = evaluate_reviewed(predictions, reviewed)
    assert summary["status"] == "REVIEWED_METRICS_READY" and summary["coverage"] == 3 / 5
    assert summary["excluded_uncertain"] == summary["excluded_data_issue"] == 1
    robust = metrics.loc[(metrics.model.eq("robust_zscore")) & metrics.symbol.eq("pooled")].iloc[0]
    assert (robust.tp, robust.fp, robust.fn) == (1, 0, 0)
    assert robust.precision == robust.recall == robust.f1 == 1.
    assert len(alert_summary(predictions)) == len(MODELS) * len(REVIEW_DATES)


def test_unknown_label_and_extra_row_are_rejected():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    reviewed = make_review_template(predictions, features())
    reviewed.loc[0, "label"] = "fraud"
    with pytest.raises(DataContractError, match="Unknown"):
        evaluate_reviewed(predictions, reviewed)
    reviewed = make_review_template(predictions, features())
    extra = reviewed.iloc[[0]].copy()
    extra["review_id"] = "VCB|2026-10-01T00:00:00+00:00"
    with pytest.raises(DataContractError, match="outside"):
        evaluate_reviewed(predictions, pd.concat([reviewed, extra], ignore_index=True))
    reviewed = make_review_template(predictions, features())
    reviewed["label"] = "ordinary"
    reviewed[["reason", "reviewer", "reviewer_kind", "reviewed_at"]] = [
        "explicit fixture", "tester", "automated", "2026-10-04T00:00:00Z"]
    with pytest.raises(DataContractError, match="reviewer_kind"):
        evaluate_reviewed(predictions, reviewed)


def test_synthetic_injection_is_deterministic_test_only_and_does_not_mutate_input():
    windows = np.zeros((15, 6, 4), dtype="float32")
    original = windows.copy()
    index = pd.DataFrame({"window_id": range(15), "symbol": ["FPT"] * 5 + ["HPG"] * 5 + ["VCB"] * 5,
                          "split": ["train"] * 3 + ["test"] * 12,
                          "timestamp": pd.date_range("2026-09-01T02:30Z", periods=15, freq="5min")})
    order = ["log_return", "range_ratio", "body_ratio", "volume_surprise"]
    first, first_index, first_spec = inject_synthetic_anomalies(windows, index, order, seed=42, fraction=.25)
    second, second_index, second_spec = inject_synthetic_anomalies(windows, index, order, seed=42, fraction=.25)
    np.testing.assert_array_equal(windows, original)
    np.testing.assert_array_equal(first, second)
    pd.testing.assert_frame_equal(first_index, second_index)
    assert first_spec == second_spec and first_index.split.eq("test").all()
    assert first_index.synthetic_label.sum() == len(first_spec["injections"])
    changed = first - original[index.split.eq("test").to_numpy()]
    assert not changed[:, :-1].any() and not changed[:, -1, 1:3].any()


def test_binary_metrics_uses_explicit_labels_and_reports_zero_denominator():
    frame = pd.DataFrame({"symbol": ["FPT"] * 4, "synthetic_label": [True, True, False, False],
                          "robust_zscore_alert": [True, False, True, False],
                          "isolation_forest_alert": [False] * 4, "lstm_alert": [True, True, False, False]})
    metrics = binary_metrics(frame)
    robust = metrics.loc[(metrics.model.eq("robust_zscore")) & metrics.symbol.eq("pooled")].iloc[0]
    assert (robust.tp, robust.fp, robust.fn) == (1, 1, 1) and robust.f1 == .5
    forest = metrics.loc[(metrics.model.eq("isolation_forest")) & metrics.symbol.eq("pooled")].iloc[0]
    assert pd.isna(forest.precision) and forest.recall == 0. and pd.isna(forest.f1)


def test_descriptive_tables_keep_alerts_segments_slots_and_coverage_separate():
    baseline, lstm = score_frames()
    predictions = align_predictions(baseline, lstm)
    tables = descriptive_tables(predictions, features())
    assert len(tables["alerts"]) == len(MODELS) * len(REVIEW_DATES)
    assert len(tables["segments"]) == len(MODELS) * len(REVIEW_DATES)
    assert len(tables["slots"]) == len(MODELS)
    assert len(tables["coverage"]) == len(REVIEW_DATES)
    assert tables["coverage"].bars.eq(6).all() and tables["coverage"].scored_windows.eq(1).all()
    assert tables["segments"].alert_segments.le(tables["segments"].alert_points).all()
