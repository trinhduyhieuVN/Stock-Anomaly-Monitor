"""Evaluation contracts that keep manual review blinded and metrics honest."""

import numpy as np
import pandas as pd

from .alerts import group_consecutive_alerts
from .io import DataContractError

REVIEW_DATES = ("2026-08-28", "2026-09-09", "2026-09-16", "2026-09-23", "2026-10-01")
MODELS = ("robust_zscore", "isolation_forest", "lstm")
LABELS = ("noteworthy", "ordinary", "uncertain", "data_issue")
REVIEWER_KINDS = ("human", "domain_expert", "ai_assisted")
KEYS = ("window_id", "symbol", "session_id", "segment_id", "split", "timestamp", "window_start")


def align_predictions(baseline_scores, lstm_scores, review_dates=REVIEW_DATES):
    """Return three frozen model predictions on one exact test key set."""
    required_baseline = set(KEYS) | {f"{model}_{kind}" for model in MODELS[:2] for kind in ("score", "threshold", "alert")}
    baseline = baseline_scores.rename(columns={model: f"{model}_score" for model in MODELS[:2]})
    required_lstm = set(KEYS) | {"score", "threshold", "alert"}
    if not required_baseline.issubset(baseline.columns) or not required_lstm.issubset(lstm_scores.columns):
        raise DataContractError("Evaluation score files are missing required prediction columns")
    baseline = baseline.loc[baseline.session_id.isin(review_dates)].copy()
    lstm = lstm_scores.loc[lstm_scores.session_id.isin(review_dates)].copy()
    for name, frame in (("baseline", baseline), ("LSTM", lstm)):
        if frame.duplicated(["symbol", "timestamp"]).any() or not frame.split.eq("test").all():
            raise DataContractError(f"{name} evaluation rows must be unique test windows")
    if set(baseline.session_id.unique()) != set(review_dates) or set(lstm.session_id.unique()) != set(review_dates):
        raise DataContractError("Evaluation requires every preselected review date")
    right = lstm[list(KEYS) + ["score", "threshold", "alert"]].rename(columns={
        "score": "lstm_score", "threshold": "lstm_threshold", "alert": "lstm_alert"})
    merged = baseline[list(KEYS) + [f"{model}_{kind}" for model in MODELS[:2]
                                              for kind in ("score", "threshold", "alert")]].merge(
        right, on=list(KEYS), how="outer", indicator=True, validate="one_to_one")
    if not merged._merge.eq("both").all():
        raise DataContractError("Models must be evaluated on identical symbol/timestamp windows")
    merged = merged.drop(columns="_merge").sort_values(["session_id", "symbol", "timestamp"]).reset_index(drop=True)
    for model in MODELS:
        if not np.isfinite(merged[[f"{model}_score", f"{model}_threshold"]].to_numpy(dtype=float)).all():
            raise DataContractError("Evaluation predictions must be finite")
        merged[f"{model}_alert"] = merged[f"{model}_alert"].astype(bool)
    return merged


def make_review_template(predictions, features):
    """Build the full five-day review sheet without model identity or output."""
    context = features[["symbol", "timestamp", "open", "high", "low", "close", "volume",
                        "slot", "quality_status"]]
    template = predictions[list(KEYS)].merge(context, on=["symbol", "timestamp"],
                                               validate="one_to_one")
    template.insert(0, "review_id", template.symbol + "|" + template.timestamp.map(pd.Timestamp.isoformat))
    for column in ("label", "reason", "reviewer", "reviewer_kind", "reference_url",
                   "reference_published_at", "reviewed_at"):
        template[column] = ""
    forbidden = ("model", "score", "threshold", "alert")
    if any(any(token in column for token in forbidden) for column in template.columns):
        raise DataContractError("Blinded review template leaked model output")
    if template.review_id.duplicated().any() or set(template.session_id.unique()) != set(REVIEW_DATES):
        raise DataContractError("Review template must cover each selected window exactly once")
    return template


def make_review_context(predictions, features):
    """Return exactly six causal OHLCV/feature bars for each blinded row."""
    feature_columns = ["log_return", "range_ratio", "body_ratio", "volume_surprise"]
    required = {"symbol", "segment_id", "timestamp", "open", "high", "low", "close", "volume",
                "slot", "quality_status"} | set(feature_columns)
    if not required.issubset(features.columns):
        raise DataContractError("Review context features are missing required columns")
    records = []
    for row in predictions.itertuples(index=False):
        window = features.loc[features.symbol.eq(row.symbol) & features.segment_id.eq(row.segment_id) &
                              features.timestamp.between(row.window_start, row.timestamp)].sort_values("timestamp")
        if (len(window) != 6 or window.timestamp.iloc[-1] != row.timestamp or
                not window.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all()):
            raise DataContractError("Every review row requires six consecutive past bars in one segment")
        review_id = row.symbol + "|" + pd.Timestamp(row.timestamp).isoformat()
        for position, bar in enumerate(window.itertuples(index=False), start=1):
            record = {"review_id": review_id, "context_position": position, "focus_bar": position == 6,
                      "symbol": row.symbol, "session_id": row.session_id, "segment_id": row.segment_id,
                      "timestamp": bar.timestamp, "open": bar.open, "high": bar.high, "low": bar.low,
                      "close": bar.close, "volume": bar.volume, "slot": bar.slot,
                      "quality_status": bar.quality_status}
            record.update({name: getattr(bar, name) for name in feature_columns})
            records.append(record)
    context = pd.DataFrame(records)
    if len(context) != len(predictions) * 6 or context.timestamp.gt(
            context.review_id.map(dict(zip(predictions.symbol + "|" + predictions.timestamp.map(pd.Timestamp.isoformat),
                                           predictions.timestamp)))).any():
        raise DataContractError("Review context contains an invalid or future bar")
    return context


def make_review_packet(template, context):
    """Compress six raw OHLCV bars for blinded row-by-row review."""
    required_template = {"review_id", "symbol", "session_id", "timestamp", "segment_id", "slot", "quality_status"}
    required_context = {"review_id", "context_position", "focus_bar", "timestamp", "open", "high", "low", "close", "volume"}
    if not required_template.issubset(template.columns) or not required_context.issubset(context.columns):
        raise DataContractError("Blinded review packet inputs are incomplete")
    rows = []
    for review_id, window in context.groupby("review_id", sort=False):
        window = window.sort_values("context_position")
        if (len(window) != 6 or window.context_position.tolist() != list(range(1, 7)) or
                window.focus_bar.astype(bool).sum() != 1 or not bool(window.iloc[-1].focus_bar)):
            raise DataContractError("Review packet requires one ordered six-bar context per review row")
        endpoint = template.loc[template.review_id.eq(review_id)]
        if len(endpoint) != 1 or pd.Timestamp(window.timestamp.iloc[-1]) != pd.Timestamp(endpoint.timestamp.iloc[0]):
            raise DataContractError("Review packet endpoint does not match its context")
        endpoint = endpoint.iloc[0]
        prior_volume_median = float(window.volume.iloc[:-1].median())
        focus_volume = float(window.volume.iloc[-1])
        rows.append({"review_id": review_id, "symbol": endpoint.symbol, "session_id": endpoint.session_id,
                     "timestamp": endpoint.timestamp, "segment_id": endpoint.segment_id, "slot": endpoint.slot,
                     "quality_status": endpoint.quality_status,
                     "bar_times": "|".join(pd.to_datetime(window.timestamp, utc=True).dt.strftime("%H:%M")),
                     "open_path": "|".join(window.open.map(lambda value: f"{value:.6g}")),
                     "high_path": "|".join(window.high.map(lambda value: f"{value:.6g}")),
                     "low_path": "|".join(window.low.map(lambda value: f"{value:.6g}")),
                     "close_path": "|".join(window.close.map(lambda value: f"{value:.6g}")),
                     "volume_path": "|".join(window.volume.map(lambda value: f"{value:.6g}")),
                     "window_return": float(window.close.iloc[-1] / window.open.iloc[0] - 1),
                     "window_high_low_range": float(window.high.max() / window.low.min() - 1),
                     "focus_volume_vs_prior_median": focus_volume / prior_volume_median if prior_volume_median else None})
    packet = pd.DataFrame(rows).sort_values(["session_id", "symbol", "timestamp"]).reset_index(drop=True)
    forbidden = ("model", "score", "threshold", "alert", "log_return", "volume_surprise")
    if len(packet) != len(template) or packet.review_id.duplicated().any() or any(
            any(token in column for token in forbidden) for column in packet.columns):
        raise DataContractError("Blinded review packet leaked model-derived information or lost rows")
    return packet


def validate_review_batch(packet, reviewed, session_id, symbol):
    """Validate one completed blinded-review batch without loading model outputs."""
    required = {"review_id", "symbol", "session_id", "timestamp", "label", "reason", "reviewer",
                "reviewer_kind", "reference_url", "reference_published_at", "reviewed_at"}
    if not required.issubset(reviewed.columns):
        raise DataContractError("Review batch is missing required audit columns")
    forbidden = ("model", "score", "threshold", "alert")
    if any(any(token in column for token in forbidden) for column in reviewed.columns):
        raise DataContractError("Review batch leaked model-derived information")

    expected = packet.loc[packet.session_id.astype(str).eq(str(session_id)) & packet.symbol.eq(symbol),
                          ["review_id", "symbol", "session_id", "timestamp"]].copy()
    labels = reviewed.copy()
    expected["timestamp"] = pd.to_datetime(expected.timestamp, utc=True, errors="coerce")
    labels["timestamp"] = pd.to_datetime(labels.timestamp, utc=True, errors="coerce")
    labels["reviewed_at"] = pd.to_datetime(labels.reviewed_at, utc=True, errors="coerce")
    if expected.empty or labels.timestamp.isna().any() or labels.reviewed_at.isna().any():
        raise DataContractError("Review batch requires valid expected and audit timestamps")
    if labels.review_id.duplicated().any() or set(labels.review_id) != set(expected.review_id) or len(labels) != len(expected):
        raise DataContractError("Review batch must cover every expected review_id exactly once")
    joined = expected.merge(labels, on="review_id", suffixes=("_expected", ""), validate="one_to_one")
    if (not joined.symbol.eq(joined.symbol_expected).all() or
            not joined.session_id.astype(str).eq(joined.session_id_expected.astype(str)).all() or
            not joined.timestamp.eq(joined.timestamp_expected).all()):
        raise DataContractError("Review batch identity fields do not match the blinded packet")
    if set(labels.label) - set(LABELS):
        raise DataContractError("Review batch contains an unknown label")
    if set(labels.reviewer_kind) - set(REVIEWER_KINDS):
        raise DataContractError("Review batch contains an unknown reviewer_kind")
    complete = labels[["label", "reason", "reviewer", "reviewer_kind", "reviewed_at"]].fillna("").astype(
        str).apply(lambda column: column.str.strip().ne("")).all(axis=1)
    if not complete.all():
        raise DataContractError("Review batch has incomplete audit fields")
    reference_url = labels.reference_url.fillna("").astype(str).str.strip().ne("")
    reference_time = labels.reference_published_at.fillna("").astype(str).str.strip().ne("")
    if not reference_url.eq(reference_time).all():
        raise DataContractError("Review references require both URL and publication time")
    return labels.sort_values("timestamp").reset_index(drop=True)


def evaluate_reviewed(predictions, reviewed):
    """Compute metrics only after every expected row has a valid review outcome."""
    required = {"review_id", "symbol", "timestamp", "label", "reason", "reviewer", "reviewer_kind", "reviewed_at"}
    if not required.issubset(reviewed.columns):
        raise DataContractError("Reviewed labels are missing required audit columns")
    labels = reviewed.copy()
    labels["timestamp"] = pd.to_datetime(labels.timestamp, utc=True, errors="coerce")
    if labels.timestamp.isna().any() or labels.review_id.duplicated().any():
        raise DataContractError("Reviewed rows require unique valid timestamps")
    expected = predictions.assign(review_id=predictions.symbol + "|" + predictions.timestamp.map(pd.Timestamp.isoformat))
    expected_ids, supplied_ids = set(expected.review_id), set(labels.review_id)
    if supplied_ids - expected_ids:
        raise DataContractError("Reviewed labels contain rows outside the frozen evaluation subset")
    nonempty = labels.label.fillna("").astype(str).ne("")
    invalid = set(labels.loc[nonempty, "label"]) - set(LABELS)
    if invalid:
        raise DataContractError("Unknown review labels: " + ", ".join(sorted(invalid)))
    complete_fields = ["label", "reason", "reviewer", "reviewer_kind", "reviewed_at"]
    complete_rows = labels[complete_fields].fillna("").astype(str).apply(lambda column: column.str.strip().ne("")).all(axis=1)
    complete = supplied_ids == expected_ids and len(labels) == len(expected) and complete_rows.all()
    if not complete:
        return {"status": "METRICS_UNAVAILABLE", "expected_rows": len(expected),
                "supplied_rows": len(labels), "complete_rows": int(complete_rows.sum()),
                "reason": "Precision/recall/F1 require a complete audited review of the frozen five-day subset."}, pd.DataFrame()

    invalid_kinds = set(labels.reviewer_kind) - set(REVIEWER_KINDS)
    if invalid_kinds:
        raise DataContractError("Unknown reviewer_kind: " + ", ".join(sorted(invalid_kinds)))

    joined = expected.merge(labels[["review_id", "label", "reviewer_kind"]], on="review_id", validate="one_to_one")
    evaluated = joined.loc[joined.label.isin(["noteworthy", "ordinary"])].copy()
    evaluated["target"] = evaluated.label.eq("noteworthy")
    metric_rows = []
    for model in MODELS:
        for symbol, rows in [("pooled", evaluated)] + list(evaluated.groupby("symbol", sort=True)):
            prediction = rows[f"{model}_alert"]
            target = rows.target
            tp = int((prediction & target).sum())
            fp = int((prediction & ~target).sum())
            fn = int((~prediction & target).sum())
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
            metric_rows.append({"model": model, "symbol": symbol, "evaluated_rows": len(rows),
                                "positive_labels": int(target.sum()), "negative_labels": int((~target).sum()),
                                "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
    summary = {"status": "REVIEWED_METRICS_READY", "expected_rows": len(expected), "reviewed_rows": len(joined),
               "evaluated_rows": len(evaluated), "coverage": len(evaluated) / len(joined),
               "excluded_uncertain": int(joined.label.eq("uncertain").sum()),
               "excluded_data_issue": int(joined.label.eq("data_issue").sum()),
               "reviewer_kind_counts": joined.reviewer_kind.value_counts().to_dict(),
               "scope": "complete reviewed five-day subset; not all test dates or the market"}
    return summary, pd.DataFrame(metric_rows)


def alert_summary(predictions):
    rows = []
    for model in MODELS:
        for (symbol, session_id), group in predictions.groupby(["symbol", "session_id"], sort=True):
            rows.append({"model": model, "symbol": symbol, "session_id": session_id,
                         "windows": len(group), "alert_points": int(group[f"{model}_alert"].sum()),
                         "alert_rate": float(group[f"{model}_alert"].mean())})
    return pd.DataFrame(rows)


def descriptive_tables(predictions, features):
    """Summarize real frozen predictions without treating alerts as labels."""
    feature_columns = ["symbol", "session_id", "timestamp", "slot", "feature_eligible", "feature_reasons"]
    if not set(feature_columns).issubset(features.columns):
        raise DataContractError("Descriptive evaluation requires feature eligibility and slot columns")
    endpoints = features[feature_columns].merge(predictions[["symbol", "timestamp"]],
                                                on=["symbol", "timestamp"], validate="one_to_one")
    enriched = predictions.merge(endpoints[["symbol", "timestamp", "slot"]],
                                 on=["symbol", "timestamp"], validate="one_to_one")
    slot_rows, segment_rows = [], []
    for model in MODELS:
        for (symbol, slot), group in enriched.groupby(["symbol", "slot"], sort=True):
            slot_rows.append({"model": model, "symbol": symbol, "slot": slot, "windows": len(group),
                              "alert_points": int(group[f"{model}_alert"].sum()),
                              "alert_rate": float(group[f"{model}_alert"].mean())})
        for (symbol, session_id), group in predictions.groupby(["symbol", "session_id"], sort=True):
            values = group[list(KEYS)].copy()
            values["score"] = group[f"{model}_score"].to_numpy()
            values["threshold"] = group[f"{model}_threshold"].to_numpy()
            values["alert"] = group[f"{model}_alert"].to_numpy()
            values["reason"] = values.alert.map({True: "score_above_frozen_validation_q98", False: ""})
            segments = group_consecutive_alerts(values)
            segment_rows.append({"model": model, "symbol": symbol, "session_id": session_id,
                                 "alert_points": int(values.alert.sum()), "alert_segments": len(segments),
                                 "maximum_segment_bars": int(segments.bars.max()) if len(segments) else 0})
    review_features = features.loc[features.session_id.isin(REVIEW_DATES)].copy()
    review_features = review_features.merge(predictions[["symbol", "timestamp"]].assign(scored=True),
                                            on=["symbol", "timestamp"], how="left", validate="one_to_one")
    coverage_rows = []
    for (symbol, session_id), group in review_features.groupby(["symbol", "session_id"], sort=True):
        coverage_rows.append({"symbol": symbol, "session_id": session_id, "bars": len(group),
                              "feature_eligible_bars": int(group.feature_eligible.sum()),
                              "scored_windows": int(group.scored.fillna(False).sum()),
                              "unscored_bars": int(group.scored.isna().sum()),
                              "feature_ineligible_bars": int((~group.feature_eligible).sum())})
    return {"alerts": alert_summary(predictions), "segments": pd.DataFrame(segment_rows),
            "slots": pd.DataFrame(slot_rows), "coverage": pd.DataFrame(coverage_rows)}


def inject_synthetic_anomalies(windows, index, feature_order, *, seed=42, fraction=0.10, magnitude=8.0):
    """Inject deterministic last-bar feature perturbations into independent test windows."""
    values = np.asarray(windows)
    feature_order = tuple(feature_order)
    if (values.ndim != 3 or len(values) != len(index) or values.shape[2] != len(feature_order) or
            not 0 < fraction < 1 or magnitude <= 0):
        raise DataContractError("Invalid synthetic benchmark inputs")
    if not {"log_return", "volume_surprise"}.issubset(feature_order):
        raise DataContractError("Synthetic benchmark requires log_return and volume_surprise")
    selected_rows = np.flatnonzero(index.split.eq("test").to_numpy())
    synthetic = values[selected_rows].copy()
    synthetic_index = index.iloc[selected_rows].reset_index(drop=True).copy()
    synthetic_index.insert(0, "source_window_id", synthetic_index.window_id.to_numpy())
    synthetic_index["window_id"] = np.arange(len(synthetic_index))
    synthetic_index["synthetic_label"] = False
    rng = np.random.default_rng(seed)
    feature_positions = [feature_order.index("log_return"), feature_order.index("volume_surprise")]
    injections = []
    for symbol, rows in synthetic_index.groupby("symbol", sort=True):
        count = max(1, int(round(len(rows) * fraction)))
        positions = np.sort(rng.choice(rows.index.to_numpy(), size=count, replace=False))
        signs = rng.choice(np.array([-1.0, 1.0]), size=count)
        for position, sign in zip(positions, signs):
            deltas = [float(sign * magnitude), float(sign * magnitude * 1.25)]
            for feature_position, delta in zip(feature_positions, deltas):
                synthetic[position, -1, feature_position] += delta
            synthetic_index.loc[position, "synthetic_label"] = True
            row = synthetic_index.loc[position]
            injections.append({"synthetic_row": int(position), "source_window_id": int(row.source_window_id),
                               "symbol": symbol, "timestamp": row.timestamp.isoformat(),
                               "last_bar_feature_deltas": dict(zip(("log_return", "volume_surprise"), deltas))})
    specification = {"seed": seed, "fraction_per_symbol": fraction, "magnitude": magnitude,
                     "scope": "independent scaled test windows; perturb final bar only",
                     "ground_truth": "synthetic_label is true exactly at injected rows, fixed before scoring",
                     "injections": injections}
    return synthetic, synthetic_index, specification


def binary_metrics(predictions, *, target_column="synthetic_label"):
    """Return TP/FP/FN and P/R/F1 for explicit binary ground truth."""
    required = {"symbol", target_column} | {f"{model}_alert" for model in MODELS}
    if not required.issubset(predictions.columns):
        raise DataContractError("Binary evaluation input is missing labels or model alerts")
    target_values = predictions[target_column]
    if not target_values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataContractError("Binary ground truth must contain only booleans")
    rows = []
    for model in MODELS:
        if not predictions[f"{model}_alert"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise DataContractError("Model alerts must contain only booleans")
        for symbol, group in [("pooled", predictions)] + list(predictions.groupby("symbol", sort=True)):
            target = group[target_column]
            predicted = group[f"{model}_alert"]
            tp = int((target & predicted).sum())
            fp = int((~target & predicted).sum())
            fn = int((target & ~predicted).sum())
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
            rows.append({"model": model, "symbol": symbol, "rows": len(group),
                         "positive_labels": int(target.sum()), "negative_labels": int((~target).sum()),
                         "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
    return pd.DataFrame(rows)
