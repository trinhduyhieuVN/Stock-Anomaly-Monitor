"""Strict JSON contracts and as-of guards for the reduced bar-level upgrade."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path
import re
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import yaml

from stock_monitor.io import DataContractError

MODES = {"SYNTHETIC", "HISTORICAL_REPLAY", "SAVED_TRIAL"}
KINDS = {"normal", "point", "collective", "insufficient_data"}
STAGES = {"initial", "post_event"}
REVIEW_KINDS = {"human", "domain_expert", "ai_assisted"}
LABELS = {"noteworthy", "ordinary", "uncertain", "data_issue"}
LOCAL_ZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def utc_time(value, name="timestamp"):
    if not isinstance(value, str):
        raise DataContractError(f"{name} must be a UTC ISO timestamp string")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataContractError(f"Invalid {name}") from exc
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise DataContractError(f"{name} must explicitly use UTC")
    return stamp.astimezone(timezone.utc)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DataContractError(f"{name} must be nonempty text without surrounding spaces")


def _choice(value, choices, name):
    if not isinstance(value, str) or value not in choices:
        raise DataContractError(f"Unknown {name}")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise DataContractError(f"{name} must be an integer >= {minimum}")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DataContractError(f"{name} must be finite numeric data")


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DataContractError(f"{name} must be a lowercase SHA256")


def _fields(record, required, optional=()):
    if not isinstance(record, dict) or set(record) - set(required) - set(optional) or set(required) - set(record):
        raise DataContractError("Missing or unknown schema fields")


def _base(record, required, optional=()):
    _fields(record, required, optional)
    if type(record.get("schema_version")) is not int or record["schema_version"] != 2:
        raise DataContractError("Expected schema_version 2")
    if record.get("event_unit") != "bar_5m":
        raise DataContractError("Only bar_5m is supported; ticks require a separate contract")
    _choice(record.get("source_mode"), MODES, "source_mode; this upgrade does not certify LIVE")
    _text(record["symbol"], "symbol")
    if record["symbol"] not in {"FPT", "HPG", "VCB"}:
        raise DataContractError("Unsupported symbol")


def _visible(available, as_of):
    if as_of is not None and available > utc_time(as_of, "query as_of"):
        raise DataContractError("Record is not available at query as_of")


def _session_bar(stamp, session_id):
    try:
        session = date.fromisoformat(session_id)
    except (TypeError, ValueError) as exc:
        raise DataContractError("Invalid session_id") from exc
    if stamp.astimezone(LOCAL_ZONE).date() != session:
        raise DataContractError("Cross-session interval")
    if stamp.second or stamp.microsecond or stamp.minute % 5:
        raise DataContractError("Expected a closed five-minute bar boundary")


DECISION_FIELDS = {"schema_version", "event_unit", "symbol", "session_id", "segment_id",
                   "version", "event_start", "event_end", "detected_at", "available_at", "as_of",
                   "stage", "kind", "model_run", "model_sha256", "data_sha256", "data_revision",
                   "source_mode", "reason_codes", "score", "threshold", "decision_method"}


def _decision(record, id_field, as_of):
    _base(record, DECISION_FIELDS | {id_field}, {"post_evidence"})
    for field in (id_field, "segment_id", "model_run", "data_revision"):
        _text(record[field], field)
    _integer(record["version"], "version")
    _choice(record["kind"], KINDS, "kind")
    _choice(record["stage"], STAGES, "stage")
    _choice(record["decision_method"], {"baseline", "heuristic", "CAPA"}, "decision_method")
    if not isinstance(record["reason_codes"], list) or not record["reason_codes"]:
        raise DataContractError("reason_codes must be nonempty")
    for reason in record["reason_codes"]:
        _text(reason, "reason_code")
    for field in ("model_sha256", "data_sha256"):
        _hash(record[field], field)
    times = {field: utc_time(record[field], field) for field in
             ("event_start", "event_end", "as_of", "detected_at", "available_at")}
    if not (times["event_start"] <= times["event_end"] <= times["as_of"] <=
            times["detected_at"] <= times["available_at"]):
        raise DataContractError("Event time ordering exposes unavailable or future evidence")
    for name in ("event_start", "event_end"):
        _session_bar(times[name], record["session_id"])
    if record["kind"] == "point" and times["event_start"] != times["event_end"]:
        raise DataContractError("Point events must contain exactly one bar")
    if record["kind"] == "collective" and times["event_start"] == times["event_end"]:
        raise DataContractError("Collective events need multiple bars")
    if record["kind"] == "insufficient_data":
        if record["score"] is not None or record["threshold"] is not None:
            raise DataContractError("insufficient_data cannot masquerade as a numeric normal score")
    else:
        _number(record["score"], "score")
        _number(record["threshold"], "threshold")
    if record["stage"] == "initial" and "post_evidence" in record:
        raise DataContractError("Initial decisions cannot contain post-event outcomes")
    if record["stage"] == "post_event":
        if "post_evidence" not in record:
            raise DataContractError("Post-event decisions require mature evidence")
        _post_evidence(record["post_evidence"], times, record["session_id"])
    _visible(times["available_at"], as_of)
    return deepcopy(record)


def _post_evidence(evidence, times, session_id):
    _fields(evidence, {"horizon_bars", "forecast_created_at", "target_bar_end", "outcome_available_at",
                       "prediction", "realized_return", "residual", "forecast_sha256"})
    if type(evidence["horizon_bars"]) is not int or evidence["horizon_bars"] not in {1, 3}:
        raise DataContractError("Unsupported forecast horizon")
    target = utc_time(evidence["target_bar_end"], "target_bar_end")
    _session_bar(target, session_id)
    if target != times["event_end"] + timedelta(minutes=5 * evidence["horizon_bars"]):
        raise DataContractError("Target does not match the event horizon")
    forecast = utc_time(evidence["forecast_created_at"], "forecast_created_at")
    outcome = utc_time(evidence["outcome_available_at"], "outcome_available_at")
    if not times["event_end"] <= forecast < target <= outcome <= times["as_of"]:
        raise DataContractError("Forecast or outcome is not causal/mature")
    for name in ("prediction", "realized_return", "residual"):
        _number(evidence[name], name)
    if not math.isclose(evidence["realized_return"] - evidence["prediction"], evidence["residual"],
                        rel_tol=1e-9, abs_tol=1e-12):
        raise DataContractError("Residual does not match forecast and outcome")
    _hash(evidence["forecast_sha256"], "forecast_sha256")


def validate_event(record, *, as_of=None):
    return _decision(record, "event_id", as_of)


def validate_candidate(record, *, as_of=None):
    result = _decision(record, "candidate_id", as_of)
    if result["stage"] != "initial" or result["kind"] not in {"point", "collective"}:
        raise DataContractError("Candidates must be initial point or collective decisions")
    return result


def validate_event_versions(records, *, as_of=None):
    seen, identities, results = set(), {}, []
    for record in records:
        value = validate_event(record, as_of=as_of)
        key = (value["event_id"], value["version"])
        identity = (value["symbol"], value["session_id"], value["source_mode"], value["model_run"])
        if key in seen or (value["event_id"] in identities and identities[value["event_id"]] != identity):
            raise DataContractError("Duplicate version or conflicting event identity")
        seen.add(key)
        identities[value["event_id"]] = identity
        results.append(value)
    return results


def validate_news(record, *, as_of=None):
    _base(record, {"schema_version", "event_unit", "symbol", "source_mode", "news_id", "title", "url",
                   "published_at", "known_at", "available_at", "time_basis"})
    for key in ("news_id", "title", "url"):
        _text(record[key], key)
    url = urlsplit(record["url"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise DataContractError("News URL must be HTTP(S) without credentials")
    _choice(record["time_basis"], {"observed", "retrospective", "synthetic"}, "news time_basis")
    published, known, available = (utc_time(record[k], k) for k in ("published_at", "known_at", "available_at"))
    if not published <= known <= available:
        raise DataContractError("News cannot be known before publication or available before observation")
    if record["source_mode"] == "SYNTHETIC" and record["time_basis"] != "synthetic":
        raise DataContractError("Synthetic news must declare synthetic time_basis")
    _visible(available, as_of)
    return deepcopy(record)


def validate_review(record, *, event=None, as_of=None):
    _fields(record, {"schema_version", "review_id", "event_id", "event_version", "reviewer",
                     "reviewer_kind", "label", "reason", "reviewed_at", "available_at", "reference_url"})
    if type(record["schema_version"]) is not int or record["schema_version"] != 2:
        raise DataContractError("Expected schema_version 2")
    for key in ("review_id", "event_id", "reviewer", "reason"):
        _text(record[key], key)
    _integer(record["event_version"], "event_version")
    _choice(record["reviewer_kind"], REVIEW_KINDS, "reviewer_kind")
    _choice(record["label"], LABELS, "label")
    if not isinstance(record["reference_url"], str):
        raise DataContractError("reference_url must be text, possibly empty")
    reviewed, available = (utc_time(record[k], k) for k in ("reviewed_at", "available_at"))
    if reviewed > available:
        raise DataContractError("Review availability precedes review")
    if event is not None:
        value = validate_event(event)
        if (record["event_id"], record["event_version"]) != (value["event_id"], value["version"]):
            raise DataContractError("Review refers to a different event/version")
        if reviewed < utc_time(value["available_at"]):
            raise DataContractError("Cannot review an unavailable event")
    _visible(available, as_of)
    return deepcopy(record)


def load_upgrade_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict) or type(settings.get("schema_version")) is not int or settings["schema_version"] != 2:
        raise DataContractError("Expected upgrade schema_version 2")
    expected = {"event_unit": "bar_5m", "interval_minutes": 5, "timestamp_timezone": "UTC",
                "timestamp_convention": "bar_end", "scope": "reduced_bar_research"}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise DataContractError("Unsupported event unit/time contract")
    try:
        modes = settings["source_modes"]
        if not isinstance(modes, list) or len(modes) != len(MODES) or set(modes) != MODES:
            raise DataContractError("Unsupported source modes")
        if settings["symbols"] != ["FPT", "HPG", "VCB"]:
            raise DataContractError("Unsupported symbols")
        if settings["segment"]["lengths"] != [3, 6, 12] or settings["forecast"]["horizons"] != [1, 3]:
            raise DataContractError("Unsupported segment lengths/forecast horizons")
        if settings["context"]["fallback_order"] != ["context", "symbol", "global", "unavailable"]:
            raise DataContractError("Unsupported context fallback")
        if settings["context"]["information_boundary"] != "strictly_before_current_bar":
            raise DataContractError("Context must precede current bar")
        _integer(settings["context"]["minimum_reference_samples"], "minimum_reference_samples")
        q = settings["context"]["quantile"]
        _number(q, "quantile")
        if not 0 < q < 1:
            raise DataContractError("Invalid quantile")
        if settings["source_policy"] != {"real_rights_status": "pending", "real_distributable": False,
                                         "tick_data_available": False, "book_data_available": False}:
            raise DataContractError("Source policy cannot silently upgrade rights or tick availability")
        if (type(settings["budget"]["provider_requests"]) is not int or
                settings["budget"]["provider_requests"] != 0 or
                settings["budget"]["paid_services"] is not False or
                settings["budget"]["new_deep_models"] is not False):
            raise DataContractError("This upgrade is offline without paid services")
        groups = [settings["synthetic"][name] for name in
                  ("development_seeds", "calibration_seeds", "holdout_seeds")]
        seeds = [seed for group in groups for seed in group]
        if any(not isinstance(group, list) or not group for group in groups) or len(seeds) != len(set(seeds)):
            raise DataContractError("Synthetic seed partitions must be nonempty and disjoint")
        for seed in seeds:
            _integer(seed, "seed", 0)
        _integer(settings["synthetic"]["sessions"], "sessions")
        _integer(settings["synthetic"]["warmup_sessions"], "warmup_sessions")
        if settings["synthetic"]["warmup_sessions"] >= settings["synthetic"]["sessions"]:
            raise DataContractError("Warmup must leave evaluation sessions")
    except (KeyError, TypeError) as exc:
        raise DataContractError("Incomplete upgrade configuration") from exc
    return settings
