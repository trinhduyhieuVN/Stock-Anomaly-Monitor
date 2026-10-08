"""Fail-closed preflight for a future bounded live-session trial."""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import load_config, pending_confirmations
from .io import DataContractError
from .polling import (PollNonRetryableError, PollPayload, PollRetryableError,
                      run_bounded_poll_cycle)
from .providers import ProviderError, parse_vci_minutes
from .realtime import verify_frozen_scoring_manifest
from .validation import load_profile


def _trial_context(project_root, config_path):
    root = Path(project_root).resolve()
    if not isinstance(config_path, str) or not config_path:
        raise DataContractError("Trial config path is required")
    config_file = (root / config_path).resolve()
    if not config_file.is_relative_to(root):
        raise DataContractError("Live-trial config must stay inside the project")
    try:
        settings = load_config(config_file)
        if settings["_root"] != root or settings["data"].get("normalization_mode") != "research":
            raise DataContractError("Live-trial preflight requires this project's explicit research profile")
        profile = load_profile(settings)
        context = {
            "config_path": config_file.relative_to(root).as_posix(),
            "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
            "calendar_path": (root / profile["calendar_file"]).resolve().relative_to(root).as_posix(),
            "calendar_sha256": profile["calendar_sha256"],
            "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DataContractError("Trial config/calendar context is invalid") from exc
    return settings, profile, context


def _trial_observation(observed_at_utc, delay, evidence):
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 3600:
        raise DataContractError("provider_delay_seconds must be explicit and bounded")
    if not isinstance(evidence, str) or not evidence.strip():
        raise DataContractError("provider_delay_evidence is required and must not be inferred")
    try:
        observed = pd.Timestamp(observed_at_utc)
    except (TypeError, ValueError) as exc:
        raise DataContractError("observed_at_utc must be a valid UTC timestamp") from exc
    if pd.isna(observed) or observed.tzinfo is None:
        raise DataContractError("observed_at_utc must explicitly include a timezone")
    return observed.tz_convert("UTC")


def _validate_trial_limits(limits):
    """Apply the same finite trial budget at planning and execution boundaries."""
    if not isinstance(limits, dict):
        raise DataContractError("Trial limits must be a mapping")
    for name in ("max_polls", "max_attempts"):
        if type(limits.get(name)) is not int:
            raise DataContractError(f"Trial limits {name} must be an integer")
    bounds = (("poll_interval_seconds", 15, 300), ("max_polls", 1, 12),
              ("max_attempts", 1, 3), ("max_wait_seconds", 0, 10))
    for name, lower, upper in bounds:
        value = limits.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lower <= value <= upper:
            raise DataContractError(f"Trial limits {name} must be bounded within {lower}..{upper}")
    planned = limits["max_polls"] * float(limits["poll_interval_seconds"])
    if planned > 3600:
        raise DataContractError("Live-trial wall-clock plan must not exceed one hour")
    return planned


def preflight_live_trial(project_root, *, config_path="configs/research.yaml",
                         manifest_path="demo/manifest.local.json", observed_at_utc,
                         provider_delay_seconds, provider_delay_evidence,
                         poll_interval_seconds=60, max_polls=6, max_attempts=3,
                         max_wait_seconds=10) -> dict:
    """Validate a finite trial plan without importing or calling a provider transport."""
    root = Path(project_root).resolve()
    settings, profile, context = _trial_context(root, str(config_path))
    observed = _trial_observation(observed_at_utc, provider_delay_seconds, provider_delay_evidence)
    limits = {"poll_interval_seconds": poll_interval_seconds, "max_polls": max_polls,
              "max_attempts": max_attempts, "max_wait_seconds": max_wait_seconds}
    limits["maximum_planned_seconds"] = _validate_trial_limits(limits)
    calendar = profile["calendar"]
    local_day = observed.tz_convert(calendar["timezone"]).date().isoformat()
    frozen = verify_frozen_scoring_manifest(root, manifest_path)
    known = local_day in set(calendar.get("dates", []))
    state = "PREFLIGHT_READY_NO_NETWORK" if known else "BLOCKED_UNKNOWN_CALENDAR"
    return {
        "schema_version": 1,
        "kind": "live_trial_preflight",
        "context": context,
        "state": state,
        "dry_run": True,
        "network_called": False,
        "live_evidence": False,
        "observed_at_utc": observed.tz_convert("UTC").isoformat(),
        "calendar_date": local_day,
        "calendar_date_known": known,
        "calendar_status": calendar.get("status"),
        "source_rights_status": frozen["source_rights_status"],
        "source_confirmations_pending": pending_confirmations(settings),
        "provider_delay_seconds": provider_delay_seconds,
        "provider_delay_evidence": provider_delay_evidence,
        "delay_measured_live": False,
        "symbols": settings["data"]["symbols"],
        "limits": limits,
        "frozen_manifest_sha256": frozen["manifest_sha256"],
        "execution_enabled": False,
        "operator_action": ("Calendar date is verified; keep network execution disabled until the guarded executor is audited."
                            if known else "Record evidence for this market date before any provider call."),
    }


def _vci_poll_once(observed_at_utc, *, transport, timeout_seconds, count_back,
                   max_wait_seconds, clock):
    observed = pd.Timestamp(observed_at_utc).tz_convert("UTC")

    def poll(symbol, attempt):
        query = {"symbols": [symbol], "timeFrame": "ONE_MINUTE",
                 "to": int(observed.timestamp()), "countBack": count_back}
        try:
            status, headers, body = transport(query, timeout_seconds)
        except (TimeoutError, OSError) as exc:
            raise PollRetryableError("transport_error") from exc
        # Capture receipt before header/JSON parsing or downstream store/scoring work.
        received_at_utc = clock() if status == 200 else None
        safe_headers = {str(key).lower(): str(value) for key, value in dict(headers).items()
                        if str(key).lower() in {"retry-after", "content-type", "date"}}
        if status == 200:
            try:
                payload = json.loads(body)
                frame = parse_vci_minutes(payload, symbol)
            except (ValueError, TypeError, DataContractError, ProviderError) as exc:
                raise PollNonRetryableError("BLOCKED_PROVIDER_PAYLOAD", "invalid_200_payload") from exc
            if frame.empty:
                raise PollNonRetryableError("BLOCKED_PROVIDER_PAYLOAD", "empty_200_payload")
            return PollPayload(body, received_at_utc)
        if status in {401, 403}:
            raise PollNonRetryableError("BLOCKED_PROVIDER_ACCESS", f"http_{status}")
        if status == 429:
            try:
                retry_after = float(safe_headers.get("retry-after", ""))
            except ValueError as exc:
                raise PollNonRetryableError("BLOCKED_PROVIDER_QUOTA", "invalid_retry_after") from exc
            if not 0 <= retry_after <= max_wait_seconds:
                raise PollNonRetryableError("BLOCKED_PROVIDER_QUOTA", "retry_after_outside_budget")
            raise PollRetryableError("http_429", retry_after_seconds=retry_after)
        if status in {500, 502, 503, 504}:
            raise PollRetryableError(f"http_{status}")
        raise PollNonRetryableError("BLOCKED_PROVIDER_HTTP", f"http_{status}")

    return poll


def execute_vci_one_cycle(preflight: dict, *, store_root, cycle_id, profile,
                          project_root, manifest_path="demo/manifest.local.json",
                          transport, sleep, readiness_fn, score_fn=None,
                          count_back=500, timeout_seconds=30,
                           clock=lambda: datetime.now(timezone.utc), delay_observations=None):
    """Execute one guarded cycle using only the explicitly injected transport.

    The bounded default includes up to 500 recent minutes so a cold start can
    cover the whole configured 225-minute trading day. Readiness still rejects
    actual gaps or truncated responses; the request size is not a completeness claim.
    """
    if not isinstance(preflight, dict) or preflight.get("state") != "PREFLIGHT_READY_NO_NETWORK":
        raise DataContractError("One-cycle execution requires a successful no-network preflight")
    if preflight.get("network_called") is not False or preflight.get("execution_enabled") is not False:
        raise DataContractError("Preflight must prove that execution was not already enabled")
    if not callable(transport) or not callable(clock):
        raise DataContractError("Executor requires explicit callable transport and clock")
    if isinstance(count_back, bool) or not isinstance(count_back, int) or not 5 <= count_back <= 500:
        raise DataContractError("count_back must stay within 5..500 for one-cycle polling")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 30:
        raise DataContractError("timeout_seconds must stay within 1..30")
    limits = preflight.get("limits", {})
    planned = _validate_trial_limits(limits)
    if limits.get("maximum_planned_seconds") != planned:
        raise DataContractError("Preflight planned duration does not match trial limits")
    expected_manifest = preflight.get("frozen_manifest_sha256")
    if not isinstance(expected_manifest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_manifest):
        raise DataContractError("Preflight requires the frozen manifest SHA256")
    context = preflight.get("context")
    if not isinstance(context, dict):
        raise DataContractError("Preflight config/calendar context is required; rerun preflight")
    settings, active_profile, current_context = _trial_context(project_root, context.get("config_path"))
    if context != current_context or profile != active_profile:
        raise DataContractError("Config/calendar/profile changed since preflight")
    observed = _trial_observation(preflight.get("observed_at_utc"),
                                  preflight.get("provider_delay_seconds"),
                                  preflight.get("provider_delay_evidence"))
    local_day = observed.tz_convert(active_profile["calendar"]["timezone"]).date().isoformat()
    if (preflight.get("calendar_date_known") is not True
            or preflight.get("calendar_date") != local_day
            or local_day not in active_profile["calendar"]["dates"]):
        raise DataContractError("Preflight date must match the current observed calendar")
    symbols = preflight.get("symbols")
    if (not isinstance(symbols, list) or not symbols
            or any(symbol not in settings["data"]["symbols"] for symbol in symbols)):
        raise DataContractError("Preflight symbols must stay within the configured scope")
    if (preflight.get("source_confirmations_pending") != pending_confirmations(settings)
            or preflight.get("source_rights_status") != "pending"):
        raise DataContractError("Preflight source status differs from the research context")
    measured = False
    if delay_observations is not None:
        from .providers import public_transport
        from .session_sampling import verified_delay_basis, utc_now, session_gate
        if transport is not public_transport or clock is not utc_now or len(symbols) != 1:
            raise DataContractError("Live delay requires the actual public transport/clock and one evidenced symbol")
        basis = verified_delay_basis(project_root, delay_observations["path"], delay_observations["sha256"],
            config_path=context["config_path"], symbol=symbols[0], now=clock())
        if (preflight["provider_delay_seconds"] != basis["empirical_delay_seconds"]
                or session_gate(active_profile, observed) != ("SESSION_OPEN", local_day)
                or not 0 <= (pd.Timestamp(clock()) - observed).total_seconds() <= 60):
            raise DataContractError("Live preflight delay/session/clock differs from measured evidence")
        measured = True
    poll_once = _vci_poll_once(
        preflight["observed_at_utc"], transport=transport, timeout_seconds=timeout_seconds,
        count_back=count_back, max_wait_seconds=limits["max_wait_seconds"], clock=clock)

    def checked_readiness(*args, **kwargs):
        _, latest_profile, latest_context = _trial_context(project_root, context["config_path"])
        if latest_context != context or latest_profile != profile:
            raise DataContractError("Config/calendar/profile changed during polling")
        frozen = verify_frozen_scoring_manifest(project_root, manifest_path)
        if frozen["manifest_sha256"] != expected_manifest.lower():
            raise DataContractError("Frozen manifest changed during polling")
        if measured and session_gate(profile, clock()) != ("SESSION_OPEN", local_day):
            raise DataContractError("Live poll crossed its market session boundary")
        if measured:
            latest_basis = verified_delay_basis(project_root, delay_observations["path"],
                delay_observations["sha256"], config_path=context["config_path"],
                symbol=symbols[0], now=clock())
            if latest_basis != basis:
                raise DataContractError("Measured delay evidence changed during polling")
        return readiness_fn(*args, **kwargs)

    if not callable(readiness_fn):
        raise DataContractError("readiness_fn must be callable")
    return run_bounded_poll_cycle(
        store_root, cycle_id, preflight["symbols"], poll_once=poll_once,
        profile=profile, observed_at_utc=preflight["observed_at_utc"],
        provider_delay_seconds=preflight["provider_delay_seconds"],
        stale_after_seconds=max(600, preflight["provider_delay_seconds"]),
        provider_delay_evidence=preflight["provider_delay_evidence"],
        project_root=project_root, manifest_path=manifest_path, delay_measured_live=measured,
        max_attempts=limits["max_attempts"], backoff_seconds=1,
        max_wait_seconds=limits["max_wait_seconds"], sleep=sleep,
        expected_manifest_sha256=expected_manifest,
        readiness_fn=checked_readiness, score_fn=score_fn)
