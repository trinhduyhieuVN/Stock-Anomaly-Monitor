"""Bounded orchestration for incremental polling; network access is caller-owned."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Callable

import pandas as pd

from .incremental import (IncrementalStoreResult, load_incremental_current,
                          load_incremental_poll, persist_incremental_poll)
from .io import DataContractError
from .realtime import ClosedBarReadiness, closed_bar_readiness, verify_frozen_scoring_manifest


class PollRetryableError(RuntimeError):
    def __init__(self, kind: str, *, retry_after_seconds=None):
        super().__init__(kind)
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds


class PollNonRetryableError(RuntimeError):
    def __init__(self, state: str, kind: str):
        super().__init__(kind)
        self.state = state
        self.kind = kind


@dataclass(frozen=True)
class PollPayload:
    raw_body: bytes
    retrieved_at_utc: object


@dataclass(frozen=True)
class PollCycleResult:
    state: str
    operator_message: str
    attempts: tuple[dict, ...]
    waited_seconds: float
    readiness: ClosedBarReadiness | None
    score_result: object | None
    store_results: tuple[IncrementalStoreResult, ...]
    live_evidence: bool


def _cycle_result(state, message, attempts, waited, stores, *, readiness=None,
                  score_result=None, live_evidence=False):
    return PollCycleResult(state, message, tuple(attempts), waited, readiness,
                           score_result, tuple(stores), live_evidence)


def _validate_bounds(cycle_id, symbols, max_attempts, backoff_seconds, max_wait_seconds):
    if not isinstance(cycle_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}", cycle_id):
        raise DataContractError("cycle_id must be a safe nonempty identifier")
    if (not isinstance(symbols, (list, tuple)) or not symbols or len(symbols) != len(set(symbols))
            or any(not isinstance(symbol, str) or not symbol.isalnum() or symbol != symbol.upper()
                   for symbol in symbols)):
        raise DataContractError("symbols must be unique uppercase identifiers")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 10:
        raise DataContractError("max_attempts must be an integer from 1 to 10")
    for name, value, upper in (("backoff_seconds", backoff_seconds, 60),
                               ("max_wait_seconds", max_wait_seconds, 300)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= upper:
            raise DataContractError(f"{name} must be explicitly bounded from 0 to {upper}")


def run_bounded_poll_cycle(store_root, cycle_id: str, symbols, *, poll_once: Callable,
                           profile: dict, observed_at_utc, provider_delay_seconds,
                           stale_after_seconds, provider_delay_evidence, project_root,
                           manifest_path="demo/manifest.local.json", delay_measured_live=False,
                           expected_manifest_sha256=None,
                           max_attempts=3, backoff_seconds=1, max_wait_seconds=10,
                           sleep: Callable[[float], None] = time.sleep,
                           readiness_fn: Callable = closed_bar_readiness,
                           score_fn: Callable | None = None) -> PollCycleResult:
    """Run one finite poll cycle without owning or silently substituting a provider.

    ``poll_once(symbol, attempt)`` must return :class:`PollPayload`. Stable poll
    IDs reuse verified saved bytes and receipt times before calling the transport.
    Use a new cycle ID to collect a new observation. Failures are exposed as operator
    states; raw exception text is intentionally not copied into status output.
    """
    _validate_bounds(cycle_id, symbols, max_attempts, backoff_seconds, max_wait_seconds)
    if not callable(poll_once) or not callable(sleep) or not callable(readiness_fn):
        raise DataContractError("poll_once, sleep and readiness_fn must be callable")
    if score_fn is not None and not callable(score_fn):
        raise DataContractError("score_fn must be callable when provided")
    try:
        frozen = verify_frozen_scoring_manifest(project_root, manifest_path)
        if expected_manifest_sha256 is not None:
            if (not isinstance(expected_manifest_sha256, str)
                    or frozen["manifest_sha256"] != expected_manifest_sha256.lower()):
                raise DataContractError("Frozen manifest differs from the pinned preflight")
    except DataContractError:
        return _cycle_result("BLOCKED_ARTIFACT", "Frozen scoring artifacts failed preflight.",
                             [], 0.0, [], live_evidence=False)

    attempts, stores, waited = [], [], 0.0
    received = []
    reused = False
    for symbol in symbols:
        poll_id = f"{cycle_id}-{symbol}"
        for attempt in range(1, max_attempts + 1):
            try:
                # Recheck on every retry: a failed pointer swap may already have
                # persisted an immutable receipt during this very cycle.
                saved = load_incremental_poll(store_root, poll_id, symbol)
                if saved is not None:
                    raw_body, retrieved_at_utc = saved
                    stored = persist_incremental_poll(store_root, poll_id, raw_body, symbol, retrieved_at_utc)
                    stores.append(stored)
                    received.append(retrieved_at_utc)
                    attempts.append({"symbol": symbol, "attempt": 0, "outcome": "reused",
                                     "already_applied": bool(stored.report["already_applied"])})
                    reused = True
                    break
                payload = poll_once(symbol, attempt)
                if not isinstance(payload, PollPayload):
                    raise DataContractError("poll_once must return PollPayload")
                stored = persist_incremental_poll(
                    store_root, poll_id, payload.raw_body, symbol,
                    payload.retrieved_at_utc)
            except PollNonRetryableError as exc:
                attempts.append({"symbol": symbol, "attempt": attempt, "outcome": "non_retryable",
                                 "error_type": exc.kind})
                return _cycle_result(exc.state, "Provider response requires operator action.",
                                     attempts, waited, stores, live_evidence=False)
            except PollRetryableError as exc:
                attempts.append({"symbol": symbol, "attempt": attempt, "outcome": "retryable",
                                 "error_type": exc.kind})
                if attempt == max_attempts:
                    return _cycle_result("RETRY_EXHAUSTED", "Maximum poll attempts reached.",
                                         attempts, waited, stores, live_evidence=False)
                delay = float(backoff_seconds) * (2 ** (attempt - 1))
                if exc.retry_after_seconds is not None:
                    delay = max(delay, float(exc.retry_after_seconds))
                if waited + delay > float(max_wait_seconds):
                    return _cycle_result("RETRY_WAIT_BUDGET_EXHAUSTED",
                                         "Retry wait budget would be exceeded.", attempts, waited,
                                         stores, live_evidence=False)
                sleep(delay)
                waited += delay
                continue
            except DataContractError:
                return _cycle_result("BLOCKED_DATA_CONTRACT", "Poll payload or store contract failed.",
                                     attempts, waited, stores, live_evidence=False)
            except Exception as exc:  # Provider/transport implementation remains caller-owned.
                attempts.append({"symbol": symbol, "attempt": attempt, "outcome": "transport_error",
                                 "error_type": type(exc).__name__})
                if attempt == max_attempts:
                    return _cycle_result("RETRY_EXHAUSTED", "Maximum poll attempts reached.",
                                         attempts, waited, stores, live_evidence=False)
                delay = float(backoff_seconds) * (2 ** (attempt - 1))
                if waited + delay > float(max_wait_seconds):
                    return _cycle_result("RETRY_WAIT_BUDGET_EXHAUSTED",
                                         "Retry wait budget would be exceeded.", attempts, waited,
                                         stores, live_evidence=False)
                sleep(delay)
                waited += delay
                continue
            attempts.append({"symbol": symbol, "attempt": attempt, "outcome": "stored",
                             "already_applied": bool(stored.report["already_applied"])})
            stores.append(stored)
            received.append(pd.Timestamp(payload.retrieved_at_utc))
            break

    minutes, _ = load_incremental_current(store_root)
    try:
        readiness = readiness_fn(
            minutes, profile, symbols=list(symbols), observed_at_utc=observed_at_utc,
            provider_delay_seconds=provider_delay_seconds,
            stale_after_seconds=stale_after_seconds,
            provider_delay_evidence=provider_delay_evidence, project_root=project_root,
            manifest_path=manifest_path, delay_measured_live=delay_measured_live)
    except DataContractError:
        return _cycle_result("BLOCKED_READINESS_CONTRACT", "Readiness checks failed closed.",
                             attempts, waited, stores, live_evidence=False)
    if readiness.status.get("gap_count", 0):
        return _cycle_result("BLOCKED_GAPS", "Closed-bar gaps require operator review before scoring.",
                             attempts, waited, stores, readiness=readiness, live_evidence=False)
    if readiness.status.get("scoring_allowed") is not True:
        state = str(readiness.status.get("state", "BLOCKED_READINESS"))
        return _cycle_result(state, "Readiness has not approved scoring.", attempts, waited,
                             stores, readiness=readiness, live_evidence=False)
    if score_fn is None:
        return _cycle_result("READY_FOR_SCORE", "Closed bars are ready; no scorer was supplied.",
                             attempts, waited, stores, readiness=readiness, live_evidence=False)
    try:
        score_result = score_fn(readiness, max(received))
    except DataContractError:
        return _cycle_result("BLOCKED_SCORING_CONTRACT", "Frozen scoring failed closed.",
                             attempts, waited, stores, readiness=readiness, live_evidence=False)
    live_candidate = bool(delay_measured_live and not reused)
    state = "SCORED_LIVE_CANDIDATE" if live_candidate else "SCORED_OFFLINE"
    return _cycle_result(state, "Scoring completed with frozen resources.", attempts, waited,
                         stores, readiness=readiness, score_result=score_result,
                         live_evidence=live_candidate)
