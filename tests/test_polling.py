import hashlib
import json

import pandas as pd
import pytest

from stock_monitor.incremental import load_incremental_current
from stock_monitor.polling import PollPayload, run_bounded_poll_cycle
from stock_monitor.realtime import ClosedBarReadiness
import stock_monitor.incremental as incremental


ROLES = ["feature_scalers", "baseline_models", "baseline_thresholds", "lstm_checkpoint", "scoring_thresholds"]


def manifest(root):
    files = []
    for role in ROLES:
        path = root / "frozen" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        files.append({"role": role, "path": path.relative_to(root).as_posix(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    value = {"schema_version": 1, "data_policy": {"classification": "local_only_real_data",
             "source_rights_status": "pending", "distributable": False}, "files": files}
    path = root / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def profile():
    return {"calendar": {"status": "observed_only", "complete": False,
            "timezone": "Asia/Ho_Chi_Minh", "dates": ["2026-10-01"],
            "sessions": [{"id": "morning", "start": "09:15", "end": "11:30"}],
            "overrides": {}}, "bar_label": "bar_start", "price": {"unit": "VND", "multiplier": 1},
            "volume": {"unit": "shares", "multiplier": 1}, "research": True, "diagnostic": False,
            "confirmations_pending": ["data_usage"]}


def raw_poll():
    stamp = int(pd.Timestamp("2026-10-01T02:15:00Z").timestamp())
    return json.dumps([{"symbol": "FPT", "t": [stamp], "o": [100], "h": [101], "l": [99],
                        "c": [100], "v": [10], "accumulatedVolume": [10]}]).encode()


def kwargs(tmp_path, poll_once):
    return dict(store_root=tmp_path / "store", cycle_id="cycle-001", symbols=["FPT"],
                poll_once=poll_once, profile=profile(), observed_at_utc="2026-10-01T02:21:00Z",
                provider_delay_seconds=60, stale_after_seconds=600,
                provider_delay_evidence="offline orchestration test; not live", project_root=tmp_path,
                manifest_path=manifest(tmp_path), delay_measured_live=False)


def test_retry_attempts_and_wait_are_bounded(tmp_path):
    calls, waits = [], []
    def fail(symbol, attempt):
        calls.append((symbol, attempt))
        raise TimeoutError("synthetic")
    result = run_bounded_poll_cycle(**kwargs(tmp_path, fail), max_attempts=3,
                                    backoff_seconds=1, max_wait_seconds=10, sleep=waits.append)
    assert result.state == "RETRY_EXHAUSTED"
    assert calls == [("FPT", 1), ("FPT", 2), ("FPT", 3)]
    assert waits == [1.0, 2.0]
    assert result.waited_seconds == 3


def test_retry_stops_before_exceeding_wait_budget(tmp_path):
    calls, waits = [], []
    def fail(symbol, attempt):
        calls.append(attempt)
        raise ConnectionError("synthetic")
    result = run_bounded_poll_cycle(**kwargs(tmp_path, fail), max_attempts=5,
                                    backoff_seconds=2, max_wait_seconds=2, sleep=waits.append)
    assert result.state == "RETRY_WAIT_BUDGET_EXHAUSTED"
    assert calls == [1, 2]
    assert waits == [2.0]
    assert result.waited_seconds == 2


def test_same_cycle_retry_is_store_idempotent(tmp_path):
    count = 0
    def flaky(symbol, attempt):
        nonlocal count
        count += 1
        if count == 1:
            raise TimeoutError("synthetic")
        return PollPayload(raw_poll(), "2026-10-01T02:16:10Z")
    values = kwargs(tmp_path, flaky)
    first = run_bounded_poll_cycle(**values, sleep=lambda _: None)
    second = run_bounded_poll_cycle(**values, sleep=lambda _: None)
    _, pointer = load_incremental_current(tmp_path / "store")
    assert first.store_results[0].report["already_applied"] is False
    assert second.store_results[0].report["already_applied"] is True
    assert pointer["applied_poll_ids"] == ["cycle-001-FPT"]


@pytest.mark.parametrize("status, expected", [
    ({"state": "STALE", "scoring_allowed": False, "gap_count": 0}, "STALE"),
    ({"state": "BLOCKED_UNKNOWN_CALENDAR", "scoring_allowed": False, "gap_count": 0},
     "BLOCKED_UNKNOWN_CALENDAR"),
    ({"state": "READY_WITH_GAPS", "scoring_allowed": True, "gap_count": 1}, "BLOCKED_GAPS"),
])
def test_readiness_outcomes_never_call_scorer(tmp_path, status, expected):
    called = []
    values = kwargs(tmp_path, lambda *_: PollPayload(raw_poll(), "2026-10-01T02:16:10Z"))
    frozen = {"placeholder": True}
    def readiness(*args, **kwargs):
        return ClosedBarReadiness(pd.DataFrame(), status, frozen)
    result = run_bounded_poll_cycle(**values, readiness_fn=readiness,
                                    score_fn=lambda *_: called.append(True), sleep=lambda _: None)
    assert result.state == expected
    assert called == []


def test_artifact_preflight_blocks_before_poll(tmp_path):
    called = []
    values = kwargs(tmp_path, lambda *_: called.append(True))
    artifact = tmp_path / "frozen" / "lstm_checkpoint.bin"
    artifact.write_bytes(b"tampered")
    result = run_bounded_poll_cycle(**values, sleep=lambda _: None)
    assert result.state == "BLOCKED_ARTIFACT"
    assert called == []


def test_invalid_wire_payload_records_the_request_but_never_scores(tmp_path):
    values=kwargs(tmp_path,lambda *_: PollPayload(b'{}','2026-10-01T02:16:10Z'))
    result=run_bounded_poll_cycle(**values)
    assert result.state=='BLOCKED_DATA_CONTRACT'
    assert len(result.attempts)==1 and result.attempts[0]['request_started']
    assert result.attempts[0]['attempt']==1 and result.score_result is None


def test_ready_cycle_calls_supplied_scorer_and_labels_offline(tmp_path):
    values = kwargs(tmp_path, lambda *_: PollPayload(raw_poll(), "2026-10-01T02:16:10Z"))
    status = {"state": "READY", "scoring_allowed": True, "gap_count": 0}
    def readiness(*args, **kwargs):
        return ClosedBarReadiness(pd.DataFrame(), status, {"placeholder": True})
    received = []
    result = run_bounded_poll_cycle(**values, readiness_fn=readiness,
                                    score_fn=lambda ready, stamp: received.append(stamp) or "scores",
                                    sleep=lambda _: None)
    assert result.state == "SCORED_OFFLINE"
    assert result.live_evidence is False
    assert result.score_result == "scores"
    assert received == [pd.Timestamp("2026-10-01T02:16:10Z")]


def ready_for_restart(*args, **kwargs):
    return ClosedBarReadiness(pd.DataFrame(),
                             {"state": "READY", "scoring_allowed": True, "gap_count": 0}, {})


def test_restart_never_repolls_or_refreshes_saved_receipt(tmp_path):
    calls, received = [], []
    def changing_poll(symbol, attempt):
        calls.append(symbol)
        body = raw_poll() if len(calls) == 1 else raw_poll().replace(b'[100]', b'[101]')
        stamp = "2026-10-01T02:16:10Z" if len(calls) == 1 else "2026-10-01T02:25:00Z"
        return PollPayload(body, stamp)
    values = kwargs(tmp_path, changing_poll)
    run_bounded_poll_cycle(**values, readiness_fn=ready_for_restart)
    pointer_before = (tmp_path / "store/current.json").read_bytes()
    values["observed_at_utc"] = "2026-10-01T02:26:00Z"
    values["delay_measured_live"] = True
    restarted = run_bounded_poll_cycle(**values, readiness_fn=ready_for_restart,
                                      score_fn=lambda ready, stamp: received.append(stamp))
    assert calls == ["FPT"]
    assert (tmp_path / "store/current.json").read_bytes() == pointer_before
    assert received == [pd.Timestamp("2026-10-01T02:16:10Z")]
    assert restarted.attempts[0]["outcome"] == "reused"
    assert restarted.attempts[0]["attempt"] == 0
    assert restarted.state == "SCORED_OFFLINE"
    assert restarted.live_evidence is False


def test_restart_checks_staleness_at_new_observation_without_network(tmp_path):
    values = kwargs(tmp_path, lambda *_: PollPayload(raw_poll(), "2026-10-01T02:16:10Z"))
    run_bounded_poll_cycle(**values)
    values["observed_at_utc"] = "2026-10-01T02:40:00Z"
    values["poll_once"] = lambda *_: pytest.fail("Saved cycle must not call provider")
    result = run_bounded_poll_cycle(**values)
    assert result.state in {"STALE", "BLOCKED_GAPS"}
    assert result.score_result is None
    assert result.store_results[0].report["already_applied"] is True


def test_partial_cycle_resumes_only_missing_symbol(tmp_path):
    calls = []
    def first_poll(symbol, attempt):
        calls.append(symbol)
        if symbol == "HPG":
            raise TimeoutError("interrupted after FPT")
        return PollPayload(raw_poll(), "2026-10-01T02:16:10Z")
    values = kwargs(tmp_path, first_poll)
    values["symbols"] = ["FPT", "HPG"]
    first = run_bounded_poll_cycle(**values, max_attempts=1)
    assert first.state == "RETRY_EXHAUSTED"
    calls.clear()
    def resumed_poll(symbol, attempt):
        calls.append(symbol)
        return PollPayload(raw_poll().replace(b'"FPT"', b'"HPG"'), "2026-10-01T02:17:10Z")
    values["poll_once"] = resumed_poll
    second = run_bounded_poll_cycle(**values, readiness_fn=ready_for_restart)
    assert calls == ["HPG"]
    assert [row["outcome"] for row in second.attempts] == ["reused", "stored"]
    assert second.store_results[-1].pointer["applied_poll_ids"] == ["cycle-001-FPT", "cycle-001-HPG"]


def test_pointer_crash_recovers_saved_poll_without_new_receipt(tmp_path, monkeypatch):
    original = incremental._atomic_json
    def crash(*_):
        raise OSError("simulated pointer swap crash")
    values = kwargs(tmp_path, lambda *_: PollPayload(raw_poll(), "2026-10-01T02:16:10Z"))
    monkeypatch.setattr(incremental, "_atomic_json", crash)
    first = run_bounded_poll_cycle(**values, max_attempts=1)
    assert first.state == "RETRY_EXHAUSTED"
    assert not (tmp_path / "store/current.json").exists()
    monkeypatch.setattr(incremental, "_atomic_json", original)
    values["poll_once"] = lambda *_: pytest.fail("Recovery must reuse the durable raw receipt")
    recovered = run_bounded_poll_cycle(**values, readiness_fn=ready_for_restart)
    assert recovered.state == "READY_FOR_SCORE"
    assert recovered.attempts[0]["outcome"] == "reused"
    assert recovered.store_results[0].report["already_applied"] is False
    assert recovered.store_results[0].minutes.retrieved_at_utc.item() == pd.Timestamp("2026-10-01T02:16:10Z")


def test_in_cycle_retry_after_pointer_failure_does_not_repoll(tmp_path, monkeypatch):
    original = incremental._atomic_json
    writes, calls = [], []
    def fail_once(path, value):
        writes.append(path)
        if len(writes) == 1:
            raise OSError("simulated one-time pointer failure")
        return original(path, value)
    def poll(symbol, attempt):
        calls.append(attempt)
        return PollPayload(raw_poll(), "2026-10-01T02:16:10Z")
    monkeypatch.setattr(incremental, "_atomic_json", fail_once)
    result = run_bounded_poll_cycle(**kwargs(tmp_path, poll), sleep=lambda _: None,
                                    readiness_fn=ready_for_restart)
    assert calls == [1]
    assert result.state == "READY_FOR_SCORE"
    assert [row["outcome"] for row in result.attempts] == ["transport_error", "reused"]
    assert result.waited_seconds == 1
    assert result.store_results[0].minutes.retrieved_at_utc.item() == pd.Timestamp("2026-10-01T02:16:10Z")


@pytest.mark.parametrize("damage", ["raw", "missing_manifest", "symbol", "raw_file", "raw_sha256",
                                    "rows", "retrieved_at_utc", "source_rights_status", "live_evidence"])
def test_corrupt_saved_poll_blocks_before_transport(tmp_path, damage):
    values = kwargs(tmp_path, lambda *_: PollPayload(raw_poll(), "2026-10-01T02:16:10Z"))
    run_bounded_poll_cycle(**values)
    directory = tmp_path / "store/polls/cycle-001-FPT"
    manifest_path = directory / "manifest.json"
    if damage == "raw":
        (directory / "response.json").write_bytes(b"tampered")
    elif damage == "missing_manifest":
        manifest_path.unlink()
    else:
        saved = json.loads(manifest_path.read_text())
        saved[damage] = {"symbol": "HPG", "raw_file": "../outside.json", "raw_sha256": "0" * 64,
                         "rows": 200, "retrieved_at_utc": "2026-10-01T02:25:00Z",
                         "source_rights_status": "confirmed", "live_evidence": True}[damage]
        manifest_path.write_text(json.dumps(saved))
    values["poll_once"] = lambda *_: pytest.fail("Corrupt store must not trigger provider repoll")
    result = run_bounded_poll_cycle(**values)
    assert result.state == "BLOCKED_DATA_CONTRACT"
    assert result.attempts == ()
    assert result.live_evidence is False


def test_new_cycle_preserves_revision_and_restarting_old_cycle_cannot_roll_back(tmp_path):
    values = kwargs(tmp_path, lambda *_: PollPayload(raw_poll(), "2026-10-01T02:16:10Z"))
    run_bounded_poll_cycle(**values)
    values["cycle_id"] = "cycle-002"
    revised_body = raw_poll().replace(b'"c": [100]', b'"c": [101]')
    values["poll_once"] = lambda *_: PollPayload(revised_body, "2026-10-01T02:17:10Z")
    revised = run_bounded_poll_cycle(**values, readiness_fn=ready_for_restart)
    assert revised.store_results[0].report["revisions_applied"] == 1
    pointer_before = (tmp_path / "store/current.json").read_bytes()
    values["cycle_id"] = "cycle-001"
    values["poll_once"] = lambda *_: pytest.fail("Old applied cycle must not repoll")
    restarted = run_bounded_poll_cycle(**values, readiness_fn=ready_for_restart)
    assert restarted.store_results[0].minutes.close.item() == 101
    assert (tmp_path / "store/current.json").read_bytes() == pointer_before
    version = json.loads((tmp_path / "store" / revised.store_results[0].pointer["version_manifest"]).read_text())
    assert version["revisions"][0]["previous_close"] == 100
    assert version["revisions"][0]["incoming_close"] == 101
