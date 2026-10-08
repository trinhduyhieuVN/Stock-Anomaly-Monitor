"""Bounded application file retries must preserve old data on permanent failure."""
import json

import pytest

import stock_monitor.incremental as incremental


def windows_error(code):
    error = PermissionError("simulated Windows file access failure")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_conflict_publishes_only_complete_new_document(tmp_path, monkeypatch, caplog, code):
    path = tmp_path / "current.json"
    incremental._atomic_json(path, {"revision": 1})
    original = incremental.os.replace
    calls, waits = [], []
    def replace(source, destination):
        assert json.loads(path.read_bytes()) == {"revision": 1}
        assert json.loads(source.read_bytes()) == {"revision": 2}
        calls.append((source, destination))
        if len(calls) < 3:
            raise windows_error(code)
        return original(source, destination)
    monkeypatch.setattr(incremental.os, "replace", replace)
    monkeypatch.setattr(incremental.time, "sleep", waits.append)
    incremental._atomic_json(path, {"revision": 2})
    assert len(calls) == 3 and waits == [0.05, 0.1]
    assert json.loads(path.read_bytes()) == {"revision": 2}
    assert list(tmp_path.iterdir()) == [path]
    assert len(caplog.records) == 2


def test_permanent_access_failure_stops_and_preserves_old_document(tmp_path, monkeypatch):
    path = tmp_path / "current.json"
    incremental._atomic_json(path, {"revision": 1})
    original_bytes = path.read_bytes()
    calls, waits = [], []
    def denied(*args):
        calls.append(args)
        raise windows_error(5)
    monkeypatch.setattr(incremental.os, "replace", denied)
    monkeypatch.setattr(incremental.time, "sleep", waits.append)
    with pytest.raises(PermissionError):
        incremental._atomic_json(path, {"revision": 2})
    assert len(calls) == 3 and sum(waits) == pytest.approx(0.15)
    assert path.read_bytes() == original_bytes and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("error", [PermissionError("no Windows error code"),
                                  OSError("disk failure"), windows_error(112)])
def test_other_errors_never_retry(tmp_path, monkeypatch, error):
    path = tmp_path / "current.json"
    calls = []
    def failed(*args):
        calls.append(args)
        raise error
    monkeypatch.setattr(incremental.os, "replace", failed)
    monkeypatch.setattr(incremental.time, "sleep", lambda _: pytest.fail("unexpected wait"))
    with pytest.raises(OSError):
        incremental._atomic_json(path, {"revision": 1})
    assert len(calls) == 1 and list(tmp_path.iterdir()) == []


def test_serialization_failure_preserves_document_without_replace(tmp_path, monkeypatch):
    path = tmp_path / "current.json"
    incremental._atomic_json(path, {"revision": 1})
    monkeypatch.setattr(incremental.os, "replace", lambda *args: pytest.fail("unexpected replace"))
    with pytest.raises(TypeError):
        incremental._atomic_json(path, {"value": object()})
    assert json.loads(path.read_bytes()) == {"revision": 1}
    assert list(tmp_path.iterdir()) == [path]
