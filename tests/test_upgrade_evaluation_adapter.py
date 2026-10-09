import pytest

from stock_monitor.io import DataContractError
from stock_monitor.upgrade.evaluation_adapter import variant_decision
from stock_monitor.upgrade.synthetic import content_hash


def fixture():
    cal = {"quantile": .98, "minimum_samples": 100}
    refs = {"calibration_sha256": content_hash(cal), "unguarded_maxima": {"FPT": {"3,6": list(range(100))}},
            "rolling_six": {"FPT": list(range(100))}}
    detail = {"event": {"event_id": "a", "symbol": "FPT", "kind": "point", "score": 1.,
               "event_start": "start", "event_end": "end"}, "point_rank": 1., "coverage_mask": "3,6",
        "candidates": [{"raw_score": 200}, {"length": 6, "raw_score": 200., "event_start": "segment_start", "admissible": False}]}
    return cal, refs, detail


def test_invariant_ablations_and_no_segment_point():
    cal, refs, detail = fixture()
    for variant in ["v2_base", "no_context", "no_post_event", "no_segment"]:
        assert variant_decision(detail, refs, cal, variant)["kind"] == "point"
    assert detail["event"]["event_id"] == "a"  # no mutable rewrite.


def test_no_guard_recalibrated_rank_and_simple_length_six():
    cal, refs, detail = fixture()
    for variant in ["no_single_point_guard", "simple_rolling"]:
        result = variant_decision(detail, refs, cal, variant)
        assert result["kind"] == "collective" and result["event_start"] == "segment_start"
    refs["rolling_six"]["FPT"] = []
    assert variant_decision(detail, refs, cal, "simple_rolling")["kind"] == "insufficient_data"
    refs["calibration_sha256"] = "bad"
    with pytest.raises(DataContractError):
        variant_decision(detail, refs, cal, "v2_base")


def test_endpoint_mask_does_not_mutate_readonly_pandas_array(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from stock_monitor.upgrade import evaluation_adapter as module
    stamp = pd.Timestamp("2025-02-03T03:00:00+00:00")
    frame = pd.DataFrame({"symbol": ["FPT"], "timestamp": [stamp], "session_id": ["2025-02-03"],
        "information_available_at": [stamp], "decision_as_of": [stamp]})
    monkeypatch.setattr(module, "causal_context", lambda *a: frame.copy())
    monkeypatch.setattr(module, "transform_features", lambda frame, scalers: frame)
    monkeypatch.setattr(module, "make_windows", lambda *a, **k: SimpleNamespace(
        values=np.zeros((1, 6, 4)), index=pd.DataFrame({"end_row": [0]})))
    values, index, rows = module.extract_frozen_windows(frame, {"dates": ["2025-02-03"]},
        {"scalers": {}}, endpoint=("FPT", stamp))
    assert values.shape == (1, 6, 4) and len(rows) == len(index) == 1
