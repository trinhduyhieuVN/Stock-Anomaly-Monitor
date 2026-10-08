"""Atomic siblings must work without changing Windows long-path policy."""
import json

import pytest

from stock_monitor.incremental import _atomic_json, _write_once
from stock_monitor.io import DataContractError


@pytest.mark.parametrize("writer", ["immutable", "pointer"])
def test_deep_store_uses_short_exclusive_temporary_sibling(tmp_path, writer):
    parent = tmp_path / ("d" * max(1, 230 - len(str(tmp_path.resolve())) - 1))
    parent.mkdir()
    path = parent / "manifest.json"
    # The old destination-name + UUID suffix would exceed Windows MAX_PATH,
    # even though the destination and parent themselves are valid paths.
    assert len(str(path)) < 260 < len(str(path)) + 37
    if writer == "immutable":
        _write_once(path, b"original")
        _write_once(path, b"original")
        with pytest.raises(DataContractError, match="Immutable"):
            _write_once(path, b"changed")
        assert path.read_bytes() == b"original"
    else:
        _atomic_json(path, {"revision": 1})
        _atomic_json(path, {"revision": 2})
        assert json.loads(path.read_bytes()) == {"revision": 2}
    assert list(parent.iterdir()) == [path]
