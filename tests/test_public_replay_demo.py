"""The public demo works without private data, trained weights or coordinator."""
import json
from pathlib import Path
import shutil
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts import run_demo
from stock_monitor.workbench.smooth import SmoothWorkbench, make_server

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def public_root(tmp_path):
    for name in ('configs/upgrade_ui.json',):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name in ('demo/upgrade_ui', 'web/research-workbench-replay/public'):
        shutil.copytree(ROOT / name, tmp_path / name)
    return tmp_path


def test_preflight_without_private_artifacts(public_root, monkeypatch, capsys):
    monkeypatch.setattr(run_demo, 'ROOT', public_root)
    assert run_demo.main(['--check-only']) == 0
    assert 'no server or provider started' in capsys.readouterr().out
    assert not (public_root / 'data').exists()
    assert not (public_root / '.autopilot').exists()


@pytest.mark.parametrize('argv', [['--port', '80'], ['--minutes', '0'], ['--minutes', '91'], ['--minutes', 'nan']])
def test_argument_bounds(argv):
    with pytest.raises(SystemExit) as exc:
        run_demo.main(argv)
    assert exc.value.code == 2


def test_missing_static_fails_before_binding(public_root, monkeypatch):
    monkeypatch.setattr(run_demo, 'ROOT', public_root)
    (public_root / 'web/research-workbench-replay/public/app.js').unlink()
    with pytest.raises(RuntimeError, match='Missing demo interface'):
        run_demo.main(['--check-only'])


def test_real_public_http_replay(public_root):
    service = SmoothWorkbench(public_root)
    assert set(service.bundles) == {'synthetic'}
    server = make_server(service, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        for route in ('/', '/app.js', '/style.css', '/api/health'):
            with urlopen(base + route, timeout=10) as response:
                assert response.status == 200
        with urlopen(base + '/api/catalog', timeout=10) as response:
            catalog = json.load(response)
        assert catalog['sources']['synthetic']['available']
        assert not catalog['sources']['historical']['available']
        assert not catalog['live']
        for symbol in ('FPT', 'HPG', 'VCB'):
            scope = f'source=synthetic&symbol={symbol}&session=2025-02-03&model=v2'
            with urlopen(base + '/api/replay-tape?' + scope, timeout=30) as response:
                tape = json.load(response)
            assert len(tape['frames']) == 48
            assert len(tape['frames'][0]['bars']) == 1
            assert tape['live'] is False and tape['received_at_utc'] is None
            with urlopen(base + '/api/snapshot?' + scope + '&replay=true&step=1', timeout=30) as response:
                snapshot = json.load(response)
            assert len(snapshot['bars']) == 1
            assert snapshot['policy']['distributable'] is True
            assert all(bar['timestamp'] <= snapshot['as_of'] for bar in snapshot['bars'])
            with urlopen(base + '/api/export?' + scope + '&replay=true&step=1&kind=bars', timeout=30) as response:
                assert len(response.read().decode('utf-8-sig').splitlines()) == 2
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + '/', headers={'Host': 'evil.example'}), timeout=10)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_bounded_launcher_stops_owned_server(public_root, monkeypatch):
    monkeypatch.setattr(run_demo, 'ROOT', public_root)
    assert run_demo.main(['--port', '19504', '--minutes', '0.001']) == 0
