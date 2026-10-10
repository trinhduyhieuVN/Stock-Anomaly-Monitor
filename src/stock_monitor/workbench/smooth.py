"""Bounded offline OHLC tapes; no scoring, training, provider or journal writes."""
from urllib.parse import parse_qs, urlsplit

from stock_monitor.io import DataContractError
from stock_monitor.research.history import historical_session
from stock_monitor.upgrade.synthetic import bars_as_of
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.workbench.service import Workbench, clean
from stock_monitor.workbench.http import STATIC, parse_scope, make_server as inherited_server


class SmoothWorkbench(Workbench):
    def tape(self, source, symbol, session, model):
        with self.lock:
            bundle = self._scope(source, symbol, session, model)
            if source == 'historical':
                history = historical_session(bundle, symbol=symbol, session_id=session, model=model)
                bars = clean(history['bars'])
                frames = [{'step': i + 1, 'as_of': b['timestamp'], 'bars': bars[:i + 1]}
                          for i, b in enumerate(bars)]
                policy = history['data_policy']
                provenance = history['provenance']
                basis = history['availability_basis']
            else:
                rows = [r for r in bundle['rows'] if r['symbol'] == symbol and r['session_id'] == session]
                deliveries = sorted({r['available_at'] for r in rows}, key=utc_time)
                if len(deliveries) > 1000:
                    raise DataContractError('Offline tape exceeds bounded session size')
                frames = []
                for step, cutoff in enumerate(deliveries, 1):
                    bars = clean(bars_as_of(rows, cutoff))
                    frame = {'step': step, 'as_of': cutoff, 'bars': bars}
                    # One slider frame per new bar count; same-count corrections are
                    # folded into the final receipt before the next bar arrives.
                    if frames and len(frames[-1]['bars']) == len(bars):
                        frames[-1] = frame
                    else:
                        frames.append(frame)
                policy = {'classification': 'synthetic', 'source_rights_status': 'synthetic', 'distributable': True}
                provenance = {'manifest_sha256': bundle['key'], 'model_kind': bundle['manifest']['model_kind']}
                basis = 'simulated_logical_deliveries_not_live'
            if not frames or len(frames) > 1000 or len(frames[-1]['bars']) > 1000:
                raise DataContractError('Empty or oversized offline tape')
            return clean({'source': source, 'symbol': symbol, 'session': session, 'model': model,
                'source_mode': 'SYNTHETIC' if source == 'synthetic' else 'HISTORICAL_REPLAY',
                'frames': frames, 'policy': policy, 'provenance': provenance,
                'availability_basis': basis, 'received_at_utc': None, 'live': False,
                'mode': 'offline_full_session_tape_not_blind_review',
                'revision_policy': 'latest_known_receipt_before_next_new_bar',
                'model_scores_included': False})


def make_server(service, port=8514):
    server = inherited_server(service, port)
    inherited = server.RequestHandlerClass
    public = service.root / 'web/research-workbench-replay/public'

    class SmoothHandler(inherited):
        def reply(self, code, data, mime='application/json; charset=utf-8', filename=None):
            # The inherited guard rejects unauthorized POST before reading its
            # body. Closing a Windows socket with pending bytes can reset the
            # connection before the client receives the 403. Drain only a small,
            # declared body, with a tighter timeout; never parse or persist it.
            if code == 403 and self.command == 'POST' and not self.headers.get('Transfer-Encoding'):
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if 0 < size <= 16384:
                        self.connection.settimeout(.25)
                        self.rfile.read(size)
                except (ValueError, OSError):
                    pass
                finally:
                    self.connection.settimeout(10)
            return super().reply(code, data, mime, filename)

        def do_GET(self):
            url = urlsplit(self.path)
            if not self._host():
                return self.reply(403, {'error': 'Invalid local Host'})
            try:
                if url.path in STATIC and not url.query:
                    name, mime = STATIC[url.path]
                    return self.reply(200, (public / name).read_bytes(), mime)
                if url.path == '/api/replay-tape':
                    query = parse_qs(url.query, keep_blank_values=True)
                    if set(query) - {'source', 'symbol', 'session', 'model'}:
                        raise DataContractError('Tape accepts only source/symbol/session/model')
                    scope = parse_scope(query)
                    del scope['replay'], scope['step']
                    return self.reply(200, service.tape(**scope))
            except (DataContractError, ValueError, KeyError, OSError) as exc:
                return self.reply(400, {'error': str(exc)})
            return super().do_GET()

    server.RequestHandlerClass = SmoothHandler
    return server
