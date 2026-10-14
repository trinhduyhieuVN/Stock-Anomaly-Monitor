"""Bounded offline OHLC tapes; no scoring, training, provider or journal writes."""
from urllib.parse import parse_qs, urlsplit

from stock_monitor.io import DataContractError
from stock_monitor.research.history import historical_session
from stock_monitor.research.ai_demo import AI_MODELS, AI_SHA
from stock_monitor.upgrade.synthetic import bars_as_of
from stock_monitor.upgrade.contracts import utc_time
from stock_monitor.workbench.service import Workbench, clean
from stock_monitor.workbench.http import STATIC, parse_scope, make_server as inherited_server
from stock_monitor.workbench.waves import WavesBundle, SESSION as WAVES_SESSION, LOGOS
from stock_monitor.workbench.extra_prices import ExtraPrices, SYMBOLS as EXTRA_SYMBOLS


class SmoothWorkbench(Workbench):
    def __init__(self, root, journal_root=None):
        super().__init__(root, journal_root)
        self.watchlist, self.watchlist_error = None, None
        from stock_monitor.workbench.watchlist import WatchlistReplay, MODEL_PIN
        if (self.root / MODEL_PIN['path']).is_file():
            try:
                self.watchlist = WatchlistReplay(self.root)
            except (DataContractError, OSError, ValueError, KeyError) as exc:
                self.watchlist_error = str(exc)
        self.waves, self.waves_error = None, None
        self.extra_prices, self.extra_prices_error = None, None
        if (self.root / 'demo/watchlist_extension.local.json').is_file():
            try:
                self.extra_prices = ExtraPrices(self.root)
            except (DataContractError, OSError, ValueError, KeyError) as exc:
                self.extra_prices_error = str(exc)
        if (self.root / 'configs/demo_waves.json').is_file():
            try:
                self.waves = WavesBundle(self.root)
            except (DataContractError, OSError, ValueError, KeyError) as exc:
                self.waves_error = str(exc)

    def catalog(self):
        catalog = super().catalog()
        catalog['sources']['watchlist'] = self.watchlist.catalog() if self.watchlist else {
            'available': False, 'error': self.watchlist_error or 'Chưa có bộ mô hình 5 mã trên máy này',
            'symbols': [], 'sessions': [], 'models': {}}
        historical = catalog['sources']['historical']
        historical['extra_prices_error'] = self.extra_prices_error
        if self.extra_prices is not None and historical['available']:
            self.extra_prices.verify()
            original = list(historical['symbols']); historical['symbols'] = original + list(EXTRA_SYMBOLS)
            historical['price_only_symbols'] = list(EXTRA_SYMBOLS)
            historical['price_only_sessions'] = {s: [self.extra_prices.manifest['session']] if s in self.extra_prices.rows else [] for s in EXTRA_SYMBOLS}
            historical['peer_manifest_by_symbol'] = {s: self.bundles['historical']['manifest_sha256'] for s in original}
            historical['peer_manifest_by_symbol'].update({s: self.extra_prices.manifest_pin['sha256'] for s in EXTRA_SYMBOLS})
        synthetic = catalog['sources']['synthetic']
        synthetic['extra_session_error'] = self.waves_error
        if self.waves is not None and synthetic['available']:
            synthetic['sessions'] = [*synthetic['sessions'], WAVES_SESSION]
            synthetic['session_labels'] = {WAVES_SESSION: 'Dao động hai chiều · 17/02/2025'}
            synthetic['default_session'] = WAVES_SESSION
        return catalog

    def _scope(self, source, symbol, session, model):
        if source == 'synthetic' and session == WAVES_SESSION:
            if 'synthetic' not in self.bundles or self.waves is None:
                raise DataContractError('Waves session unavailable: ' + (self.waves_error or 'missing fixture'))
            return self.waves.scope(symbol, model)
        return super()._scope(source, symbol, session, model)

    def tape(self, source, symbol, session, model):
        with self.lock:
            if source == 'watchlist':
                return self._watchlist().tape(symbol, session, model)
            if source == 'historical' and symbol in EXTRA_SYMBOLS and self.extra_prices is not None:
                return clean(self.extra_prices.tape(symbol, session, model))
            bundle = self._scope(source, symbol, session, model)
            if source == 'historical':
                history = historical_session(bundle, symbol=symbol, session_id=session, model='lstm' if model in AI_MODELS.values() else model)
                bars = clean(history['bars'])
                frames = [{'step': i + 1, 'as_of': b['timestamp'], 'bars': bars[:i + 1]}
                          for i, b in enumerate(bars)]
                policy = history['data_policy']
                provenance = history['provenance']
                if model in AI_MODELS.values():
                    provenance = {**provenance, 'model_run': 'ai_outline:' + model, 'ai_manifest_sha256': AI_SHA}
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
                'demo_scenario': bundle['manifest'].get('scenario') if source == 'synthetic' else None,
                'model_scores_included': False})

    def _watchlist(self):
        if self.watchlist is None:
            raise DataContractError('Watchlist model unavailable: ' + (self.watchlist_error or 'missing local artifacts'))
        return self.watchlist

    def snapshot(self, source, symbol, session, model, replay=False, step=None):
        if source == 'watchlist':
            with self.lock:
                return self._watchlist().snapshot(symbol, session, model, replay, step)
        return super().snapshot(source, symbol, session, model, replay, step)

    def export(self, scope, kind):
        if scope.get('source') == 'watchlist':
            with self.lock:
                return self._watchlist().export(scope, kind)
        return super().export(scope, kind)


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
                if url.path in {'/monitor', '/monitor.js'} and not url.query:
                    name, mime = ('monitor.html', 'text/html; charset=utf-8') if url.path == '/monitor' else ('monitor.js', 'text/javascript; charset=utf-8')
                    return self.reply(200, (public / name).read_bytes(), mime)
                if url.path == '/ui-copy.js' and not url.query:
                    return self.reply(200, (public / 'ui-copy.js').read_bytes(), 'text/javascript; charset=utf-8')
                if url.path == '/api/monitor' and not url.query:
                    from stock_monitor.near_realtime import read_projection
                    return self.reply(200, read_projection(service.root))
                if url.path == '/api/monitor/gru6' and not url.query:
                    from stock_monitor.research.short_ui import read_candidate_projection
                    return self.reply(200, read_candidate_projection(service.root))
                if url.path in STATIC and not url.query:
                    name, mime = STATIC[url.path]
                    return self.reply(200, (public / name).read_bytes(), mime)
                if url.path in LOGOS and not url.query:
                    name, mime = LOGOS[url.path]
                    return self.reply(200, (public / 'logos' / name).read_bytes(), mime)
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
