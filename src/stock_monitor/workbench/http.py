"""Small loopback-only HTTP boundary; no repository-wide file serving."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import sqlite3
from urllib.parse import parse_qs, urlsplit

from stock_monitor.io import DataContractError
from stock_monitor.upgrade.journal import JournalConflict

STATIC = {'/': ('index.html', 'text/html; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8')}


def parse_scope(query):
    if set(query) - {'source', 'symbol', 'session', 'model', 'replay', 'step', 'kind'} or any(len(v) != 1 for v in query.values()):
        raise DataContractError('Unexpected query parameters')
    replay = query.get('replay', ['false'])[0]
    if replay not in {'true', 'false'}:
        raise DataContractError('Invalid Replay mode')
    return {'source': query.get('source', ['historical'])[0], 'symbol': query.get('symbol', ['FPT'])[0],
            'session': query.get('session', [''])[0], 'model': query.get('model', ['robust_zscore'])[0],
            'replay': replay == 'true', 'step': int(query['step'][0]) if 'step' in query else None}


def make_server(service, port=8511):
    token = secrets.token_urlsafe(32)
    public = service.root / 'web/research-workbench/public'

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def _host(self):
            expected = f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host') == expected

        def reply(self, code, data, mime='application/json; charset=utf-8', filename=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            if filename:
                self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._host():
                return self.reply(403, {'error': 'Invalid local Host'})
            url = urlsplit(self.path)
            try:
                if url.path in STATIC and not url.query:
                    name, mime = STATIC[url.path]
                    return self.reply(200, (public / name).read_bytes(), mime)
                if url.path == '/api/catalog' and not url.query:
                    return self.reply(200, {**service.catalog(), 'csrf_token': token})
                if url.path == '/api/health' and not url.query:
                    return self.reply(200, {'status': 'ready', 'live': False, 'provider_calls': 0})
                if url.path in {'/api/snapshot', '/api/export'}:
                    query = parse_qs(url.query, keep_blank_values=True)
                    scope = parse_scope(query)
                    if url.path == '/api/snapshot':
                        return self.reply(200, service.snapshot(**scope))
                    kind = query.get('kind', ['bars'])[0]
                    return self.reply(200, service.export(scope, kind), 'text/csv; charset=utf-8', f"{scope['source']}_{scope['symbol']}_{scope['session']}_{kind}.csv")
                return self.reply(404, {'error': 'Route not found'})
            except (DataContractError, ValueError, KeyError, OSError, sqlite3.Error) as exc:
                return self.reply(400, {'error': str(exc)})

        def do_POST(self):
            origin = f'http://127.0.0.1:{self.server.server_port}'
            if not self._host() or self.headers.get('Origin') != origin or not secrets.compare_digest(self.headers.get('X-Workbench-Token', ''), token):
                return self.reply(403, {'error': 'Local origin/token required'})
            if self.path != '/api/review':
                return self.reply(404, {'error': 'Route not found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384 or self.headers.get('Content-Type') != 'application/json' or self.headers.get('Transfer-Encoding'):
                    return self.reply(400, {'error': 'Invalid bounded JSON request'})
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or set(body) != {'scope', 'review'}:
                    raise DataContractError('Invalid review request')
                return self.reply(200, service.review(body['scope'], body['review']))
            except JournalConflict as exc:
                return self.reply(409, {'error': str(exc)})
            except (DataContractError, ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
                return self.reply(400, {'error': str(exc)})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server
