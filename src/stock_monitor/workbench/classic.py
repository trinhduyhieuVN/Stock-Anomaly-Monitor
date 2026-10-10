"""Classic static surface over the accepted API boundary, unchanged guards."""
from urllib.parse import urlsplit

from stock_monitor.workbench.http import STATIC, make_server as integrated_server


def make_server(service, port=8512):
    server = integrated_server(service, port)
    inherited = server.RequestHandlerClass
    public = service.root / 'web/research-workbench-classic/public'

    class ClassicHandler(inherited):
        def do_GET(self):
            url = urlsplit(self.path)
            if url.path in STATIC and not url.query:
                if not self._host():
                    return self.reply(403, {'error': 'Invalid local Host'})
                name, mime = STATIC[url.path]
                try:
                    return self.reply(200, (public / name).read_bytes(), mime)
                except OSError as exc:
                    return self.reply(400, {'error': str(exc)})
            return super().do_GET()

    server.RequestHandlerClass = ClassicHandler
    return server
