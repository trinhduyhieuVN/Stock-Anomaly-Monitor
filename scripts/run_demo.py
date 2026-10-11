"""Run the existing replay workbench locally without a provider or coordinator."""
import argparse
from pathlib import Path
import threading

from stock_monitor.workbench.smooth import SmoothWorkbench, make_server

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8515)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535 or not 0 < args.minutes <= 90:
        parser.error('Use port 1024–65535 and a duration greater than zero, at most 90 minutes')
    for name in ('index.html', 'app.js', 'style.css'):
        if not (ROOT / 'web/research-workbench-replay/public' / name).is_file():
            raise RuntimeError('Missing demo interface: ' + name)
    service = SmoothWorkbench(ROOT, ROOT / '.local/workbench/smooth')
    if 'synthetic' not in service.bundles:
        raise RuntimeError('Synthetic fixture is unavailable: ' + service.errors.get('synthetic', 'unknown'))
    if args.check_only:
        print('Demo ready: synthetic fixture verified; no server or provider started.')
        return 0
    try:
        server = make_server(service, args.port)
    except OSError as exc:
        raise RuntimeError(f'Port {args.port} unavailable. Use --port with another port.') from exc
    timer = threading.Timer(args.minutes * 60, server.shutdown)
    timer.daemon = True
    timer.start()
    print(f'http://127.0.0.1:{server.server_port} — Ctrl+C to stop; {args.minutes:g} minute limit.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
