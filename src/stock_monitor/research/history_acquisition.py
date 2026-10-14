"""Durable one-attempt history collection; stdlib worker has a hard process deadline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone


class AcquisitionError(ValueError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable_json(path, value):
    """Exclusive durable receipt. An interrupted receipt blocks further requests."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc_now():
    return datetime.now(timezone.utc)


def query_for(protocol, symbol):
    if symbol not in protocol['symbols']:
        raise AcquisitionError('Unregistered symbol')
    end = datetime.fromisoformat(protocol['requested_end']).replace(tzinfo=timezone(timedelta(hours=7))) + timedelta(days=1)
    if end > utc_now().replace(hour=0, minute=0, second=0, microsecond=0):
        raise AcquisitionError('Only completed historical dates are allowed')
    return {'symbols': [symbol], 'timeFrame': 'ONE_MINUTE', 'to': int(end.timestamp()),
            'countBack': protocol['acquisition_budget']['count_back']}


def stream_response(response, target, maximum_bytes):
    """Never read beyond the declared cap; reject a cap-sized unknown-length body."""
    headers = {k.lower(): str(v) for k, v in response.headers.items()}
    length = headers.get('content-length')
    if length is not None and (not length.isdigit() or int(length) > maximum_bytes):
        raise AcquisitionError('Response Content-Length exceeds budget or is invalid')
    if headers.get('content-encoding', 'identity').lower() != 'identity':
        raise AcquisitionError('Unexpected compressed response')
    received = 0
    with Path(target).open('xb') as stream:
        while received < maximum_bytes:
            chunk = response.read1(min(65536, maximum_bytes - received))
            if not chunk: break
            stream.write(chunk); received += len(chunk)
        stream.flush(); os.fsync(stream.fileno())
    if received == maximum_bytes and (length is None or int(length) != received):
        raise AcquisitionError('Response reached byte budget')
    if length is not None and int(length) != received:
        raise AcquisitionError('Truncated response')
    return {'status': response.status if hasattr(response, 'status') else response.code,
        'headers': {k: v for k, v in headers.items() if k in {'retry-after', 'content-type', 'date'}},
        'bytes': received, 'sha256': checksum(target)}


def fetch_worker(request_path, raw_path, result_path):
    from urllib.request import HTTPRedirectHandler, Request, build_opener
    from urllib.error import HTTPError
    # The endpoint is pinned in the request by the parent, which reuses providers.ENDPOINT.
    item = read(request_path)
    if item['endpoint'] != 'https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart':
        raise AcquisitionError('Endpoint mismatch')
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs): return None
    request = Request(item['endpoint'], data=json.dumps(item['query']).encode(), method='POST',
        headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'Accept-Encoding': 'identity',
            'Origin': 'https://trading.vietcap.com.vn', 'User-Agent': 'stock-anomaly-monitor/0.1 (historical research)'})
    try:
        response = build_opener(NoRedirect()).open(request, timeout=item['timeout_seconds'])
    except HTTPError as exc:
        response = exc
    with response:
        result = stream_response(response, raw_path, item['maximum_bytes'])
    immutable_json(result_path, result)


def process_transport(request_path, raw_path, result_path, timeout):
    """Parent kills only its own child at the deadline, including slow headers/body."""
    command = [sys.executable, '-X', 'utf8', '-m', __name__, '--fetch', str(request_path), str(raw_path), str(result_path)]
    started = time.monotonic()
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=timeout)
        return {'command': command, 'elapsed_seconds': time.monotonic() - started, 'exit_code': proc.returncode,
            'outcome': 'returned' if proc.returncode == 0 else 'transport_error',
            'stderr': proc.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {'command': command, 'elapsed_seconds': time.monotonic() - started, 'exit_code': 124,
            'outcome': 'timeout', 'stderr': 'Owned transport process exceeded hard timeout; no retry'}


def collect(directory, protocol, registration_sha256, *, transport=process_transport, clock=utc_now):
    """A durable claim consumes an attempt even if interrupted before the request."""
    from stock_monitor.providers import ENDPOINT, parse_vci_minutes
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    budget = protocol['acquisition_budget']; state_path = directory / 'budget.json'
    if not state_path.exists():
        started = clock()
        immutable_json(state_path, {'registration_sha256': registration_sha256,
            'started_at': started.isoformat(), 'deadline': (started + timedelta(seconds=budget['wall_seconds'])).isoformat()})
    state = read(state_path)
    if state['registration_sha256'] != registration_sha256:
        raise AcquisitionError('Registration changed; attempts cannot be reset')
    deadline = datetime.fromisoformat(state['deadline']); results = []; stop = None
    total_bytes = total_rows = 0
    for symbol in protocol['symbols']:
        slot = directory / symbol; claim = slot / 'request.json'; receipt = slot / 'receipt.json'
        if slot.exists():
            if not claim.exists() or not receipt.exists():
                stop = 'interrupted_attempt_requires_audit_no_repoll'; break
            prior = read(receipt); request = read(claim)
            if request['query'] != query_for(protocol, symbol) or request['registration_sha256'] != registration_sha256:
                raise AcquisitionError('Saved request binding mismatch')
            raw = slot / 'response.json'
            if prior.get('sha256') and (not raw.exists() or checksum(raw) != prior['sha256']):
                raise AcquisitionError('Saved raw hash mismatch')
            results.append(prior); total_bytes += prior['bytes']; total_rows += prior['source_rows']
            if prior['outcome'] != 'received': stop = prior['outcome']; break
            continue
        remaining = (deadline - clock()).total_seconds()
        if remaining <= 0: stop = 'wall_budget_exhausted'; break
        if len(results) >= budget['maximum_requests']: stop = 'request_budget_exhausted'; break
        maximum_bytes = min(budget['maximum_response_bytes_per_request'], budget['maximum_total_response_bytes'] - total_bytes)
        if maximum_bytes <= 0: stop = 'byte_budget_exhausted'; break
        slot.mkdir(exist_ok=False)
        timeout = min(budget['timeout_seconds'], remaining)
        request = {'symbol': symbol, 'endpoint': ENDPOINT, 'query': query_for(protocol, symbol),
            'registration_sha256': registration_sha256, 'timeout_seconds': timeout, 'maximum_bytes': maximum_bytes,
            'started_at': clock().isoformat(), 'requested_start': protocol['requested_start'], 'requested_end': protocol['requested_end']}
        immutable_json(claim, request)
        raw = slot / 'response.json'; result_path = slot / 'http.json'
        execution = transport(claim, raw, result_path, timeout)
        item = {'symbol': symbol, 'attempt': 1, 'execution': execution, 'completed_at': clock().isoformat(),
            'bytes': raw.stat().st_size if raw.exists() else 0, 'sha256': checksum(raw) if raw.exists() else None,
            'source_rows': 0, 'outcome': execution['outcome']}
        if execution['exit_code'] == 0:
            try:
                http = read(result_path); item['http'] = http
                if item['bytes'] > maximum_bytes or http['sha256'] != item['sha256'] or http['bytes'] != item['bytes']:
                    raise AcquisitionError('Transport hash/byte budget mismatch')
                if http['status'] != 200:
                    item['outcome'] = 'access_denied' if http['status'] in {401, 403} else 'quota' if http['status'] == 429 else 'http_error'
                else:
                    payload = read(raw)
                    # Envelope and all arrays are validated by the existing source adapter.
                    frame = parse_vci_minutes(payload, symbol); item['source_rows'] = len(frame)
                    if frame.empty or len(frame) > budget['count_back'] or total_rows + len(frame) > budget['maximum_source_rows']:
                        raise AcquisitionError('Empty payload or source-row budget exceeded')
                    item['outcome'] = 'received'
            except (ValueError, TypeError, KeyError, OSError) as exc:
                item['outcome'] = 'invalid_payload'; item['error'] = str(exc)[:400]
        immutable_json(receipt, item); results.append(item)
        total_bytes += item['bytes']; total_rows += item['source_rows']
        if item['outcome'] != 'received': stop = item['outcome']; break
    return {'results': results, 'stop_reason': stop, 'attempts_consumed': len(list(directory.glob('*/request.json'))),
        'response_bytes': total_bytes, 'source_rows': total_rows, 'all_received': len(results) == len(protocol['symbols']) and stop is None}


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--fetch': raise SystemExit('Internal bounded transport worker only')
    fetch_worker(*sys.argv[2:])
