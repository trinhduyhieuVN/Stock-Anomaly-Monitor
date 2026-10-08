"""VCI raw adapter: exact source, immutable responses, bounded public HTTP calls."""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import pandas as pd

from .io import DataContractError, sha256

ENDPOINT = "https://trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart"
ADAPTER_VERSION = "vci-raw-1"
ARRAYS = {"t": "timestamp", "o": "open", "h": "high", "l": "low",
          "c": "close", "v": "volume", "accumulatedVolume": "accumulated_volume"}


class ProviderError(ValueError):
    """A source or response failure without credentials in its message."""


def parse_vci_minutes(payload, symbol):
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict) or payload[0].get("symbol") != symbol:
        raise ProviderError("Unexpected VCI symbol/envelope")
    row = payload[0]
    if any(not isinstance(row.get(k), list) for k in ARRAYS):
        raise ProviderError("Missing VCI minute arrays")
    if len({len(row[k]) for k in ARRAYS}) != 1:
        raise ProviderError("VCI minute arrays have unequal lengths")
    frame = pd.DataFrame({target: row[key] for key, target in ARRAYS.items()})
    frame["raw_timestamp"] = frame.timestamp.map(str)
    if "accumulatedValue" in row:
        if not isinstance(row["accumulatedValue"], list) or len(row["accumulatedValue"]) != len(frame):
            raise ProviderError("VCI accumulatedValue array has unequal length")
        frame["accumulated_value"] = pd.to_numeric(pd.Series(row["accumulatedValue"]), errors="coerce")
    epochs = pd.to_numeric(frame.timestamp, errors="coerce")
    integer_epochs = epochs.notna() & np.isfinite(epochs) & epochs.mod(1).eq(0) & ~frame.timestamp.map(lambda v: isinstance(v, bool))
    frame["timestamp"] = pd.to_datetime(epochs.where(integer_epochs), unit="s", utc=True, errors="coerce")
    for key in set(ARRAYS.values()) - {"timestamp"}:
        # Invalid values survive as NaN and are rejected with row-level reasons.
        boolean = frame[key].map(lambda v: isinstance(v, bool))
        frame[key] = pd.to_numeric(frame[key], errors="coerce").mask(boolean)
    frame["symbol"] = symbol
    frame["source"] = "VCI"
    frame["raw_row"] = np.arange(len(frame))
    return frame


def load_snapshot(path, root, raw_root=None):
    """Recheck both raw and metadata hashes before interpreting any snapshot."""
    path, root = Path(path).resolve(), Path(root).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "raw_source_snapshot" or manifest.get("source") != "VCI" or manifest.get("source_interval") != "1m":
        raise ProviderError("Expected a VCI raw minute snapshot")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ProviderError("Empty source snapshot")
    frames, provenance = [], []
    for item in files:
        raw = (root / item["raw_path"]).resolve()
        metadata = (root / item["metadata_path"]).resolve() if item.get("metadata_path") else raw.with_name("metadata.json")
        storage = Path(raw_root).resolve() if raw_root else (root / "data/raw").resolve()
        if not raw.is_relative_to(storage) or not metadata.is_relative_to(storage):
            raise ProviderError("Snapshot paths must stay inside configured raw storage")
        if sha256(raw) != item["raw_sha256"] or sha256(metadata) != item["metadata_sha256"]:
            raise DataContractError("Source or metadata checksum mismatch")
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        http = meta.get("http", [])
        if meta.get("symbol") != item["symbol"] or not http or any(r.get("url") != ENDPOINT or r.get("method") != "POST" for r in http):
            raise ProviderError("Raw metadata source/symbol differs from snapshot")
        matching = [r for r in http if r.get("sha256") == item["raw_sha256"] and r.get("status") == 200]
        if len(matching) != 1 or matching[0].get("public_candle_query", {}).get("timeFrame") != "ONE_MINUTE" or matching[0]["public_candle_query"].get("symbols") != [item["symbol"]]:
            raise ProviderError("Raw response is not a successful ONE_MINUTE query for the exact symbol")
        if (meta.get("requested_start"), meta.get("requested_end")) != (manifest["requested_start"], manifest["requested_end"]):
            raise ProviderError("Raw metadata date range differs from snapshot")
        expected_cutoff = int((pd.Timestamp(manifest["requested_end"], tz="Asia/Ho_Chi_Minh") + pd.Timedelta(days=1)).timestamp())
        if matching[0]["public_candle_query"].get("to") != expected_cutoff:
            raise ProviderError("Query cutoff differs from inclusive local end date")
        frame = parse_vci_minutes(json.loads(raw.read_bytes()), item["symbol"])
        frame["raw_path"] = item["raw_path"]
        frames.append(frame)
        provenance.append({"symbol": item["symbol"], "raw_path": item["raw_path"],
            "raw_sha256": item["raw_sha256"], "metadata_sha256": item["metadata_sha256"],
            "retrieved_at_utc": meta["retrieved_at_utc"], "endpoint": ENDPOINT,
            "query": matching[0]["public_candle_query"], "packages_at_fetch": meta.get("packages", {})})
    if len({f["symbol"] for f in files}) != len(files) or set(manifest["symbols"]) != {f["symbol"] for f in files}:
        raise ProviderError("Duplicate or missing symbol snapshots")
    return pd.concat(frames, ignore_index=True), manifest, provenance


def public_transport(payload, timeout):
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, new_url):
            return None

    request = Request(ENDPOINT, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "stock-anomaly-monitor/0.1 (historical research)",
                 "Origin": "https://trading.vietcap.com.vn"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def fetch_vci_raw(raw_root, symbol, start, end, count_back, *, transport=public_transport,
                  sleep=time.sleep, max_attempts=3, backoff_seconds=1, max_wait_seconds=10, timeout=30):
    """One symbol/request scope. Never bypass access/quota errors or switch source."""
    if not isinstance(symbol, str) or not symbol.isalnum() or symbol != symbol.upper():
        raise ProviderError("Invalid symbol")
    if not isinstance(count_back, int) or not 1 <= count_back <= 100000:
        raise ProviderError("countBack must be 1..100000")
    if not 1 <= max_attempts <= 3 or not 0 < backoff_seconds <= max_wait_seconds <= 10 or not 0 < timeout <= 30:
        raise ProviderError("HTTP limits exceed the adapter budget")
    first = pd.Timestamp(start, tz="Asia/Ho_Chi_Minh")
    stop = pd.Timestamp(end, tz="Asia/Ho_Chi_Minh") + pd.Timedelta(days=1)
    if first >= stop or stop > pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").normalize():
        raise ProviderError("Use ordered, fully historical local dates")
    directory = Path(raw_root) / "vci" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{symbol}_{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    query = {"symbols": [symbol], "timeFrame": "ONE_MINUTE", "to": int(stop.timestamp()), "countBack": count_back}
    meta = {"provider": "VCI", "symbol": symbol, "interval": "1m", "requested_start": start,
        "requested_end": end, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_version": ADAPTER_VERSION, "http": [], "status": "failed", "is_synthetic": False}
    result_path = None
    error = "Request budget exhausted"
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                status, headers, body = transport(query, timeout)
            except (URLError, TimeoutError, OSError):
                meta["http"].append({"url": ENDPOINT, "method": "POST", "status": None, "error": "transport_error", "attempt": attempt})
                if attempt == max_attempts:
                    break
                sleep(min(backoff_seconds * 2 ** (attempt - 1), max_wait_seconds))
                continue
            response = directory / f"response_{attempt:02d}.json"
            with response.open("xb") as stream:
                stream.write(body)
            safe_headers = {k.lower(): str(v) for k, v in headers.items() if k.lower() in {"retry-after", "content-type", "date"}}
            meta["http"].append({"url": ENDPOINT, "method": "POST", "status": status,
                "body_file": response.name, "sha256": sha256(response), "bytes": len(body),
                "headers": safe_headers, "public_candle_query": query})
            if status == 200:
                # Permission messages returned as HTTP 200 fail envelope parsing, with no retry.
                try:
                    frame = parse_vci_minutes(json.loads(body), symbol)
                except (ValueError, TypeError):
                    error = "Invalid or access-denied source payload; no retry"
                    break
                if frame.empty:
                    error = "Empty source response; no retry"
                    break
                meta["status"] = "received"
                result_path = response
                break
            if status in {401, 403}:
                error = "Source access denied; stopped without retry"
                break
            if status not in {429, 500, 502, 503, 504}:
                error = f"Non-retryable HTTP {status}"
                break
            if attempt == max_attempts:
                break
            wait = min(backoff_seconds * 2 ** (attempt - 1), max_wait_seconds)
            if status == 429:
                try:
                    retry_after = float(safe_headers.get("retry-after", ""))
                except ValueError:
                    error = "Quota exceeded without numeric Retry-After; stopped"
                    break
                if not 0 <= retry_after <= max_wait_seconds:
                    error = "Quota wait exceeds adapter budget; stopped"
                    break
                wait = max(wait, retry_after)
            sleep(wait)
    finally:
        meta["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        if result_path is None:
            meta["error"] = error
        with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if result_path is None:
        raise ProviderError(error)
    return result_path, directory / "metadata.json"
