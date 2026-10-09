"""Project-local append-only SQLite journal with explicit availability queries."""
from contextlib import closing, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from stock_monitor.io import DataContractError
from .contracts import utc_time, validate_event, validate_news, validate_review
from .forecast import validate_forecast
from .synthetic import content_hash

COLLECTIONS = {"events", "forecasts", "assessments", "news", "reviews"}
APPLICATION_ID = 0x53414D32
SCHEMA_VERSION = 2


class JournalConflict(DataContractError):
    """An identity/version already exists with different semantic content."""


class Journal:
    def __init__(self, path, *, busy_timeout_ms=2000):
        if type(busy_timeout_ms) is not int or not 0 <= busy_timeout_ms <= 5000:
            raise DataContractError("SQLite timeout must be bounded to 0..5000ms")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = busy_timeout_ms
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            app_id = conn.execute("PRAGMA application_id").fetchone()[0]
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            if version == 0 and app_id == 0 and not tables:
                conn.execute("CREATE TABLE records (collection TEXT NOT NULL CHECK(collection IN ('events','forecasts','assessments','news','reviews')), record_id TEXT NOT NULL, version INTEGER NOT NULL CHECK(version>0), event_id TEXT, event_version INTEGER, symbol TEXT, available TEXT NOT NULL, payload TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(collection,record_id,version))")
                conn.execute("CREATE INDEX availability_idx ON records(collection,available,symbol)")
                for operation in ("UPDATE", "DELETE"):
                    conn.execute(f"CREATE TRIGGER forbid_{operation.lower()} BEFORE {operation} ON records BEGIN SELECT RAISE(ABORT,'append-only journal'); END")
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
            elif version != SCHEMA_VERSION or app_id != APPLICATION_ID or [row[0] for row in tables] != ["records"]:
                raise DataContractError("Unknown journal schema; no automatic migration")
            else:
                triggers = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
                if triggers != {"forbid_update", "forbid_delete"}:
                    raise DataContractError("Journal immutability triggers missing")
            conn.commit()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=self.timeout / 1000)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self.timeout}")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def transaction(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _decode(row):
        value = json.loads(row["payload"])
        if content_hash(value) != row["sha256"]:
            raise DataContractError("Stored journal payload hash mismatch")
        return value

    def _find(self, conn, collection, record_id, version=1):
        row = conn.execute("SELECT * FROM records WHERE collection=? AND record_id=? AND version=?",
                           (collection, record_id, version)).fetchone()
        return self._decode(row) if row else None

    def _validate(self, conn, collection, value):
        if collection == "events":
            result = validate_event(value)
            identity, version = result["event_id"], result["version"]
            original = self._find(conn, "events", identity)
            if original is None and (version != 1 or result["stage"] != "initial"):
                raise JournalConflict("First event version must be initial version1")
            if original is not None:
                if any(result[name] != original[name] for name in ("symbol", "session_id", "source_mode", "model_run")):
                    raise JournalConflict("Event ID collision across immutable identity")
                last = conn.execute("SELECT * FROM records WHERE collection='events' AND record_id=? ORDER BY version DESC LIMIT 1", (identity,)).fetchone()
                existing = self._find(conn, "events", identity, version)
                if existing is None:
                    if version != last["version"] + 1 or utc_time(result["available_at"]) < utc_time(last["available"]):
                        raise JournalConflict("Versions must be sequential and availability monotonic")
                    if result["stage"] == "post_event" and any(result[name] != original[name] for name in
                            ("kind", "score", "threshold", "data_sha256", "event_start", "event_end", "model_sha256")):
                        raise JournalConflict("Post-event cannot rewrite initial decision/input")
            return result, identity, version, identity, version, result["symbol"], result["available_at"]
        if collection == "forecasts":
            result = validate_forecast(value)
            return result, result["forecast_id"], 1, None, None, result["symbol"], result["available_at"]
        if collection == "news":
            result = validate_news(value)
            return result, result["news_id"], 1, None, None, result["symbol"], result["available_at"]
        if collection == "reviews":
            if not isinstance(value, dict):
                raise DataContractError("Review must be a record")
            event = self._find(conn, "events", value.get("event_id"), value.get("event_version"))
            if event is None:
                raise DataContractError("Review requires a persisted event/version")
            result = validate_review(value, event=event)
            return result, result["review_id"], 1, event["event_id"], event["version"], event["symbol"], result["available_at"]
        if collection == "assessments":
            return self._assessment(conn, value)
        raise DataContractError("Unknown journal collection")

    def _assessment(self, conn, value):
        required = {"assessment_key", "forecast_id", "status", "as_of", "outcome_sha256", "post_evidence", "source_mode", "origin_revision_observed"}
        if not isinstance(value, dict) or set(value) != required:
            raise DataContractError("Invalid assessment schema")
        result = deepcopy(value)
        forecast = self._find(conn, "forecasts", result["forecast_id"])
        if forecast is None or result["source_mode"] != forecast["source_mode"]:
            raise DataContractError("Assessment requires a persisted matching forecast")
        cutoff = utc_time(result["as_of"])
        if cutoff < utc_time(forecast["available_at"]):
            raise DataContractError("Assessment precedes forecast")
        if result["origin_revision_observed"] is not None and type(result["origin_revision_observed"]) is not bool:
            raise DataContractError("Invalid revision diagnostic")
        status = result["status"]
        if status not in {"pending", "pending_missing_bar", "censored_boundary", "censored_missing_bar", "mature"}:
            raise DataContractError("Unknown assessment status")
        if forecast["horizon_status"] == "censored_boundary" and status != "censored_boundary":
            raise DataContractError("Boundary horizon cannot become mature/pending")
        if status in {"mature", "pending_missing_bar"} and cutoff < utc_time(forecast["target_bar_end"]):
            raise DataContractError("Outcome status precedes target")
        if status == "mature":
            evidence = result["post_evidence"]
            # Reuse the strict post-event contract for horizon/time/formula checks.
            endpoint = forecast["origin_bar_end"]
            event = {"schema_version": 2, "event_unit": "bar_5m", "event_id": "assessment-validator", "version": 1,
                     "symbol": forecast["symbol"], "session_id": forecast["session_id"], "segment_id": forecast["segment_id"],
                     "event_start": endpoint, "event_end": endpoint, "as_of": result["as_of"], "detected_at": result["as_of"],
                     "available_at": result["as_of"], "stage": "post_event", "kind": "normal", "source_mode": "SYNTHETIC",
                     "model_run": "zero-return", "model_sha256": forecast["model_sha256"], "data_sha256": forecast["input_sha256"],
                     "data_revision": str(forecast["origin_revision"]), "reason_codes": ["mature"], "score": 0., "threshold": 1.,
                     "decision_method": "baseline", "post_evidence": evidence}
            validate_event(event)
            if (not isinstance(result["outcome_sha256"], str) or len(result["outcome_sha256"]) != 64 or
                    any(char not in "0123456789abcdef" for char in result["outcome_sha256"]) or
                    evidence["forecast_sha256"] != content_hash(forecast) or evidence["prediction"] != forecast["prediction"] or
                    evidence["horizon_bars"] != forecast["horizon_bars"] or utc_time(evidence["forecast_created_at"]) != utc_time(forecast["created_at"])):
                raise DataContractError("Assessment forecast/evidence mismatch")
        elif result["post_evidence"] is not None or result["outcome_sha256"] is not None:
            raise DataContractError("Unmature assessment cannot contain realized/residual")
        if status == "censored_boundary" and forecast["horizon_status"] != "censored_boundary":
            raise DataContractError("Incorrect boundary censoring")
        if status == "censored_missing_bar" and cutoff < utc_time(forecast["period_end"]):
            raise DataContractError("Missing path cannot be final before period end")
        key = content_hash({"forecast_sha256": content_hash(forecast), "status": status, "outcome_sha256": result["outcome_sha256"]})
        if key != result["assessment_key"]:
            raise DataContractError("Assessment key mismatch")
        return result, key, 1, None, None, forecast["symbol"], result["as_of"]

    def _append(self, conn, collection, value):
        result, identity, version, event_id, event_version, symbol, available = self._validate(conn, collection, value)
        existing = self._find(conn, collection, identity, version)
        if existing is not None:
            if collection == "assessments":
                # Same outcome observed at later T is not a new outcome version.
                comparable = lambda row: {key: value for key, value in row.items() if key not in {"as_of", "origin_revision_observed"}}
                if comparable(existing) != comparable(result) or utc_time(result["as_of"]) < utc_time(existing["as_of"]):
                    raise JournalConflict("Conflicting outcome under stable key")
            elif existing != result:
                raise JournalConflict("Conflicting payload under immutable ID/version")
            return False
        payload = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        conn.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)", (collection, identity, version, event_id, event_version,
                     symbol, utc_time(available).isoformat(), payload, content_hash(result)))
        return True

    def append_many(self, records):
        with self.transaction() as conn:
            return [self._append(conn, collection, value) for collection, value in records]

    def append(self, collection, value):
        return self.append_many([(collection, value)])[0]

    def query(self, collection, *, as_of, symbol=None, event_id=None, latest=False):
        if collection not in COLLECTIONS or type(latest) is not bool or latest and collection != "events":
            raise DataContractError("Invalid journal query")
        clauses, params = ["collection=?", "available<=?"], [collection, utc_time(as_of).isoformat()]
        for column, value in (("symbol", symbol), ("event_id", event_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM records WHERE " + " AND ".join(clauses) + " ORDER BY available,record_id,version", params).fetchall()
            values = [self._decode(row) for row in rows]
        finally:
            conn.close()
        if latest:
            versions = {}
            for value in values:
                if value["event_id"] not in versions or value["version"] > versions[value["event_id"]]["version"]:
                    versions[value["event_id"]] = value
            return list(versions.values())
        return values
