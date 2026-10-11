"""Offline projections only. No provider, fitting, or edits to inherited journals."""
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import threading

import numpy as np
import pandas as pd

from stock_monitor.io import DataContractError, sha256
from stock_monitor.services import MODELS
from stock_monitor.research.history import load_historical_bundle, historical_session, historical_csv
from stock_monitor.research.history_review import review_snapshot
from stock_monitor.research.ai_demo import AI_MODELS, AI_SHA, load_ai, verify_ai, project_ai
from stock_monitor.research.ai_contracts import AIJournal, structural_event
from stock_monitor.research.analysis import summarize_bars, summary_text, explain_method
from stock_monitor.upgrade.contracts import utc_time, validate_review
from stock_monitor.upgrade.journal import Journal, JournalConflict
from stock_monitor.upgrade.ui_service import load_ui_bundle, replay_session, event_csv, forecast_view

# Binding inherited from the accepted historical demo, not self-pinned at runtime.
HISTORY_SHA = 'ab42fa8720162c301a9b4a6affecb767986c62147ab6d649d7c406a1e5811d1e'
SYNTHETIC_SHA = 'e7ccd397730889a424305c275e4a87936fe8cdab4e0ca6ca9437f78aaa92effc'
SOURCES = {'historical', 'synthetic'}


def model_is_ai(scope):
    return scope.get('source') == 'historical' and scope.get('model') in AI_MODELS.values()


def clean(value):
    """Strict JSON: unavailable values are null, never NaN or zero."""
    if isinstance(value, pd.DataFrame):
        return clean(value.to_dict('records'))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Workbench:
    def __init__(self, root, journal_root=None):
        self.root = Path(root).resolve()
        self.journal_root = Path(journal_root or self.root / '.local/workbench').resolve()
        allowed = (self.root / '.local/workbench', self.root / 'reports/research')
        if not any(self.journal_root.is_relative_to(p) for p in allowed):
            raise DataContractError('Journal must use the new workbench or test report namespace')
        self.lock = threading.RLock()
        self.bundles, self.errors, self.cache = {}, {}, OrderedDict()
        self.ai_bundle = None
        for source, loader in (
            ('historical', lambda: load_historical_bundle(self.root, expected_manifest_sha256=HISTORY_SHA)),
            ('synthetic', self._load_synthetic),
        ):
            try:
                self.bundles[source] = loader()
            except (DataContractError, OSError, ValueError, KeyError) as exc:
                self.errors[source] = str(exc)
        self.synthetic_pins = {}
        if 'synthetic' in self.bundles:
            config = self.root / 'configs/upgrade_ui.json'
            manifest = self.root / 'demo/upgrade_ui/manifest.json'
            names = [config, manifest] + [manifest.parent / row['name'] for row in self.bundles['synthetic']['manifest']['files']]
            self.synthetic_pins = {p: sha256(p) for p in names}

    def _load_synthetic(self):
        bundle = load_ui_bundle(self.root)
        if bundle['key'] != SYNTHETIC_SHA:
            raise DataContractError('Synthetic manifest is not the approved frozen bundle')
        return bundle

    def catalog(self):
        return {'sources': {s: {'available': s in self.bundles, 'error': self.errors.get(s),
            'sessions': self.bundles[s]['sessions'] if s in self.bundles else [],
            'symbols': self.bundles[s].get('symbols', ['FPT', 'HPG', 'VCB']) if s in self.bundles else [],
            'models': {**MODELS, **(AI_MODELS if (self.root / 'demo/ai_outline.local.json').is_file() else {})} if s == 'historical' else {'Điểm / đoạn v2': 'v2'}} for s in sorted(SOURCES)},
            'live': False, 'reviewer_kinds': ['human', 'ai_assisted']}

    def _scope(self, source, symbol, session, model):
        if source not in SOURCES or source not in self.bundles:
            raise DataContractError('Nguồn chưa sẵn sàng: ' + self.errors.get(source, 'không hợp lệ'))
        bundle = self.bundles[source]
        if symbol not in bundle.get('symbols', ['FPT', 'HPG', 'VCB']) or session not in bundle['sessions']:
            raise DataContractError('Unknown symbol or session')
        if model not in ([*MODELS.values(), *AI_MODELS.values()] if source == 'historical' else ['v2']):
            raise DataContractError('Model cannot be used with this source')
        if model in AI_MODELS.values():
            if self.ai_bundle is None:
                self.ai_bundle = load_ai(self.root)
            verify_ai(self.ai_bundle)
        if source == 'synthetic':
            for path, digest in self.synthetic_pins.items():
                if not path.is_file() or sha256(path) != digest:
                    raise DataContractError('Synthetic artifact checksum mismatch: ' + path.name)
        return bundle

    def timeline(self, source, symbol, session, model):
        bundle = self._scope(source, symbol, session, model)
        if source == 'historical':
            # This also verifies all transitive inherited pins on every request.
            history = historical_session(bundle, symbol=symbol, session_id=session, model='lstm' if model in AI_MODELS.values() else model)
            return [t.isoformat() for t in history['bars'].timestamp]
        return sorted({row['available_at'] for row in bundle['rows'] if row['symbol'] == symbol and row['session_id'] == session}, key=utc_time)

    def _journal(self, source, bundle, model):
        # Flat full-SHA identity avoids Windows MAX_PATH failures for SQLite WAL
        # inside long audit directories, without weakening source/model isolation.
        identity = hashlib.sha256('|'.join((source, AI_SHA if model in AI_MODELS.values() else bundle['key'], model)).encode()).hexdigest()
        journal_type = AIJournal if model in AI_MODELS.values() else Journal
        return journal_type(self.journal_root / (identity + '.sqlite'))

    def _snapshot(self, source, symbol, session, model, replay=False, step=None):
        if not isinstance(replay, bool):
            raise DataContractError('Replay must be boolean')
        bundle = self._scope(source, symbol, session, model)
        timeline = self.timeline(source, symbol, session, model)
        if not timeline:
            raise DataContractError('Session has no observations')
        if step is None:
            step = len(timeline)
        if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= len(timeline):
            raise DataContractError('Invalid replay step')
        if not replay and step != len(timeline):
            raise DataContractError('Prefix requires explicit Replay mode')
        cutoff = timeline[step - 1]
        journal = self._journal(source, bundle, model)
        history = None
        if source == 'historical':
            history = historical_session(bundle, symbol=symbol, session_id=session, model='lstm' if model in AI_MODELS.values() else model, as_of=cutoff)
            snap = project_ai(self.ai_bundle, bundle, history, journal, model=model, replay=replay) if model in AI_MODELS.values() else review_snapshot(bundle, history, journal, replay=replay)
        else:
            key = (bundle['key'], symbol, session, cutoff)
            if key not in self.cache:
                self.cache[key] = replay_session(bundle, journal, symbol=symbol, session_id=session, as_of=cutoff)
                if len(self.cache) > 12:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            snap = dict(self.cache[key])
            review_time = cutoff if replay else datetime.now(timezone.utc).isoformat()
            versions = {(e['event_id'], e['version']) for e in snap['versions']}
            snap['reviews'] = [r for r in journal.query('reviews', as_of=review_time, symbol=symbol)
                               if (r['event_id'], r['event_version']) in versions]
        return bundle, journal, history, snap, timeline, cutoff

    def snapshot(self, source, symbol, session, model, replay=False, step=None):
        with self.lock:
            bundle, _, history, snap, timeline, cutoff = self._snapshot(source, symbol, session, model, replay, step)
            bars = clean(snap['bars'])
            details = {}
            for event in snap['events']:
                detail = snap['details'].get(event['event_id'], {})
                end = utc_time(event.get('decision_bar_end', event['event_end']))
                # Selected event context must stop at that event, even in full-session view.
                frame = snap['bars']
                prefix = frame.loc[pd.to_datetime(frame.timestamp, utc=True).le(end)]
                analysis = summarize_bars(prefix, symbol=symbol, session_id=session, as_of=event['available_at'],
                                          source_mode='HISTORICAL_REPLAY' if source == 'historical' else 'SYNTHETIC')
                details[event['event_id']] = {'description': summary_text(analysis), 'analysis': analysis,
                    'method': detail['method'] if model in AI_MODELS.values() else explain_method(event, detail, model=model if source == 'historical' else None),
                    'evidence': detail.get('evidence', detail),
                    'initial_event': detail.get('initial_event'), 'versions': detail.get('versions', []),
                    'forecasts': detail.get('forecasts', []) if model in AI_MODELS.values() else forecast_view(snap, event, bundle) if source == 'synthetic' else []}
            return clean({'source': source, 'source_mode': 'HISTORICAL_REPLAY' if source == 'historical' else 'SYNTHETIC',
                'symbol': symbol, 'session': session, 'model': model, 'replay': replay, 'as_of': cutoff,
                'timeline': timeline, 'step': timeline.index(cutoff) + 1, 'bars': bars, 'events': snap['events'],
                'reviews': snap['reviews'], 'details': details,
                'episodes': snap.get('episodes', []), 'versions': snap.get('versions', []),
                'score_duration_seconds': snap.get('score_duration_seconds'), 'model_metadata': snap.get('model_metadata'),
                'policy': history['data_policy'] if history else {'classification': 'synthetic', 'source_rights_status': 'synthetic', 'distributable': True},
                'provenance': ({**history['provenance'], 'model_run': 'ai_outline:' + model, 'ai_manifest_sha256': AI_SHA}
                               if model in AI_MODELS.values() else history['provenance']) if history else {'manifest_sha256': bundle['key'], 'model_kind': bundle['manifest']['model_kind']},
                'availability_basis': history['availability_basis'] if history else 'simulated_logical_deliveries_not_live',
                'received_at_utc': None, 'live': False,
                'news': snap.get('news', []),
                'limitations': ['Không phải dữ liệu LIVE hay khuyến nghị giao dịch.',
                    'Lịch sử thật: chỉ dùng nội bộ, quyền nguồn/đơn vị còn pending; thời điểm biết nến là giả định bar-end.' if source == 'historical' else
                    'Giá/khối lượng và tin đều giả lập. Dự báo v2 là mốc lợi suất 0, không phải dự báo thị trường đã kiểm chứng.']})

    def review(self, scope, payload):
        with self.lock:
            if scope.get('replay') is not False:
                raise DataContractError('Finish Replay before saving a present-day review')
            bundle, journal, _, snap, _, _ = self._snapshot(**scope)
            allowed = {'review_id', 'event_id', 'event_version', 'reviewer', 'reviewer_kind', 'label', 'reason', 'reference_url'}
            if set(payload) != allowed or payload['reviewer_kind'] not in {'human', 'ai_assisted'}:
                raise DataContractError('Invalid review fields or reviewer kind')
            if any(not isinstance(payload[k], str) or len(payload[k]) > (2000 if k == 'reason' else 256) for k in allowed - {'event_version'}):
                raise DataContractError('Invalid review text')
            now = datetime.now(timezone.utc).isoformat()
            event = next((e for e in snap['events'] if e['event_id'] == payload['event_id'] and e['version'] == payload['event_version']), None)
            if event is None:
                raise DataContractError('Review must refer to a visible event version in this source/model')
            existing = next((r for r in journal.query('reviews', as_of=now) if r['review_id'] == payload['review_id']), None)
            if existing:
                if any(existing[k] != payload[k] for k in allowed):
                    raise JournalConflict('Review ID already has different content')
                return existing
            record = validate_review({'schema_version': 2, **payload, 'reviewed_at': now, 'available_at': now}, event=structural_event(event) if model_is_ai(scope) else event, as_of=now)
            journal.append('reviews', record)
            return record

    def export(self, scope, kind):
        with self.lock:
            _, _, history, snap, _, _ = self._snapshot(**scope)
            if kind == 'bars':
                return historical_csv(history) if history and not model_is_ai(scope) else event_csv(clean(snap['bars']))
            if kind not in {'events', 'reviews'}:
                raise DataContractError('Unknown export kind')
            return event_csv(clean(snap[kind]))
