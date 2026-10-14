"""Pinned historical quotes for new symbols, explicitly without model scores."""
import json
from pathlib import Path
import pandas as pd

from stock_monitor.config import load_config
from stock_monitor.io import DataContractError
from stock_monitor.providers import ENDPOINT, parse_vci_minutes
from stock_monitor.research.frozen_density import verify_pin
from stock_monitor.validation import load_profile, normalize_minutes
from stock_monitor.workbench.service import clean

SYMBOLS = ('VNM', 'ACB', 'TCB', 'MBB', 'SSI')


class ExtraPrices:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.pointer_path = self.root / 'demo/watchlist_extension.local.json'
        pointer = json.loads(self.pointer_path.read_text(encoding='utf-8'))
        self.manifest_pin = pointer['manifest']
        path = verify_pin(self.root, self.manifest_pin)
        if not path.is_relative_to(self.root / 'artifacts/research/watchlist_extension'):
            raise DataContractError('Price extension outside registered namespace')
        self.manifest = json.loads(path.read_text(encoding='utf-8')); m = self.manifest
        if (m.get('kind') != 'historical_watchlist_extension' or m.get('symbols') != list(SYMBOLS) or
                m.get('source') != 'VCI' or m.get('source_mode') != 'HISTORICAL_REPLAY' or
                m.get('source_rights_status') != 'pending' or m.get('distributable') is not False or
                m.get('model_scoring_available') is not False or m.get('model_fits') != 0 or m['profile']['path'] != 'configs/research.yaml'):
            raise DataContractError('Price extension source/model policy mismatch')
        self.profile = load_profile(load_config(verify_pin(self.root, m['profile'])))
        self.pins = [self.manifest_pin, m['profile'], m['registration']]; self.rows = {}
        registration = json.loads(verify_pin(self.root, m['registration']).read_text(encoding='utf-8'))
        if registration['symbols'] != list(SYMBOLS) or registration['day'] != m['session'] or registration['profile'] != m['profile']:
            raise DataContractError('Price extension differs from its prior registration')
        for symbol in SYMBOLS:
            entry = m['entries'][symbol]
            if entry['status'] == 'unavailable': continue
            if entry['status'] != 'available' or entry.get('scoring_available') is not False:
                raise DataContractError('Price extension cannot promote scoring availability')
            raw = verify_pin(self.root, entry['raw']); metadata = verify_pin(self.root, entry['metadata'])
            storage = self.root / 'data/raw/research/watchlist_extension_20261013'
            if not raw.is_relative_to(storage) or not metadata.is_relative_to(storage):
                raise DataContractError('New-symbol raw paths escape their namespace')
            meta = json.loads(metadata.read_text(encoding='utf-8'))
            http = meta['http']
            if (meta.get('symbol') != symbol or meta.get('is_synthetic') is not False or
                    (meta['requested_start'], meta['requested_end']) != (m['session'], m['session']) or
                    len(http) != 1 or http[0]['url'] != ENDPOINT or http[0]['status'] != 200 or
                    http[0]['sha256'] != entry['raw']['sha256'] or
                    http[0]['public_candle_query']['symbols'] != [symbol] or
                    http[0]['public_candle_query']['timeFrame'] != 'ONE_MINUTE' or
                    http[0]['public_candle_query']['countBack'] != 500 or
                    http[0]['public_candle_query']['to'] != int((pd.Timestamp(m['session'], tz='Asia/Ho_Chi_Minh') + pd.Timedelta(days=1)).timestamp())):
                raise DataContractError('New-symbol source receipt does not match exact successful query')
            frame = parse_vci_minutes(json.loads(raw.read_bytes()), symbol)
            normalized = normalize_minutes(frame, self.profile, start=m['session'], end=m['session'], symbols=[symbol])
            rows = json.loads(verify_pin(self.root, entry['bars']).read_text(encoding='utf-8'))
            if rows != clean(normalized.bars.to_dict('records')) or not 1 <= len(rows) <= 100:
                raise DataContractError('Stored new-symbol bars differ from original validated minutes')
            self.rows[symbol] = rows
            self.pins.extend(entry[k] for k in ('raw', 'metadata', 'bars', 'validation'))
        self.verify()

    def verify(self):
        for record in self.pins: verify_pin(self.root, record)
        current = json.loads(self.pointer_path.read_text(encoding='utf-8'))
        if current.get('manifest') != self.manifest_pin:
            raise DataContractError('Price extension pointer changed; restart before adopting a new bundle')

    def tape(self, symbol, session, model):
        self.verify()
        if model not in {'robust_zscore', 'isolation_forest', 'lstm', 'ai_context', 'ai_no_context', 'ai_point_only', 'ai_zero_forecast'}:
            raise DataContractError('Unregistered price-only method ID')
        if symbol not in self.rows or session != self.manifest['session']:
            raise DataContractError('Mã này chưa có dữ liệu giá cho phiên đang chọn')
        rows = self.rows[symbol]
        return {'source': 'historical', 'symbol': symbol, 'session': session, 'model': model,
            'source_mode': 'HISTORICAL_REPLAY', 'live': False, 'model_scoring_available': False,
            'model_scores_included': False, 'bar_export_supported': False,
            'frames': [{'step': i + 1, 'as_of': b['timestamp'], 'bars': rows[:i + 1]} for i, b in enumerate(rows)],
            'policy': {'classification': 'historical_research_price_only', 'source_rights_status': 'pending', 'distributable': False},
            'provenance': {'manifest_sha256': self.manifest_pin['sha256'], 'model_kind': 'unavailable_for_new_symbol',
                'raw_sha256': self.manifest['entries'][symbol]['raw']['sha256']},
            'availability_basis': self.manifest['availability_basis'], 'received_at_utc': None,
            'mode': 'offline_full_session_tape_not_blind_review'}
