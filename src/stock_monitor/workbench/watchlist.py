"""Read-only development Replay for the separately trained five-symbol GRU6.

The calibration population is deliberately visible, never an independent test.
No fitting, inference, provider calls, or writes occur in this projection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stock_monitor.io import DataContractError
from stock_monitor.research.frozen_density import verify_pin
from stock_monitor.upgrade.ui_service import event_csv

SOURCE = 'watchlist'
MODEL = 'watchlist_gru6'
SYMBOLS = ('VNM', 'ACB', 'TCB', 'MBB', 'SSI')
MODEL_PIN = {'path':'artifacts/research/watchlist_model/gru6_20261013/manifest.json',
             'sha256':'0391befcfbb55ae7f4c119ede23ec74a61ec76345db47b1f356d9e85f54cbd18', 'bytes':4470}
BASIS = 'historical_bar_end_assumption_without_per_bar_receive_receipts'


class WatchlistReplay:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads(verify_pin(self.root, MODEL_PIN).read_text(encoding='utf-8'))
        if (self.manifest['model_id'] != MODEL or self.manifest['symbols'] != list(SYMBOLS)
                or self.manifest['source_rights_status'] != 'pending' or self.manifest['distributable'] is not False):
            raise DataContractError('Invalid watchlist manifest policy')
        self.verify()
        members = {Path(p['path']).name:self.root/p['path'] for p in self.manifest['members']}
        self.thresholds = json.loads(members['thresholds.json'].read_text(encoding='utf-8'))
        scores = pd.read_parquet(members['development_scores.parquet'])
        self.scores = scores.loc[scores.split.eq('validation')].copy()
        self.scores['target_time'] = pd.to_datetime(self.scores.target_time, utc=True)
        if (self.scores.duplicated(['symbol','target_time']).any() or not np.isfinite(self.scores[['score','threshold']].to_numpy()).all()
                or set(self.scores.symbol) != set(SYMBOLS)):
            raise DataContractError('Invalid development scores')
        for s, group in self.scores.groupby('symbol'):
            if not group.threshold.eq(self.thresholds[s]['threshold']).all(): raise DataContractError('Calibration mismatch')
        self.sessions = sorted(self.scores.session_id.unique().tolist())
        bars = json.loads(verify_pin(self.root, self.manifest['dataset']).read_text(encoding='utf-8'))
        self.rows = [r for r in bars if r['session_id'] in self.sessions]
        self.verify()

    def verify(self):
        verify_pin(self.root, MODEL_PIN)
        for p in self.manifest['members'] + self.manifest['input_pins']: verify_pin(self.root, p)

    def catalog(self):
        self.verify()
        return {'available':True,'symbols':list(SYMBOLS),'sessions':self.sessions,
                'models':{'GRU6 · 5 mã':MODEL}, 'default_model':MODEL,
                'default_session':self.sessions[-1], 'evaluation_split':'validation_calibration',
                'peer_manifest_by_symbol':dict.fromkeys(SYMBOLS, MODEL_PIN['sha256'])}

    def _rows(self, symbol, session, model):
        self.verify()
        if symbol not in SYMBOLS or session not in self.sessions or model != MODEL:
            raise DataContractError('Unknown watchlist symbol/session/model')
        rows = sorted((dict(r) for r in self.rows if r['symbol']==symbol and r['session_id']==session),key=lambda r:r['timestamp'])
        if not rows or len(rows)>1000: raise DataContractError('Empty or oversized watchlist session')
        return rows

    def tape(self, symbol, session, model):
        rows=self._rows(symbol,session,model)
        return {'source':SOURCE,'symbol':symbol,'session':session,'model':model,'source_mode':'HISTORICAL_REPLAY',
            'frames':[{'step':i+1,'as_of':b['timestamp'],'bars':rows[:i+1]} for i,b in enumerate(rows)],
            'policy':{'classification':'local_research','source_rights_status':'pending','distributable':False},
            'provenance':{'manifest_sha256':MODEL_PIN['sha256'],'dataset_sha256':self.manifest['dataset']['sha256'],
                          'model_run':MODEL,'split':'validation_calibration'},
            'availability_basis':BASIS,'received_at_utc':None,'live':False,'model_scores_included':False,
            'model_scoring_available':True,'evaluation_split':'validation_calibration',
            'mode':'offline_full_session_tape_not_blind_review','revision_policy':'frozen_historical_snapshot'}

    def snapshot(self, symbol, session, model, replay=False, step=None):
        tape=self.tape(symbol,session,model); frames=tape.pop('frames')
        if step is not None and (isinstance(step,bool) or not isinstance(step,int) or not 1<=step<=len(frames)):
            raise DataContractError('Replay step out of range')
        if replay and step is None: raise DataContractError('Replay requires explicit step')
        frame=frames[(step or len(frames))-1] if replay else frames[-1]
        selected=self.scores.loc[self.scores.symbol.eq(symbol)&self.scores.session_id.eq(session)]
        index={r['target_time'].isoformat():r for r in selected.to_dict('records')}
        events=[]; details={}
        for b in frame['bars']:
            row=index.get(pd.Timestamp(b['timestamp']).isoformat()); value=float(row['score']) if row else None
            threshold=float(self.thresholds[symbol]['threshold']) if row else None
            kind='insufficient_data' if row is None else 'point' if value>threshold else 'normal'
            identity=hashlib.sha256(f"{MODEL_PIN['sha256']}|{symbol}|{b['timestamp']}".encode()).hexdigest()
            event={'event_id':identity,'symbol':symbol,'event_end':b['timestamp'],'decision_bar_end':b['timestamp'],
                'available_at':b['timestamp'],'version':1,'stage':'initial','kind':kind,'score':value,'threshold':threshold,
                'source_mode':'HISTORICAL_REPLAY','evaluation_split':'validation_calibration'}
            events.append(event)
            details[identity]={'method':['GRU đọc 6 nến trước và bối cảnh; điểm cao là mẫu ít gặp.',
                'Ngưỡng riêng từng mã: phân vị 98% trên tập hiệu chỉnh. Đây không phải kiểm định độc lập.'],
                'forecasts':[],'versions':[], 'input_start':row['input_start'] if row else None,
                'input_end':row['input_end'] if row else None}
        return {**tape,**frame,'replay':replay,'timeline':[f['as_of'] for f in frames], 'events':events,'details':details,
                'reviews':[],'episodes':[],'versions':[],'news':[],'model_metadata':None,'score_duration_seconds':None,
                'limitations':['Tập hiệu chỉnh đã dùng chọn mô hình/ngưỡng; không đo hiệu quả ngoài mẫu.',
                               'Phát lại theo giờ đóng nến giả định, không phải nhận dữ liệu trực tiếp.']}

    def export(self, scope, kind):
        if kind not in {'bars','events'}: raise DataContractError('Only visible bars/events may be exported')
        scope=dict(scope); scope.pop('source',None)
        return event_csv(self.snapshot(**scope)[kind])
