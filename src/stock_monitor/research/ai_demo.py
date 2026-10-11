"""Frozen new AI experiment projected onto the existing historical Replay."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from stock_monitor.features import FEATURE_ORDER
from stock_monitor.io import DataContractError, sha256
from stock_monitor.research.ai_contracts import validate_ai_event
from stock_monitor.research.density_samples import AVAILABILITY
from stock_monitor.research.density_selector import scan_residuals, select_episodes
from stock_monitor.research.frozen_density import verify_pin, load_density, infer_density
from stock_monitor.research.protocol import digest
from stock_monitor.research.real_forecast import create_real_forecast, assess_real_forecast
from stock_monitor.upgrade.contracts import utc_time

AI_SHA = 'bf4de45a8a806699f0fc164341c88a9673bb7c0782c64e7c51fff63e303042bd'
AI_MODELS = {'AI · điểm / đoạn': 'ai_context', 'AI · bỏ bối cảnh bổ sung': 'ai_no_context',
             'AI · chỉ điểm (đối chứng)': 'ai_point_only', 'AI · forecast baseline (đối chứng)': 'ai_zero_forecast'}


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_ai(bundle):
    verify_pin(bundle['root'], {'path': 'demo/ai_outline.local.json', 'sha256': AI_SHA})
    for record in bundle['pins']: verify_pin(bundle['root'], record)


def load_ai(root):
    root = Path(root).resolve()
    path = verify_pin(root, {'path': 'demo/ai_outline.local.json', 'sha256': AI_SHA})
    manifest = _read(path)
    if (manifest['source_mode'] != 'HISTORICAL_REPLAY' or manifest['source_rights_status'] != 'pending' or
            manifest['distributable'] is not False):
        raise DataContractError('AI demo must remain historical, local-only, rights pending')
    members = manifest['members']
    pins = members['selector'] + members['forecast'] + members['samples'] + sum(members['density'].values(), [])
    for record in pins: verify_pin(root, record)
    selector_paths = {Path(v['path']).name: root / v['path'] for v in members['selector']}
    forecast_paths = {Path(v['path']).name: root / v['path'] for v in members['forecast']}
    sample_paths = {Path(v['path']).name: root / v['path'] for v in members['samples']}
    with np.load(sample_paths['density_samples.npz'], allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in ('X', 'y', 'context')}
    return {'root': root, 'key': AI_SHA, 'manifest': manifest, 'pins': pins,
        'models': {variant: load_density(root, records, variant=variant) for variant, records in members['density'].items()},
        'design': _read(selector_paths['design.json']),
        'calibrations': {k: _read(selector_paths[k + '_calibration.json']) for k in ('E1', 'E2', 'E3')},
        'arrays': arrays, 'rows': _read(sample_paths['sample_rows.json']),
        'forecast': joblib.load(forecast_paths['ridge.joblib']), 'forecast_policy': _read(forecast_paths['policy.json']),
        'forecast_policy_sha256': sha256(forecast_paths['policy.json'])}


def project_ai(bundle, history_bundle, history, journal, *, model, replay):
    verify_ai(bundle)
    started = time.perf_counter()
    config = bundle['manifest']['configurations'][model]
    variant = 'without_context' if config == 'E2' else 'with_context'
    segments = config != 'E3'
    calibration = bundle['calibrations']['E1' if config == 'E4' else config]
    cutoff = utc_time(history['as_of'])
    policy = bundle['forecast_policy']
    # A frozen train/validation model cannot be backdated to before its calibration.
    after_freeze = cutoff > utc_time(calibration['validation_end']) and cutoff > utc_time(policy['validation_end'])
    indices = [i for i, r in enumerate(bundle['rows']) if after_freeze and r['symbol'] == history['symbol'] and
               r['session_id'] == history['session_id'] and utc_time(r['target_time']) <= cutoff]
    rows = [bundle['rows'][i] for i in indices]
    arrays = {k: v[indices] for k, v in bundle['arrays'].items()}
    inferred = infer_density(bundle['models'][variant], arrays['X'], arrays['y'], arrays['context'], [r['symbol'] for r in rows], stable_prefix=True)
    selector_revision = digest({'model_files': bundle['manifest']['members']['density'][variant],
        'design': next(v for v in bundle['manifest']['members']['selector'] if v['path'].endswith('/design.json')),
        'segments': segments})
    emissions = scan_residuals(rows, inferred['residuals'], bundle['design'], as_of=history['as_of'],
                              revision=selector_revision, segments=segments) if rows else []
    selected = select_episodes(emissions, calibration, bundle['design'])
    emissions_by_time = {e['target_time']: e for e in selected['emissions']}
    index_by_time = {r['target_time']: i for i, r in enumerate(rows)}
    data_name = (history_bundle['dataset_directory'] / 'bars.parquet').relative_to(bundle['root']).as_posix()
    model_hash = next(v['sha256'] for v in bundle['manifest']['members']['density'][variant] if v['path'].endswith('/model.pt'))
    events, versions, forecasts, details, records = [], [], [], {}, []
    bars = history['bars'].copy()
    if history_bundle['dataset_metadata'].get('source') != 'VCI':
        raise DataContractError('AI outcome source differs from registered VCI dataset')
    outcome_bars = bars.assign(source='VCI')
    for bar in bars.to_dict('records'):
        stamp = bar['timestamp'].isoformat()
        e = emissions_by_time.get(stamp)
        ready = e is not None and e['state'] != 'insufficient_reference'
        initial = {'schema_version': 3, 'event_unit': 'bar_5m',
            'event_id': digest({'manifest': AI_SHA, 'model': model, 'symbol': bar['symbol'], 'decision': stamp}),
            'symbol': bar['symbol'], 'session_id': bar['session_id'], 'segment_id': str(bar['segment_id']),
            'version': 1, 'event_start': e['start'] if ready and e['state'] == 'alert' else stamp,
            'event_end': e['end'] if ready and e['state'] == 'alert' else stamp,
            'decision_bar_end': stamp, 'detected_at': stamp, 'available_at': stamp, 'as_of': stamp,
            'stage': 'initial', 'kind': e['classification'] if ready and e['state'] == 'alert' else 'normal' if ready else 'insufficient_data',
            'model_run': 'ai_outline:' + model, 'model_sha256': model_hash, 'data_sha256': history_bundle['pins'][data_name],
            'data_revision': data_name, 'source_mode': 'HISTORICAL_REPLAY', 'availability_basis': AVAILABILITY,
            'episode_id': e['episode_id'] if ready else None,
            'reason_codes': [e['state'] if ready else 'before_frozen_validation' if not after_freeze else 'density_history_unavailable'],
            'score': e['joint_gain'] if ready else None, 'threshold': e['threshold'] if ready else None,
            'decision_method': 'conditional_penalized_likelihood'}
        validate_ai_event(initial, as_of=history['as_of'])
        versions.append(initial); records.append(('events', initial)); latest = initial
        journal.append('events', initial)
        local_forecasts = []
        if ready:
            i = index_by_time[stamp]
            features = arrays['X'][i, -5:, 0].tolist() + [float(arrays['y'][i, 0])]
            algorithm = policy['algorithm'] if config != 'E4' else 'zero_return'
            prediction = (bundle['forecast']['model'].predict(bundle['forecast']['scaler'].transform([features]))[0]
                          if algorithm == 'ridge_h1_h3' else [0., 0.])
            # Period/source/quality comes from the checked frozen historical bar, not invented receipt fields.
            origin = {k: bar[k] for k in ('symbol', 'session_id', 'segment_id', 'quality_status', 'close')}
            origin.update(timestamp=stamp, source='VCI')
            for j, horizon in enumerate((1, 3)):
                forecast = create_real_forecast(origin, features, horizon=horizon, prediction=prediction[j], algorithm=algorithm,
                    model_sha256=bundle['forecast_policy_sha256'], calendar=policy['calendar'])
                forecasts.append(forecast); records.append(('forecasts', forecast))
                # Commit the immutable prefix-only prediction before reading its
                # later outcome, even when the caller requests a whole session.
                journal.append('forecasts', forecast)
                assessment = assess_real_forecast(forecast, outcome_bars, as_of=history['as_of'])
                evidence = assessment.get('post_evidence')
                reference = policy['references'].get(bar['symbol'] + ':h' + str(horizon), [])
                rank = None
                if evidence and len(reference) >= policy['minimum_reference_count']:
                    residual = abs(evidence['residual'])
                    rank = (np.searchsorted(reference, residual, 'left') + np.searchsorted(reference, residual, 'right')) / (2 * len(reference))
                local_forecasts.append({**forecast, 'status': assessment['status'],
                    'realized_return': evidence['realized_return'] if evidence else None,
                    'residual': evidence['residual'] if evidence else None, 'empirical_residual_rank': rank,
                    'reference_count': len(reference), 'baseline': algorithm == 'zero_return'})
                if evidence:
                    mature_time = evidence['outcome_available_at']
                    latest = {**initial, 'version': latest['version'] + 1, 'stage': 'post_event',
                        'as_of': mature_time, 'detected_at': mature_time, 'available_at': mature_time,
                        'post_evidence': evidence, 'reason_codes': initial['reason_codes'] + ['mature_h' + str(horizon)]}
                    validate_ai_event(latest, as_of=history['as_of'])
                    records.append(('events', latest)); versions.append(latest)
        events.append(latest)
        if ready:
            i = index_by_time[stamp]
            names = ('Lợi suất', 'Biên độ nến', 'Thân nến', 'Volume so với cùng giờ')
            contributions = (3. * np.log1p(inferred['residuals'][i] ** 2 / 5.)).tolist()
            top = int(np.argmax(contributions))
            method = [f"AI so sánh nền bình thường, một điểm và một đoạn trong {e['region_count']} nến đã biết.",
                f"Chi phí H0 {e['h0_cost']:.3f}; H1 {e['h1_cost']:.3f}; H2 " + (f"{e['h2_cost']:.3f}." if e['h2_cost'] is not None else 'chưa có đoạn hợp lệ.'),
                f"Tiết kiệm chi phí {e['joint_gain']:.3f} / ngưỡng {e['threshold']:.3f}; {e['reference_count']} mẫu validation.",
                f"Sai lệch nền lớn nhất ở nến quyết định: {names[top]}; đây là phân rã loss, không phải nguyên nhân kinh tế."]
            if e['state'] == 'budget_suppressed': method.append('Không mở hồ sơ mới vì đã đủ 3 hồ sơ trong phiên; emission vẫn được giữ.')
            if e['classification'] == 'collective':
                guard = next(v for v in e['candidates'] if v['length'] == e['segment_length'])
                method.append(f"Bỏ đóng góp lớn nhất vẫn giữ {100 * guard['retained_gain_fraction']:.1f}% mức cải thiện; không xóa nến gốc.")
            context = arrays['context'][i].tolist()
            method.append(f"Bối cảnh 12 nến trước: lợi suất trung bình {100 * context[1]:.3f}%; biến động {100 * context[0]:.3f}%.")
            evidence_detail = {**e, 'density_nll': float(inferred['scores'][i]), 'residuals': inferred['residuals'][i].tolist(),
                               'feature_loss_without_common_scale_constant': dict(zip(FEATURE_ORDER, contributions)),
                               'conditioning_end': rows[i]['input_end'], 'lagged_context': context}
        else:
            method = ['Chưa chấm AI trước khi kết thúc train/validation đã khóa.' if not after_freeze else
                      'Chưa đủ 12 nến liên tục có đặc trưng hợp lệ; không coi thiếu score là bình thường.']
            evidence_detail = {}
        details[initial['event_id']] = {'method': method, 'evidence': evidence_detail, 'forecasts': local_forecasts,
                                        'initial_event': initial, 'versions': [v for v in versions if v['event_id'] == initial['event_id']]}
    journal.append_many(records)
    review_as_of = history['as_of'] if replay else datetime.now(timezone.utc).isoformat()
    keys = {(e['event_id'], e['version']) for e in versions}
    reviews = [r for r in journal.query('reviews', as_of=review_as_of, symbol=history['symbol']) if (r['event_id'], r['event_version']) in keys]
    by_time = {e['decision_bar_end']: e for e in events}
    bars['score'] = [by_time[t.isoformat()]['score'] for t in bars.timestamp]
    bars['threshold'] = [by_time[t.isoformat()]['threshold'] for t in bars.timestamp]
    bars['score_eligible'] = bars.score.notna()
    bars['score_status'] = bars.score.notna().map({True: 'frozen_AI_conditional_cost', False: 'AI_unavailable'})
    bars['model'] = model
    bars['ai_manifest_sha256'] = AI_SHA
    bars['source_mode'] = 'HISTORICAL_REPLAY'
    bars['availability_basis'] = AVAILABILITY
    bars['source_rights_status'] = 'pending'
    bars['distributable'] = False
    bars['received_at_utc'] = None
    return {'bars': bars, 'events': events, 'versions': versions, 'reviews': reviews, 'details': details,
        'episodes': selected['episodes'], 'forecasts': forecasts, 'news': [],
        'score_duration_seconds': time.perf_counter() - started, 'received_at_utc': None,
        'model_metadata': {'manifest_sha256': AI_SHA, 'configuration': config, 'algorithm': 'GRU + Student-t, conditional cost selector',
            'training_role': 'old explored train/validation', 'effectiveness_status': bundle['manifest']['effectiveness_status'],
            'forecast_algorithm': policy['algorithm'] if config != 'E4' else 'zero_return',
            'independent_test': False, 'human_labels': 0}}
