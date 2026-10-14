"""Explicit receipt context independent of the existence of an AI decision.

Schema6 is a snapshot envelope only; immutable schema5 journal records and
frozen model artifacts are unchanged. Reading never issues a live certificate.
"""
from pathlib import Path
from stock_monitor.io import DataContractError
from stock_monitor.research.receipt_density import saved_poll_prefix
from stock_monitor.research.receipt_monitor import bound_history
from stock_monitor.research.short_alerts import journal_path, POLICY_SHA
from stock_monitor.session_sampling import verified_delay_basis
from stock_monitor.session_trial import trial_history
from stock_monitor.upgrade.contracts import utc_time


def verified_context(root, value, bundle):
    root=Path(root).resolve(); context=value.get('receipt_context')
    if (value.get('schema_version') != 6 or value.get('source_mode') != 'RECEIVED_MARKET_DATA'
            or not isinstance(context,dict) or set(context) != {'history_binding','delay_evidence','store_path'}):
        raise DataContractError('Received snapshot requires explicit schema6 context')
    evidence=context['history_binding']; history,profile,_=trial_history(root,evidence['path'],evidence['sha256'])
    bound_history(bundle,profile,evidence)
    store=(journal_path(root,bundle['manifest_pin']['sha256'],'RECEIVED_MARKET_DATA').parent/'store').resolve()
    if (not store.is_relative_to(root) or context['store_path']!=store.relative_to(root).as_posix()
            or history['symbol_scope'] != [value['symbol']] or value.get('config_path')!=history['config_path']):
        raise DataContractError('Received context symbol/config/store mismatch')
    delay=context['delay_evidence']
    measured=verified_delay_basis(root,delay['path'],delay['sha256'],config_path=history['config_path'],
        symbol=value['symbol'],now=value['input_cutoff_utc'])
    seconds=measured['empirical_delay_seconds']
    if value['readiness']['provider_delay_seconds'] != seconds:
        raise DataContractError('Received context delay differs from measured evidence')
    if value['readiness']['provider_delay_evidence'] != f"{delay['path']} sha256={delay['sha256']}":
        raise DataContractError('Received readiness delay binding mismatch')
    for record in value.get('events',[])+value.get('outcomes',[]):
        if record['inputs']['history_binding'] != evidence:
            raise DataContractError('Received decision history differs from snapshot context')
    return store,profile,seconds


def bind_received_snapshot(value, bundle, *, store, history_evidence, delay_evidence):
    """Attach only verified persisted receipt context; never create score/latency."""
    root=Path(bundle['root']).resolve(); store=Path(store).resolve()
    if not store.is_relative_to(root): raise DataContractError('Received store outside project')
    ready=value['readiness']
    cutoff=utc_time(value.get('input_cutoff_utc') or ready['observed_at_utc']).isoformat()
    result={**value,'schema_version':6,'source_mode':'RECEIVED_MARKET_DATA',
        'manifest_sha256':bundle['manifest_pin']['sha256'],'policy_sha256':POLICY_SHA,
        'distributable':False,'live':False,'live_certified':False,'input_cutoff_utc':cutoff,
        'calendar_date':ready['calendar_date'],
        'receipt_context':{'history_binding':history_evidence,'delay_evidence':delay_evidence,
                           'store_path':store.relative_to(root).as_posix()}}
    verified_context(root,result,bundle)
    prefix=saved_poll_prefix(store,as_of=cutoff,symbols=[value['symbol']])
    result.update(received_at_utc=prefix['received_at_utc'],prefix_sha256=prefix['prefix_sha256'],poll_pins=prefix['poll_pins'])
    if not result.get('events') and not result.get('outcomes') and not result.get('score_finished_at_utc'):
        result.update(actual_receive_to_score_seconds=None,score_duration_seconds=None)
    return result
