"""Contract3 historical AI decisions; isolated journal, inherited append-only storage."""
from copy import deepcopy

from stock_monitor.io import DataContractError
from stock_monitor.research.density_samples import AVAILABILITY
from stock_monitor.research.real_forecast import validate_real_forecast
from stock_monitor.upgrade.contracts import validate_event, validate_review, utc_time, _post_evidence
from stock_monitor.upgrade.journal import Journal, JournalConflict


def structural_event(record):
    """Reuse contract2 time/numeric shape checks, not its algorithm classification."""
    proxy = deepcopy(record)
    for key in ('decision_bar_end', 'availability_basis', 'episode_id'): proxy.pop(key, None)
    proxy.update(schema_version=2, decision_method='baseline', stage='initial')
    proxy.pop('post_evidence', None)
    return proxy


def validate_ai_event(record, *, as_of=None):
    if (record.get('schema_version') != 3 or record.get('source_mode') != 'HISTORICAL_REPLAY' or
            record.get('decision_method') != 'conditional_penalized_likelihood' or
            record.get('availability_basis') != AVAILABILITY or record.get('stage') not in {'initial', 'post_event'}):
        raise DataContractError('Expected historical conditional-cost contract3')
    decision = utc_time(record['decision_bar_end'])
    if decision < utc_time(record['event_end']) or decision > utc_time(record['as_of']):
        raise DataContractError('AI decision endpoint precedes observed interval or follows query')
    if record['stage'] == 'initial' and ('post_evidence' in record or decision != utc_time(record['as_of'])):
        raise DataContractError('Initial must stop at decision bar without outcome')
    validate_event(structural_event(record), as_of=as_of)
    if record['stage'] == 'post_event':
        evidence = record.get('post_evidence')
        if not isinstance(evidence, dict) or evidence.get('origin_bar_end') != record['decision_bar_end']:
            raise DataContractError('Post-event forecast must originate at initial decision, not backfill old point')
        if utc_time(evidence['forecast_created_at']) != decision:
            raise DataContractError('Forecast created after initial decision')
        stripped = {k: v for k, v in evidence.items() if k != 'origin_bar_end'}
        times = {k: utc_time(record[k]) for k in ('event_start', 'event_end', 'as_of', 'detected_at', 'available_at')}
        times['event_end'] = decision
        _post_evidence(stripped, times, record['session_id'])
    return deepcopy(record)


class AIJournal(Journal):
    def _validate(self, conn, collection, value):
        if collection == 'events':
            result = validate_ai_event(value)
            if result['stage'] == 'post_event':
                from stock_monitor.research.protocol import digest
                evidence = result['post_evidence']
                candidates = [self._decode(row) for row in conn.execute("SELECT * FROM records WHERE collection='forecasts' AND symbol=? AND available=?", (result['symbol'], utc_time(result['decision_bar_end']).isoformat()))]
                forecast = next((f for f in candidates if digest(f) == evidence['forecast_sha256']), None)
                if forecast is None or any(forecast[k] != evidence[k] for k in ('horizon_bars', 'prediction', 'target_bar_end', 'origin_bar_end')):
                    raise DataContractError('AI post-event needs a matching persisted frozen forecast')
            identity, version = result['event_id'], result['version']
            original = self._find(conn, collection, identity)
            if original is None and (version != 1 or result['stage'] != 'initial'):
                raise JournalConflict('First AI event must be initial version1')
            if original is not None:
                fields = ('symbol', 'session_id', 'source_mode', 'model_run', 'model_sha256', 'data_sha256',
                          'event_start', 'event_end', 'decision_bar_end', 'score', 'threshold', 'kind',
                          'data_revision', 'availability_basis', 'episode_id', 'decision_method', 'segment_id')
                if any(result[k] != original[k] for k in fields):
                    raise JournalConflict('AI post-event cannot rewrite initial decision')
                last = conn.execute("SELECT version,available FROM records WHERE collection='events' AND record_id=? ORDER BY version DESC LIMIT 1", (identity,)).fetchone()
                if self._find(conn, collection, identity, version) is None and (
                        version != last['version'] + 1 or utc_time(result['available_at']) < utc_time(last['available'])):
                    raise JournalConflict('AI versions must be sequential, availability monotone')
            return result, identity, version, identity, version, result['symbol'], result['available_at']
        if collection == 'forecasts':
            result = validate_real_forecast(value)
            return result, result['forecast_id'], 1, None, None, result['symbol'], result['available_at']
        if collection == 'reviews':
            event = self._find(conn, 'events', value.get('event_id'), value.get('event_version'))
            if event is None: raise DataContractError('AI review requires persisted event/version')
            validate_ai_event(event)
            result = validate_review(value, event=structural_event(event))
            return result, result['review_id'], 1, event['event_id'], event['version'], event['symbol'], result['available_at']
        raise DataContractError('AI journal accepts events, forecasts and reviews only')
