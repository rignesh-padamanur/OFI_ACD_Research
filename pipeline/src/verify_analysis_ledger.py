"""Independent stage-06c ledger and interval-dependence reconstruction.

Uses only frozen parent artifacts. Graph edges are tested against all unordered
interval pairs, and components are traversed by breadth-first search; no
production ledger function or sweep implementation is imported.
"""
from pathlib import Path
from collections import Counter, deque
import hashlib
import json
import math

import numpy as np
import pandas as pd

SECOND = 1_000_000_000
HORIZONS = (0, 60, 300)
FREQUENCIES = (5, 15, 30)
CAPS = (15, 30, 60)


def _ns(values, *, missing=False):
    index = pd.DatetimeIndex(values)
    if index.tz is None or str(index.tz) != 'UTC' or (index.hasnans and not missing):
        raise AssertionError('Nonmissing UTC timestamps required')
    return index.as_unit('ns').asi8


def _compare(actual, expected):
    """Compare every expected column, including masks, identities and row order."""
    if len(actual) != len(expected):
        raise AssertionError('Frame lengths disagree')
    if set(actual.columns) != set(expected.columns):
        raise AssertionError('Unexpected or omitted frame columns: '+repr(set(actual.columns) ^ set(expected.columns)))
    for name in expected.columns:
        if name not in actual:
            raise AssertionError('Missing column: '+name)
        a, e = actual[name], expected[name]
        if pd.api.types.is_datetime64_any_dtype(e.dtype):
            np.testing.assert_array_equal(_ns(a, missing=True), _ns(e, missing=True), err_msg=name)
        elif pd.api.types.is_bool_dtype(e.dtype):
            np.testing.assert_array_equal(a.isna(), e.isna(), err_msg=name+' missingness')
            np.testing.assert_array_equal(a.fillna(False).to_numpy(dtype=bool), e.fillna(False).to_numpy(dtype=bool), err_msg=name)
        elif pd.api.types.is_float_dtype(e.dtype):
            np.testing.assert_allclose(a.to_numpy(dtype=float), e.to_numpy(dtype=float),
                                      rtol=1e-12, atol=1e-9, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(a.to_numpy(), e.to_numpy(), err_msg=name)


def _json_compare(actual, expected, location='statistics', *, exact=False):
    if isinstance(expected, dict):
        if exact and set(actual) != set(expected):
            raise AssertionError(location+': unexpected or omitted keys')
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(location+': missing '+key)
            _json_compare(actual[key], value, location+'.'+key, exact=exact)
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(location+': list length')
        for i, value in enumerate(expected):
            _json_compare(actual[i], value, location+'.'+str(i), exact=exact)
    elif expected is None:
        if actual is not None:
            raise AssertionError(location+': expected null')
    elif isinstance(expected, (bool, int, str)):
        if actual != expected:
            raise AssertionError(location+': '+repr(actual)+' != '+repr(expected))
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9, err_msg=location)


def _registered_hashes(run, cache=None):
    manifest = json.loads((run/'manifest.json').read_text())
    if cache is None:
        cache = {}

    def digest(path):
        path = Path(path)
        key = str(path.resolve())
        if key not in cache:
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024*1024), b''):
                    h.update(chunk)
            cache[key] = h.hexdigest()
        return cache[key]

    checked = 0
    for name, record in manifest['inputs'].items():
        if digest(record['path']) != record['sha256']:
            raise AssertionError('Changed input: '+name)
        checked += 1
    for name, expected in manifest.get('outputs', {}).items():
        if digest(run/name) != expected:
            raise AssertionError('Changed output: '+name)
        checked += 1
    for name, expected in manifest.get('code_sha256', {}).items():
        if digest(run/'code'/name) != expected:
            raise AssertionError('Changed executed source: '+name)
        checked += 1
    for name, expected in manifest.get('ancestor_hashes', {}).items():
        if digest(name) != expected:
            raise AssertionError('Changed ancestor: '+name)
        checked += 1
    for name, expected in manifest.get('previous_files', {}).items():
        if digest(Path(manifest['snapshot_path'])/name) != expected:
            raise AssertionError('Changed before-repair snapshot: '+name)
        checked += 1
    return manifest, checked




PRE_FIELDS = ('pre_rv_300s_5s_bps','pre_quote_ofi_300s','pre_spread_pips',
    'pre_displayed_size','pre_quote_age_sec','state_history_available','state_sample_eligible',
    'pre_mark_max_age_sec','pre_mark_repeat_fraction','pre_recorded_updates_300s','pre_max_observed_silence_sec')
DETECTION_FIELDS = ('quote_ofi','distance','z_ofi','z_intensity','z_duration',
                    'direction','direction_available','onset_confirmed')
ANCHOR_FIELDS = ('episode_id','bin_start','available_at','support_id', *PRE_FIELDS,
                 *DETECTION_FIELDS,'pre_mark_at','detection_mark_at')
PATH_FIELDS = ('episode_id','bin_start','detected_at','horizon_sec','target_at','endpoint_status',
    'endpoint_quote_at','endpoint_age_sec','pre_age_sec','detection_age_sec','initial_pips',
    'continuation_pips','total_pips','continuation_quote_available','total_quote_available',
    'triplet_quote_available','triplet_age_le_15','triplet_age_le_30','triplet_age_le_60')


def reference_specification():
    """Explicit roles and fixed choices from the recorded stage-06c protocol."""
    roles = {}
    for name in PRE_FIELDS:
        roles[name] = 'strictly_before_onset'
    for name in DETECTION_FIELDS:
        roles[name] = 'detection_time'
    for name in ('episode_id','bin_start','available_at','support_id','onset_day','onset_week_start',
                 'onset_hour_utc','onset_weekday_utc','onset_slot_15m_utc'):
        roles[name] = 'identity_or_calendar'
    for name in ('pre_mark_at','detection_mark_at','nominal_start_at','source_start_at'):
        roles[name] = 'source_accounting_not_predictor'
    for freq in FREQUENCIES:
        for field in ('rv_bps','status','first_quote_at','last_quote_at'):
            roles['pre_'+field+'_'+str(freq)+'s'] = 'strictly_before_onset'
    for cap in CAPS:
        roles['pre_common_age_le_'+str(cap)] = 'strictly_before_onset'
    roles['cancellation_ratio'] = 'detection_time'
    roles['pre_state_eligible'] = 'detection_time_cohort_flag_not_predictor'
    roles['state_sample_eligible'] = 'detection_time_cohort_flag_not_predictor'
    roles['state_history_available'] = 'historical_support_accounting_not_predictor'
    return {'version':'06c.1','policy':'structural_only','window_seconds':300,
        'working_state':'pre_rv_bps_5s','units':'bps; square root of summed squared log returns',
        'frequency_sensitivities_seconds':[15,30],'age_sensitivities_seconds':[15,30,60],
        'direction_ratio_strict_thresholds':[.01,.05,.10],
        'primary_outcome':'continuation_pips at horizon_sec=300',
        'secondary_horizon_seconds':60,'identity_horizon_seconds':0,
        'estimand':'conditional mean recorded continuation within selected directed onsets',
        'pre_state_eligible_role':'cohort at detection, not a strictly pre-onset predictor',
        'response_age_rule':'three marks; future endpoint quality is not ex-ante eligibility',
        'interval':'[earliest actual state/pre/detection quote, detection+horizon)',
        'week_convention':'preceding Sunday 00:00 UTC; reporting label only',
        'inference_approved':False,'latent_volatility_validated':False,'causal_identification':False,
        'anchor_column_roles':roles}


def _unique_lookup(frame, keys, label):
    if frame.duplicated(list(keys)).any() or frame[list(keys)].isna().any().any():
        raise AssertionError(label+': duplicate or missing keys')
    return {tuple(row[key] for key in keys):row for row in frame.to_dict('records')}


def reference_ledger(anchors, measurements, direction, paths):
    """Recreate ledger rows through explicit parent-key dictionaries."""
    parent_anchor = _unique_lookup(anchors,('episode_id',),'anchors')
    dev = measurements.loc[measurements.kind.eq('development')]
    state_lookup = _unique_lookup(dev,('episode_id','frequency_sec'),'states')
    direction_lookup = _unique_lookup(direction,('episode_id',),'direction')
    if paths.duplicated(['episode_id','horizon_sec']).any() or paths[['episode_id','horizon_sec']].isna().any().any():
        raise AssertionError('Paths contain duplicate or missing keys')
    path_lookup = _unique_lookup(paths.loc[paths.horizon_sec.isin(HORIZONS)],('episode_id','horizon_sec'),'selected paths')
    identifiers = sorted(key[0] for key in parent_anchor)
    if set(direction_lookup) != set(parent_anchor):
        raise AssertionError('Direction and anchor identities differ')
    a = anchors.set_index('episode_id',drop=False).loc[identifiers,list(ANCHOR_FIELDS)].reset_index(drop=True).copy()
    extra = []
    for row in a.to_dict('records'):
        identity, onset, detection = row['episode_id'], row['bin_start'], row['available_at']
        if detection-onset != pd.Timedelta(seconds=1):
            raise AssertionError('Detection is not the onset-bin right boundary')
        for field,bound in [('pre_mark_at',onset),('detection_mark_at',detection)]:
            if pd.notna(row[field]) and row[field] >= bound:
                raise AssertionError(field+': quote is not strictly prior')
        derived = {}
        common = None
        for frequency in FREQUENCIES:
            state = state_lookup[(identity,frequency)]
            if state['end_at'] != onset or state['onset_confirmed'] != row['onset_confirmed']:
                raise AssertionError('Pre-state identity/timing differs')
            if state['status'] == 'available':
                if not (state['first_quote_at'] < onset-pd.Timedelta(seconds=300)
                        and state['first_quote_at'] <= state['last_quote_at'] < onset):
                    raise AssertionError('Pre-state quote sources reach the wrong boundary')
            for field in ('rv_bps','status','first_quote_at','last_quote_at'):
                derived['pre_'+field+'_'+str(frequency)+'s'] = state[field]
            flags = [bool(state['common_age_le_'+str(cap)]) for cap in CAPS]
            if common is None:
                common = flags
            elif flags != common:
                raise AssertionError('Common one-second-grid masks differ across frequencies')
        for cap,flag in zip(CAPS,common):
            derived['pre_common_age_le_'+str(cap)] = flag
        direction_row = direction_lookup[(identity,)]
        if direction_row['bin_start'] != onset or direction_row['onset_confirmed'] != row['onset_confirmed']:
            raise AssertionError('Cancellation row timing/identity differs')
        np.testing.assert_allclose(row['direction'],direction_row['inherited_direction'],rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(row['quote_ofi'],direction_row['inherited_net'],rtol=0,atol=0,equal_nan=True)
        derived['cancellation_ratio'] = direction_row['cancellation_ratio']
        ref = derived['pre_rv_bps_5s']
        if row['state_sample_eligible'] and np.isfinite(row['pre_rv_300s_5s_bps']):
            np.testing.assert_allclose(ref,row['pre_rv_300s_5s_bps'],rtol=1e-12,atol=1e-9)
        valid_direction = np.isfinite(row['direction']) and row['direction'] != 0 and row['direction_available']
        derived['pre_state_eligible'] = bool(row['onset_confirmed'] and valid_direction
            and row['state_sample_eligible'] and derived['pre_status_5s'] == 'available'
            and np.isfinite(ref) and ref >= 0 and np.isfinite(row['pre_quote_ofi_300s']))
        day = pd.Timestamp(onset.date(),tz='UTC')
        derived['onset_day'] = day
        derived['onset_week_start'] = day-pd.Timedelta(days=onset.isoweekday()%7)
        derived['onset_hour_utc'] = int(onset.hour)
        derived['onset_weekday_utc'] = int(onset.isoweekday()-1)
        derived['onset_slot_15m_utc'] = int((onset.hour*60+onset.minute)//15)
        derived['nominal_start_at'] = onset-pd.Timedelta(seconds=300)
        sources = [derived['pre_first_quote_at_'+str(frequency)+'s'] for frequency in FREQUENCIES]
        sources += [row['pre_mark_at'],row['detection_mark_at']]
        sources = [stamp for stamp in sources if pd.notna(stamp)]
        derived['source_start_at'] = min(sources) if sources else pd.NaT
        if derived['pre_state_eligible'] and not sources:
            raise AssertionError('Eligible state has no recorded source start')
        extra.append(derived)
    augmented = pd.DataFrame(extra,index=a.index)
    for name in augmented:
        a[name] = augmented[name]
    a_by_id = {row['episode_id']:row for row in a.to_dict('records')}
    rows = []
    expected_keys = {(identity,h) for identity in identifiers for h in HORIZONS}
    saved_keys = {key for key in path_lookup if key[1] in HORIZONS}
    if expected_keys != saved_keys:
        raise AssertionError('Selected-horizon path conservation failed')
    for identity in identifiers:
        state = a_by_id[identity]
        for horizon in HORIZONS:
            parent = path_lookup[(identity,horizon)]
            for name,anchor_name in [('bin_start','bin_start'),('detected_at','available_at'),
                                     ('direction_available','direction_available'),('state_sample_eligible','state_sample_eligible'),
                                     ('onset_confirmed','onset_confirmed')]:
                if parent[name] != state[anchor_name]:
                    raise AssertionError('Outcome parent identity/timing differs: '+name)
            np.testing.assert_allclose(parent['direction'],state['direction'],rtol=0,atol=0,equal_nan=True)
            if parent['target_at'] != state['available_at']+pd.Timedelta(seconds=horizon):
                raise AssertionError('Outcome target does not equal detection plus horizon')
            if pd.notna(parent['endpoint_quote_at']) and parent['endpoint_quote_at'] >= parent['target_at']:
                raise AssertionError('Outcome mark reaches or exceeds target')
            row = {name:parent[name] for name in PATH_FIELDS}
            row['pre_state_eligible'] = state['pre_state_eligible']
            row['initial_quote_available'] = bool(parent_anchor[(identity,)]['baseline_quote_available'])
            for component in ('initial','continuation','total'):
                row['eligible_'+component] = bool(state['pre_state_eligible'] and row[component+'_quote_available']
                                                   and np.isfinite(row[component+'_pips']))
            row['eligible_joint'] = all(row['eligible_'+component] for component in ('initial','continuation','total'))
            primary = row['eligible_continuation']
            for freq in (15,30):
                value = state['pre_rv_bps_'+str(freq)+'s']
                row['eligible_frequency_'+str(freq)] = bool(primary and state['pre_status_'+str(freq)+'s'] == 'available'
                                                          and np.isfinite(value) and value >= 0)
            for cap in CAPS:
                row['eligible_pre_age_'+str(cap)] = bool(primary and state['pre_common_age_le_'+str(cap)])
                row['eligible_response_age_'+str(cap)] = bool(primary and row['triplet_age_le_'+str(cap)])
                row['eligible_both_age_'+str(cap)] = row['eligible_pre_age_'+str(cap)] and row['eligible_response_age_'+str(cap)]
            for percentage in (1,5,10):
                row['eligible_direction_ratio_gt_'+str(percentage)+'pct'] = bool(primary and state['cancellation_ratio'] > percentage/100)
            rows.append(row)
    return {'anchors':a,'outcomes':pd.DataFrame(rows)}


INTERVAL_COLUMNS = ('episode_id','horizon_sec','nominal_start_at','source_start_at',
                    'end_at','onset_day','onset_week_start','degree','component_id','component_size')
PAIR_COLUMNS = ('horizon_sec','episode_id_left','episode_id_right','overlap_seconds',
                'cross_onset_day','cross_onset_week')
COMPONENT_COLUMNS = ('horizon_sec','component_id','events','source_start_at','end_at')
CALENDAR_COLUMNS = ('horizon_sec','block_type','block_start','events','share')


def reference_dependence(anchors,outcomes):
    """Test all unordered interval pairs, then traverse graph components.

    No active-interval sweep or production union-find is used. Exact nanosecond
    intersections distinguish a positive overlap from half-open touching.
    """
    anchor_by_id = {row['episode_id']:row for row in anchors.to_dict('records')}
    all_intervals, all_pairs, all_components, all_calendar = [], [], [], []
    for horizon in HORIZONS:
        selected = outcomes.loc[outcomes.horizon_sec.eq(horizon) & outcomes.eligible_continuation]
        selected = selected.sort_values('episode_id')
        identities = selected.episode_id.astype(int).tolist()
        rows = [anchor_by_id[identity] for identity in identities]
        starts = np.asarray([row['source_start_at'].value for row in rows],dtype='int64')
        ends = _ns(selected.target_at)
        if (starts >= ends).any() or (starts == np.iinfo('int64').min).any():
            raise AssertionError('Invalid source-extended analysis interval')
        adjacency = {identity:set() for identity in identities}
        index_by_id = {identity:position for position,identity in enumerate(identities)}
        for i,identity in enumerate(identities):
            intersections = np.minimum(ends[i],ends[i+1:])-np.maximum(starts[i],starts[i+1:])
            for offset in np.flatnonzero(intersections > 0):
                j = i+1+int(offset)
                other = identities[j]
                adjacency[identity].add(other)
                adjacency[other].add(identity)
                all_pairs.append({'horizon_sec':horizon,'episode_id_left':identity,'episode_id_right':other,
                    'overlap_seconds':float(intersections[offset]/SECOND),
                    'cross_onset_day':rows[i]['onset_day'] != rows[j]['onset_day'],
                    'cross_onset_week':rows[i]['onset_week_start'] != rows[j]['onset_week_start']})
        unvisited = set(identities)
        assigned = {}
        while unvisited:
            seed = min(unvisited)
            frontier = deque([seed])
            component = set()
            while frontier:
                identity = frontier.popleft()
                if identity in component:
                    continue
                component.add(identity)
                frontier.extend(adjacency[identity]-component)
            unvisited -= component
            component_id = min(component)
            for identity in component:
                assigned[identity] = (component_id,len(component))
            locations = [index_by_id[identity] for identity in component]
            all_components.append({'horizon_sec':horizon,'component_id':component_id,'events':len(component),
                'source_start_at':pd.Timestamp(int(min(starts[locations])),unit='ns',tz='UTC'),
                'end_at':pd.Timestamp(int(max(ends[locations])),unit='ns',tz='UTC')})
        for i,identity in enumerate(identities):
            source = anchor_by_id[identity]
            component_id,size = assigned[identity]
            all_intervals.append({'episode_id':identity,'horizon_sec':horizon,
                'nominal_start_at':source['nominal_start_at'],'source_start_at':source['source_start_at'],
                'end_at':pd.Timestamp(int(ends[i]),unit='ns',tz='UTC'),
                'onset_day':source['onset_day'],'onset_week_start':source['onset_week_start'],
                'degree':len(adjacency[identity]),'component_id':component_id,'component_size':size})
        for label,key in [('day','onset_day'),('week','onset_week_start')]:
            frequency = Counter(row[key] for row in rows)
            for stamp,count in sorted(frequency.items()):
                all_calendar.append({'horizon_sec':horizon,'block_type':label,'block_start':stamp,
                                     'events':count,'share':count/len(identities)})
    result = {}
    for name,records,columns,ordering in [
        ('intervals',all_intervals,INTERVAL_COLUMNS,['horizon_sec','episode_id']),
        ('pairs',all_pairs,PAIR_COLUMNS,['horizon_sec','episode_id_left','episode_id_right']),
        ('components',all_components,COMPONENT_COLUMNS,['horizon_sec','component_id']),
        ('calendar',all_calendar,CALENDAR_COLUMNS,['horizon_sec','block_type','block_start'])]:
        result[name] = pd.DataFrame(records,columns=columns).sort_values(ordering).reset_index(drop=True)
    return result


def _quantile(values,probability):
    sequence = sorted(float(value) for value in values)
    if not sequence:
        return None
    position = (len(sequence)-1)*probability
    low,high = int(math.floor(position)),int(math.ceil(position))
    return sequence[low]+(position-low)*(sequence[high]-sequence[low])


def _distribution(values):
    observed = [float(value) for value in values if np.isfinite(value)]
    return {'n':len(observed),'zero':sum(value == 0 for value in observed),
        'mean':float(np.mean(observed)) if observed else None,
        'p10':_quantile(observed,.1),'median':_quantile(observed,.5),'p90':_quantile(observed,.9)}


def reference_statistics(ledger,dependence):
    anchors,outcomes = ledger['anchors'],ledger['outcomes']
    selected = set(anchors.index)
    attrition = []
    conditions = [('all_inherited_onsets',set(anchors.index)),
        ('confirmed',set(anchors.index[anchors.onset_confirmed])),
        ('finite_nonzero_direction',set(anchors.index[np.isfinite(anchors.direction) & anchors.direction.ne(0) & anchors.direction_available])),
        ('inherited_state_sample_eligible',set(anchors.index[anchors.state_sample_eligible])),
        ('available_finite_nonnegative_reference',set(anchors.index[anchors.pre_status_5s.eq('available') & np.isfinite(anchors.pre_rv_bps_5s) & anchors.pre_rv_bps_5s.ge(0)])),
        ('finite_prior_signed_imbalance',set(anchors.index[np.isfinite(anchors.pre_quote_ofi_300s)]))]
    for label,allowed in conditions:
        before = len(selected)
        selected &= allowed
        attrition.append({'step':label,'remaining':len(selected),'newly_excluded':before-len(selected)})
    if selected != set(anchors.index[anchors.pre_state_eligible]):
        raise AssertionError('Sequential attrition and pre-state cohort disagree')
    masks = [column for column in outcomes if column.startswith('eligible_')]
    horizons = []
    for h in HORIZONS:
        group = outcomes.loc[outcomes.horizon_sec.eq(h)]
        joint = group.loc[group.eligible_joint]
        own = group.loc[group.eligible_continuation]
        intervals = dependence['intervals'].loc[dependence['intervals'].horizon_sec.eq(h)]
        pairs = dependence['pairs'].loc[dependence['pairs'].horizon_sec.eq(h)]
        components = dependence['components'].loc[dependence['components'].horizon_sec.eq(h)]
        calendar = dependence['calendar'].loc[dependence['calendar'].horizon_sec.eq(h)]
        graph = {'events':len(intervals),'overlap_pairs':len(pairs),
            'events_with_overlap':int(np.count_nonzero(intervals.degree.to_numpy() > 0)),
            'components':len(components),'largest_component_events':max(components.events.tolist(),default=0),
            'maximum_degree':max(intervals.degree.tolist(),default=0),
            'source_extended_before_nominal':int(np.count_nonzero(intervals.source_start_at.lt(intervals.nominal_start_at))),
            'cross_day_pairs':int(np.count_nonzero(pairs.cross_onset_day)),
            'cross_week_pairs':int(np.count_nonzero(pairs.cross_onset_week))}
        for label in ('day','week'):
            block = calendar.loc[calendar.block_type.eq(label)]
            shares = block.share.to_numpy(dtype=float)
            graph[label] = {'blocks':len(block),'largest_share':float(max(shares)) if len(shares) else None,
                            'inverse_hhi_descriptive':float(1/np.dot(shares,shares)) if len(shares) else None}
        horizons.append({'horizon_sec':h,'inherited_onsets':len(group),
            'mask_counts':{name:int(np.count_nonzero(group[name])) for name in masks},
            'unavailable_continuation_after_pre_state':int(np.count_nonzero(group.pre_state_eligible & ~group.eligible_continuation)),
            'joint_component_means_pips':{component:float(np.mean(joint[component+'_pips'].to_numpy())) if len(joint) else None
                                          for component in ('initial','continuation','total')},
            'continuation_mean_own_sample_pips':float(np.mean(own.continuation_pips.to_numpy())) if len(own) else None,
            'dependence':graph})
    eligible = anchors.loc[sorted(selected)]
    return {'scope':'reused_development_only','primary_policy':'structural_only',
        'anchor_rows':len(anchors),'outcome_rows':len(outcomes),'primary_horizon_seconds':300,
        'pre_state_eligible':len(eligible),'attrition_sequential':attrition,
        'state_distributions_pre_state_cohort':{name:_distribution(eligible[name]) for name in
            ('pre_rv_bps_5s','pre_rv_bps_15s','pre_rv_bps_30s','pre_quote_ofi_300s','pre_spread_pips','pre_displayed_size','cancellation_ratio')},
        'horizons':horizons,'working_specification_frozen':True,'production_inference_approved':False,
        'latent_volatility_validated':False,'comparison_model_fitted':False,'downstream_ready':False}


def verify_analysis_ledger(run):
    """Independently audit every stage-06c table, role and registered hash."""
    run = Path(run)
    cache = {}
    manifest, hash_count = _registered_hashes(run,cache)
    inputs = manifest['inputs']
    parents = {}
    parent_statistics = {}
    for label,expected_status in [('measurement','verified_state_measurement_sensitivity_audit'),
                                  ('response','verified_development_price_responses')]:
        directory = Path(inputs[label+'_manifest']['path']).parent
        parent, count = _registered_hashes(directory,cache)
        hash_count += count
        parents[label] = parent
        _json_compare(parent,{'status':expected_status,'primary_policy':'structural_only'})
        verification = json.loads(Path(inputs[label+'_verification']['path']).read_text())
        if verification.get('status') != 'passed':
            raise AssertionError('Parent verification failed: '+label)
        for suffix in ('statistics','verification'):
            if inputs[label+'_'+suffix]['sha256'] != parent['outputs'][suffix+'.json']:
                raise AssertionError('Parent output hash differs: '+label+'_'+suffix)
        parent_statistics[label] = json.loads(Path(inputs[label+'_statistics']['path']).read_text())
    measurement,response = parents['measurement'],parents['response']
    if measurement['parent_price_response_audit'] != response['run_id']:
        raise AssertionError('State/response parent run relationship differs')
    if inputs['response_manifest']['sha256'] != measurement['inputs']['parent_manifest']['sha256']:
        raise AssertionError('State parent response manifest differs')
    if measurement['inputs']['onsets']['sha256'] != response['inputs']['structural_only_onsets']['sha256']:
        raise AssertionError('State and response onset populations differ')
    for key in ('parameter_cutoff','development_end_exclusive'):
        if pd.Timestamp(parent_statistics['measurement'][key]) != pd.Timestamp(parent_statistics['response'][key]):
            raise AssertionError('State and response time scopes differ: '+key)
    for name in ('measurements','direction'):
        if inputs[name]['sha256'] != measurement['outputs'][name+'.parquet']:
            raise AssertionError('Measurement table lineage differs: '+name)
    for name in ('anchors','paths'):
        if inputs[name]['sha256'] != response['outputs']['structural_only/'+name+'.parquet']:
            raise AssertionError('Response table lineage differs: '+name)
    _json_compare(manifest,{'parent_price_response_audit':response['run_id'],
        'parent_state_measurement_audit':measurement['run_id'],'primary_policy':'structural_only',
        'scope':'reused_development_only','working_specification_frozen':True,
        'production_inference_approved':False,'downstream_ready':False})
    original = {name:pd.read_parquet(inputs[name]['path']) for name in ('anchors','measurements','direction','paths')}
    start = pd.Timestamp(parent_statistics['response']['parameter_cutoff'])
    stop = pd.Timestamp(parent_statistics['response']['development_end_exclusive'])
    if not (original['anchors'].bin_start.ge(start) & original['anchors'].available_at.le(stop)).all():
        raise AssertionError('Inherited anchor leaves the frozen development window')
    ledger = reference_ledger(**original)
    for name,expected in ledger.items():
        _compare(pd.read_parquet(run/(name+'.parquet')),expected)
    dependence = reference_dependence(**ledger)
    for name,expected in dependence.items():
        actual = pd.read_parquet(run/(name+'.parquet'))
        _compare(actual,expected)
        if name == 'pairs':
            np.testing.assert_array_equal(actual.overlap_seconds,expected.overlap_seconds,
                                          err_msg='Exact positive overlap length')
    specification = reference_specification()
    _json_compare(json.loads((run/'specification.json').read_text()),specification,exact=True)
    if set(specification['anchor_column_roles']) != set(ledger['anchors'].columns):
        raise AssertionError('Anchor timing roles do not exactly cover the stored ledger')
    expected_statistics = reference_statistics(ledger,dependence)
    _json_compare(json.loads((run/'statistics.json').read_text()),expected_statistics,exact=True)
    outcomes = ledger['outcomes']
    complete = outcomes.loc[outcomes.eligible_joint]
    np.testing.assert_allclose(complete.total_pips,complete.initial_pips+complete.continuation_pips,
                               rtol=0,atol=1e-9,err_msg='Ledger J+C=I')
    zero = outcomes.loc[outcomes.horizon_sec.eq(0) & outcomes.eligible_continuation]
    np.testing.assert_allclose(zero.continuation_pips,0,rtol=0,atol=1e-9,err_msg='Ledger C(0)=0')
    if len(outcomes) != len(original['anchors'])*len(HORIZONS):
        raise AssertionError('Three-horizon ledger row conservation failed')
    for h in HORIZONS:
        intervals = dependence['intervals'].loc[dependence['intervals'].horizon_sec.eq(h)]
        pairs = dependence['pairs'].loc[dependence['pairs'].horizon_sec.eq(h)]
        components = dependence['components'].loc[dependence['components'].horizon_sec.eq(h)]
        if int(intervals.degree.sum()) != 2*len(pairs) or int(components.events.sum()) != len(intervals):
            raise AssertionError('Graph incidence or component conservation failed')
        for label in ('day','week'):
            calendar = dependence['calendar'].loc[dependence['calendar'].horizon_sec.eq(h) & dependence['calendar'].block_type.eq(label)]
            if int(calendar.events.sum()) != len(intervals):
                raise AssertionError('Calendar cohort conservation failed')
            if len(calendar):
                np.testing.assert_allclose(calendar.share.sum(),1,rtol=0,atol=1e-12)
    evidence = json.loads((run/'test_evidence.json').read_text())
    if evidence.get('status') != 'passed' or evidence.get('errors') != 0 or evidence.get('failures') != 0:
        raise AssertionError('Saved full-suite test evidence did not pass')
    for name,sha in evidence.get('test_source_sha256',{}).items():
        if manifest['outputs'].get('full_test_sources/'+name) != sha:
            raise AssertionError('Executed test source is not the registered snapshot: '+name)
    return {'status':'passed','hashes_verified':hash_count,'unique_files_hashed':len(cache),
        'parent_runs_verified':2,'anchor_rows_verified':len(ledger['anchors']),
        'outcome_rows_verified':len(outcomes),'interval_rows_verified':len(dependence['intervals']),
        'overlap_pairs_verified':len(dependence['pairs']),'component_rows_verified':len(dependence['components']),
        'calendar_rows_verified':len(dependence['calendar']),
        'anchor_timing_roles_verified':len(specification['anchor_column_roles']),
        'all_unordered_pairs_tested':sum(int(n*(n-1)//2) for n in dependence['intervals'].groupby('horizon_sec').size()),
        'methods':'Explicit frozen-parent key reconstruction and independent cohort masks; all unordered half-open interval pairs in nanoseconds; breadth-first graph components; Sunday calendar counts and concentration; every ledger/specification/statistics cell, component identities, direct and parent registered input/output/source/snapshot hashes.',
        'production_inference_approved':False,'comparison_model_fitted':False,'downstream_ready':False}
