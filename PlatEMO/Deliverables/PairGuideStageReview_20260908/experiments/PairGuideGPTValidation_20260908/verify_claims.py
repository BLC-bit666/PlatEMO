"""Recompute the shared analysis using cached, already evaluated evidence."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'Deliverables/PairGuideLateQualityReview_20260907/numeric_evidence'


def resolve(value, folder=DATA):
    if isinstance(value, dict):
        if set(value) == {'$ref'}:
            path = folder / value['$ref']
            return resolve(json.loads(path.read_text()), path.parent)
        return {key: resolve(item, folder) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, folder) for item in value]
    return value


def read(name):
    return resolve(json.loads((DATA / name).read_text()))


def directions(y, W, scale):
    normalized = (y-np.array(scale['minimum']))/np.array(scale['span'])
    unit = normalized/np.maximum(np.linalg.norm(normalized, axis=1)[:, None], np.finfo(float).eps)
    wn = W/np.linalg.norm(W, axis=1)[:, None]
    refs = np.argmax(unit @ wn.T, axis=1)
    zero = np.linalg.norm(normalized, axis=1) <= np.finfo(float).eps
    if zero.any():
        refs[zero] = np.argmin(((normalized[zero, None, :]-W[None, :, :])**2).sum(axis=2), axis=1)
    angles = np.rad2deg(np.arctan2(normalized[:, 1], normalized[:, 0]))
    return refs, angles


def forward(model, conditions):
    params = {(row['layer'], row['parameter']): np.asarray(row['value'], dtype=np.float32).reshape(row['shape'])
              for row in model['generator']}
    x = np.vstack((np.zeros((model['zDim'], len(conditions)), dtype=np.float32),
                   np.asarray(conditions, dtype=np.float32).T))
    for layer in ['pair_guide_g_fc1', 'pair_guide_g_fc2', 'pair_guide_g_out']:
        x = params[(layer, 'Weights')] @ x + params[(layer, 'Bias')]
        if layer != 'pair_guide_g_out': x = np.maximum(x, np.float32(.2)*x)
    return ((np.tanh(x).T.astype(float)+1)/2)


def main():
    results = []
    for number in range(5, 9):
        name = f'LIRCMOP{number}_BC'
        query = read(name+'_run01_stage01_query.json')
        W = np.empty((100, 2))
        for ref, condition in zip(query['refs'], query['conditions']): W[int(ref)-1] = condition[:2]
        S = read(name+'_run01_stage06_plot_state.json')
        y = np.asarray(S['searchState']['p1Objs'])
        population = read(name+'_seed01_cgan_population_counters.json')
        first = read(name+'_seed01_cgan_first_model.json')
        late = [p for p in population if p['observationFE'] >= 50000]
        assert len(late) == 251 and len(y) == 100
        original = [directions(y, W, p['referenceScale']) for p in late]
        stable = [directions(y, W, dict(minimum=p['referenceScale']['minimum'],
                  span=first['lastData']['referenceScale']['span'])) for p in late]
        changed = np.mean([a[0] != b[0] for a, b in zip(original[:-1], original[1:])])
        fixed_changed = np.mean([a[0] != b[0] for a, b in zip(stable[:-1], stable[1:])])
        angle = np.mean([np.abs(a[1]-b[1]) for a, b in zip(original[:-1], original[1:])])
        train = read(name+'_seed01_cgan_training_events.json')
        late_train = [e for e in train if e['trained'] and e['generation'] >= 249]
        last = read(name+'_seed01_cgan_last_model.json')
        c0 = np.column_stack((W, np.zeros(100))); c1 = np.column_stack((W, np.ones(100)))
        side_gap = np.median(np.linalg.norm(forward(last,c1)-forward(last,c0),axis=1))
        below_first = np.mean(np.any(y < np.array(first['lastData']['referenceScale']['minimum']),axis=1))
        generations = read(name+'_seed01_cgan_generation_counters.json')
        late_gen = [g for g in generations if g['startFE'] >= 50000]
        result = dict(problem=name, seed=1, fixed_points=100, frame_transitions=250,
                      dynamic_changed_fraction=float(changed), stable_span_changed_fraction=float(fixed_changed),
                      dynamic_angle_step=float(angle), side_gap_median=float(side_gap),
                      below_initial_minimum=float(below_first), late_training_events=len(late_train),
                      late_selected=sum(g['use']['selected'] for g in late_gen),
                      late_guided_infeasible_replacements=sum(g['archive']['guidedTightenedInfeasible'] for g in late_gen))
        results.append(result)
        print(result)
    with (OUT/'verified_scale_claims.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    expected=[.76612,.91332,.86408,.89044]
    assert np.allclose([r['dynamic_changed_fraction'] for r in results],expected,atol=1e-9)
    assert np.allclose([r['stable_span_changed_fraction'] for r in results],[.00508,.00028,.01768,.00020],atol=1e-9)
    (OUT/'claim_verification.json').write_text(json.dumps(dict(passed=True,new_training=0,new_objective_calls=0,
        new_constraint_calls=0,scope='Frozen end-point relabelling, not performance attribution'),indent=2))


if __name__=='__main__': main()
