"""Audit the completed matched four-case experiment and export R figure data.

Only saved predictions, checkpoints and contracts are read. No training occurs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'article/evidence'
VERSION = 'dlan_matched4_10seeds_v1'
CASES = {'CO2': ['dlan', 'geoaggregator'], 'OD-Wuhan': ['dlan', 'geoaggregator'],
         'WuhanHP': ['dlan', 'geoaggregator', 'gcnnwr'],
         'GVI': ['dlan', 'geoaggregator', 'gcnnwr']}
LABELS = {'dlan': 'DLAN', 'geoaggregator': 'GeoAggregator', 'gcnnwr': 'GCNNWR'}
sources = {}
def record(path):
    path = Path(path)
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def metrics(actual, prediction):
    error = prediction.astype(np.float64) - actual.astype(np.float64)
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))),
                r2=float(1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2)))

status = json.loads(record(ROOT / 'runs' / VERSION / 'status.json').read_text())
assert status['status'] == 'complete' and status['completed_fits'] == 100
for name, digest in status['source_hashes'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
records, predictions, bandwidths, group_errors, ols = [], [], [], [], []
for case, models in CASES.items():
    base = ROOT / 'share' / case
    contract = json.loads(record(base / 'data' / VERSION / 'contract.json').read_text())
    datafile = record(base / 'data' / VERSION / 'ordinary.npz')
    assert sources[str(datafile.relative_to(ROOT))] == contract['data_sha256']
    data = np.load(datafile)
    map_coordinates = data['u_raw']
    if case == 'WuhanHP':
        # The frozen matched input inherits normalised coordinates; use the
        # row-aligned source geometry solely for maps in projected metres.
        geometry_source = np.load(record(base / 'data' / 'processed.npz'))
        np.testing.assert_allclose(geometry_source['y_raw'].ravel(), data['actual'], rtol=1e-6)
        map_coordinates = geometry_source['u_raw']
    partitions = [data[f'{part}_ids'] for part in ('train', 'validation', 'test')]
    assert len(np.unique(np.concatenate(partitions))) == contract['rows']
    train_groups = set(data['groups'][partitions[0]])
    if case in ('WuhanHP', 'GVI'):
        assert train_groups.isdisjoint(data['groups'][partitions[1]])
        assert train_groups.isdisjoint(data['groups'][partitions[2]])
        assert set(data['groups'][partitions[1]]).isdisjoint(data['groups'][partitions[2]])
    for kind in models:
        folder = base / 'outputs' / VERSION / kind
        config = json.loads(record(folder / 'configuration.json').read_text())
        assert config['parameters'] == 6957
        for seed in range(1, 11):
            frame = pd.read_csv(record(folder / f'seed_{seed}_metrics.csv'))
            assert len(frame) == 2 and set(frame.split) == {'validation', 'test'}
            assert (frame.parameters == 6957).all() and (frame.seed == seed).all()
            assert set(frame.config_digest) == {config['config_digest']}
            saved = np.load(record(folder / f'seed_{seed}_predictions.npz'))
            audit = pd.read_csv(record(folder / f'seed_{seed}_audit.csv'))
            # Audit files record checkpoint-reloaded prediction consistency.
            assert len(audit) == 2 and audit['prediction_reload_pass'].all()
            for part in ('validation', 'test'):
                ids = saved[f'{part}_row_id'].astype(int)
                np.testing.assert_array_equal(ids, data['row_id'][data[f'{part}_ids']])
                actual = saved[f'{part}_actual'].astype(np.float64)
                pred = saved[f'{part}_predicted'].astype(np.float64)
                np.testing.assert_allclose(actual, data['actual'][data[f'{part}_ids']], rtol=1e-6, atol=1e-6)
                recalculated = metrics(actual, pred)
                original = frame[frame.split == part].iloc[0]
                for key, value in recalculated.items():
                    np.testing.assert_allclose(value, original[key], rtol=2e-5, atol=1e-6)
                row = original.to_dict()
                row.update(recalculated, model_label=LABELS[kind], audited_rows=len(ids))
                if case == 'WuhanHP':
                    groups = data['groups'][data[f'{part}_ids']]
                    group_mse = pd.DataFrame({'group': groups, 'squared': (pred - actual)**2}).groupby('group').squared.mean()
                    row['equal_group_rmse'] = float(np.sqrt(group_mse.mean()))
                    for group, subset in pd.DataFrame(dict(group=groups, actual=actual, predicted=pred, row_id=ids)).groupby('group'):
                        first = int(subset.row_id.iloc[0]); ge = subset.predicted - subset.actual
                        group_errors.append(dict(case=case, model=LABELS[kind], seed=seed, split=part,
                            group=group, n=len(subset), rmse=np.sqrt(np.mean(ge**2)),
                            mean_residual=ge.mean(), x=map_coordinates[first,0], y=map_coordinates[first,1]))
                if case == 'OD-Wuhan':
                    threshold = np.quantile(actual, .99)
                    tail = actual >= threshold
                    row['upper_1pct_sse_share'] = float(np.sum((pred[tail]-actual[tail])**2) / np.sum((pred-actual)**2))
                    row['body_rmse'] = float(np.sqrt(np.mean((pred[~tail]-actual[~tail])**2)))
                records.append(row)
                if part == 'test':
                    indices = data[f'{part}_ids']
                    predictions.append(pd.DataFrame(dict(case=case, model=LABELS[kind], seed=seed,
                        row_id=ids, actual=actual, predicted=pred, residual=pred-actual,
                        group=data['groups'][indices], x=map_coordinates[indices,0], y=map_coordinates[indices,1])))
            checkpoint = torch.load(record(folder / f'seed_{seed}_checkpoint.pt'), map_location='cpu', weights_only=False)
            assert checkpoint['configuration']['config_digest'] == config['config_digest']
            st = checkpoint['state']
            assert set(st['context_ids'].tolist()) == set(data['train_ids'].tolist())
            if kind == 'dlan':
                for key, value in st['model_state'].items():
                    if key.endswith('raw_bandwidth'):
                        values = torch.nn.functional.softplus(value).numpy() + 1e-6
                        names = {'CO2': ['Space x','Space y','Time'], 'OD-Wuhan': ['Origin x','Origin y','Destination x','Destination y'],
                                 'WuhanHP': ['Space x','Space y'], 'GVI': ['Space x','Space y']}[case]
                        for axis, name in enumerate(names):
                            bandwidths.append(dict(case=case, seed=seed, axis=name, bandwidth=float(values[axis,0])))
            if case == 'GVI' and kind == 'gcnnwr':
                # Read the already fitted training-only OLS buffers; do not fit a new model.
                test = data['test_ids']
                x = torch.as_tensor(data['x'][test], dtype=torch.float32)
                x = (x - st['x_mean']) / st['x_std']
                design = torch.cat((torch.ones_like(x[:,:1]),x),1)
                pred = (design @ st['model_state']['network.ols_coefficients'] * st['y_std'] + st['y_mean']).numpy()
                ols.append(dict(case=case, seed=seed, **metrics(data['actual'][test].astype(float),pred.astype(float))))
                if seed == 1:
                    predictions.append(pd.DataFrame(dict(case=case, model='OLS buffer', seed=seed,
                        row_id=data['row_id'][test], actual=data['actual'][test], predicted=pred,
                        residual=pred-data['actual'][test],group=data['groups'][test],
                        x=data['u_raw'][test,0],y=data['u_raw'][test,1])))

all_metrics = pd.DataFrame(records)
assert len(all_metrics) == 200
summary = all_metrics.groupby(['case','model_label','split']).agg(
    runs=('seed','nunique'), parameters=('parameters','first'), rmse_mean=('rmse','mean'),
    rmse_sd=('rmse','std'), mae_mean=('mae','mean'), r2_mean=('r2','mean'),
    seconds_mean=('seconds','mean'), seconds_sd=('seconds','std'), epochs_mean=('epochs','mean'),
    equal_group_rmse=('equal_group_rmse','mean'), upper_1pct_sse_share=('upper_1pct_sse_share','mean'), body_rmse=('body_rmse','mean')).reset_index()
paired=[]
for case, subset in all_metrics[all_metrics.split == 'test'].groupby('case'):
    p = subset.pivot(index='seed',columns='model_label',values='rmse')
    for other in p.columns.drop('DLAN'):
        paired.append(dict(case=case, comparator=other, dlan_wins=int((p.DLAN < p[other]).sum()),
            mean_reduction_pct=float((1-p.DLAN.mean()/p[other].mean())*100),
            mean_paired_difference=float((p.DLAN-p[other]).mean())))
for name, data in [('matched_metrics.csv',all_metrics),('matched_summary.csv',summary),
                   ('matched_predictions.csv',pd.concat(predictions,ignore_index=True)),
                   ('matched_bandwidths.csv',pd.DataFrame(bandwidths)),
                   ('matched_hp_groups.csv',pd.DataFrame(group_errors)),
                   ('matched_paired.csv',pd.DataFrame(paired)),('gvi_ols_buffer.csv',pd.DataFrame(ols))]:
    data.to_csv(OUT/name,index=False)
(OUT/'matched_source_hashes.json').write_text(json.dumps(sources,indent=2))
print(summary[summary.split=='test'].to_string(index=False))
print(pd.DataFrame(paired).to_string(index=False))
print('GVI fitted OLS buffer:',pd.DataFrame(ols)[['rmse','mae','r2']].mean().to_dict())
print('Audited 100 fits and 200 held-out prediction arrays. No training.')
