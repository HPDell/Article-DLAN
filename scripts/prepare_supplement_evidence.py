"""Audit completed extensions, keeping budgets, machines and protocols separate."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'article/evidence'
MAIN='dlan_matched4_10seeds_v1'
SOURCES={}
def record(p):
    p=Path(p);SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p

jobs=[]
for kind,label in [('dlan','DLAN'),('geoaggregator','GeoAggregator')]:
    jobs.append(('OD capacity','OD-Wuhan',label,ROOT/'share/OD-Wuhan/outputs/wuhan_od_attention_capacity10_v1'/kind,
        ROOT/'share/OD-Wuhan/data/dlan_matched4_10seeds_v1/ordinary.npz',7069))
jobs.append(('OD capacity','OD-Wuhan','GNNWR',ROOT/'share/OD-Wuhan/outputs/gnnwr_upstream_attention10_v1/gnnwr',
    ROOT/'share/OD-Wuhan/data/dlan_matched4_10seeds_v1/ordinary.npz',7069))
for kind,label in [('dlan','DLAN'),('geoaggregator','GeoAggregator')]:
    jobs.append(('GVI 3D','GVI',label,ROOT/'share/GVI/outputs/gvi_spatiotemporal_matched10_v1'/kind,
        ROOT/'share/GVI/data/gvi_spatiotemporal_matched10_v1/ordinary.npz',6957))
jobs.append(('OD width16','OD-Wuhan','DLAN width16',ROOT/'share/OD-Wuhan/outputs/wuhan_od_dlan_width16_10_v1/dlan',
    ROOT/'share/OD-Wuhan/data/dlan_matched4_10seeds_v1/ordinary.npz',10773))
for k in [512,2048]:
    jobs.append(('OD neighbours','OD-Wuhan',f'K{k}',ROOT/f'experiments/WuhanOD-DLANNeighbors/outputs/wuhan_od_dlan_knn{k}_10_v1/dlan',
        ROOT/'share/OD-Wuhan/data/dlan_matched4_10seeds_v1/ordinary.npz',7069))
rows=[];predictions=[]
for experiment,case,label,folder,data_path,expected in jobs:
    config=json.loads(record(folder/'configuration.json').read_text())
    assert config['parameters']==expected
    assert hashlib.sha256(record(data_path).read_bytes()).hexdigest()==config['contract']['data_sha256']
    data=np.load(data_path)
    base=np.load(ROOT/'share'/case/'data'/MAIN/'ordinary.npz')
    for key in ['x','y','actual','row_id','groups','train_ids','validation_ids','test_ids']:
        np.testing.assert_array_equal(data[key],base[key])
    for seed in range(1,11):
        frame=pd.read_csv(record(folder/f'seed_{seed}_metrics.csv'))
        assert len(frame)==2 and set(frame.config_digest)=={config['config_digest']}
        assert frame.parameters.eq(expected).all()
        a=np.load(record(folder/f'seed_{seed}_predictions.npz'))
        audit=pd.read_csv(record(folder/f'seed_{seed}_audit.csv'))
        assert len(audit)==2 and audit.prediction_reload_pass.all()
        state=torch.load(record(folder/f'seed_{seed}_checkpoint.pt'),map_location='cpu',weights_only=False)
        assert state['configuration']['config_digest']==config['config_digest']
        np.testing.assert_array_equal(state['state']['context_ids'],data['train_ids'])
        for split in ['validation','test']:
            ids=a[f'{split}_row_id'];actual=a[f'{split}_actual'].astype(float);pred=a[f'{split}_predicted'].astype(float)
            np.testing.assert_array_equal(ids,data['row_id'][data[f'{split}_ids']])
            np.testing.assert_allclose(actual,data['actual'][data[f'{split}_ids']],rtol=1e-6,atol=1e-6)
            err=pred-actual
            calc=dict(rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err)),r2=1-np.sum(err**2)/np.sum((actual-actual.mean())**2))
            row=frame[frame.split==split].iloc[0].to_dict()
            for key,v in calc.items():np.testing.assert_allclose(v,row[key],rtol=2e-5,atol=1e-6)
            row.update(calc,experiment=experiment,model_label=label)
            row['log1p_rmse']=np.sqrt(np.mean((np.log1p(pred)-np.log1p(actual))**2)) if case=='OD-Wuhan'else np.nan
            rows.append(row)
            if split=='test':predictions.append(pd.DataFrame(dict(experiment=experiment,case=case,model=label,seed=seed,row_id=ids,actual=actual,predicted=pred)))
extra=pd.DataFrame(rows);assert len(extra)==160
summary=extra.groupby(['experiment','case','model_label','split']).agg(runs=('seed','nunique'),parameters=('parameters','first'),
    rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),rmse_median=('rmse','median'),rmse_worst=('rmse','max'),
    mae_mean=('mae','mean'),r2_mean=('r2','mean'),seconds_mean=('seconds','mean'),epochs_mean=('epochs','mean'),log1p_rmse_mean=('log1p_rmse','mean')).reset_index()
extra.to_csv(OUT/'supplement_metrics.csv',index=False);summary.to_csv(OUT/'supplement_summary.csv',index=False)
pd.concat(predictions,ignore_index=True).to_csv(OUT/'supplement_predictions.csv',index=False)
main=pd.read_csv(OUT/'matched_metrics.csv');pair=[]
for label in ['DLAN','GeoAggregator']:
    p=extra[(extra.experiment=='GVI 3D')&(extra.model_label==label)&(extra.split=='test')].set_index('seed')
    q=main[(main.case=='GVI')&(main.model_label==label)&(main.split=='test')].set_index('seed')
    pair.append(dict(model=label,rmse_2d=q.rmse.mean(),rmse_3d=p.rmse.mean(),rmse_3d_sd=p.rmse.std(),wins_3d=int((p.rmse<q.rmse).sum()),reduction_pct=(1-p.rmse.mean()/q.rmse.mean())*100))
pd.DataFrame(pair).to_csv(OUT/'gvi_2d_3d_paired.csv',index=False)
# Freeze ongoing diagnostic status without counting its partial results.
p=ROOT/'runs/wuhan_od_hpc_parity_v1/status.json'
(OUT/'hpc_parity_status_snapshot.json').write_text(record(p).read_text())
# The complete historical HPC model is a separate execution protocol.
hfolder=ROOT/'hpc/OD-Wuhan/outputs/gtdr_reference_support_hpc_v1/ordinary/product_knn'
h=[]
for seed in range(1,11):
    m=pd.read_csv(record(hfolder/f'seed_{seed}_metrics.csv'))
    a=np.load(record(hfolder/f'seed_{seed}_predictions.npz'))
    for split in ['validation','test']:
        actual=a[f'{split}_actual'].astype(float);pred=a[f'{split}_predicted'].astype(float)
        base=np.load(ROOT/'share/OD-Wuhan/data'/MAIN/'ordinary.npz')
        np.testing.assert_array_equal(a[f'{split}_row_id'],base['row_id'][base[f'{split}_ids']])
        np.testing.assert_allclose(actual,base['actual'][base[f'{split}_ids']],rtol=1e-6,atol=1e-6)
        rmse=np.sqrt(np.mean((pred-actual)**2));r=m[m.split==split].iloc[0].to_dict()
        np.testing.assert_allclose(rmse,r['rmse'],rtol=2e-5);h.append(r)
pd.DataFrame(h).to_csv(OUT/'historical_hpc_od_metrics.csv',index=False)
(OUT/'supplement_source_hashes.json').write_text(json.dumps(SOURCES,indent=2))
print(summary[summary.split=='test'].to_string(index=False));print(pd.DataFrame(pair).to_string(index=False))
print('Audited 80 supplementary fits / 160 prediction sets, plus 10 historical HPC fits / 20 prediction sets.')
