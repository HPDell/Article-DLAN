"""Select explicitly reported configurations without overwriting controlled history.

The reported WuhanOD DLAN is width 12 / head (10,8) / K512 at 7,069
parameters; its comparators are the complete 7,069-parameter budget group.
The other cases retain their 6,957-parameter configurations.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'article/evidence'
main=pd.read_csv(OUT/'matched_metrics.csv');extra=pd.read_csv(OUT/'supplement_metrics.csv')
parts=[main[main.case!='OD-Wuhan'].copy()]
od=extra[((extra.experiment=='OD capacity')&extra.model_label.isin(['GeoAggregator','GNNWR']))|((extra.experiment=='OD neighbours')&(extra.model_label=='K512'))].copy()
od.loc[od.model_label=='K512','model_label']='DLAN'
od['case']='OD-Wuhan';parts.append(od)
reported=pd.concat(parts,ignore_index=True);assert len(reported)==280
# Verify the exact selected K512 mean from saved predictions again.
folder=ROOT/'experiments/WuhanOD-DLANNeighbors/outputs/wuhan_od_dlan_knn512_10_v1/dlan'
config=json.loads((folder/'configuration.json').read_text())
assert config['candidate_count']==512 and config['parameters']==7069
raw=np.load(ROOT/'share/OD-Wuhan/data/dlan_matched4_10seeds_v1/ordinary.npz')
checks=[];bands=[]
for seed in range(1,11):
 a=np.load(folder/f'seed_{seed}_predictions.npz')
 for split in ['validation','test']:
  actual=a[f'{split}_actual'].astype(float);pred=a[f'{split}_predicted'].astype(float)
  np.testing.assert_array_equal(a[f'{split}_row_id'],raw['row_id'][raw[f'{split}_ids']])
  np.testing.assert_allclose(actual,raw['actual'][raw[f'{split}_ids']],rtol=1e-6)
  rmse=np.sqrt(np.mean((pred-actual)**2));row=reported[(reported.case=='OD-Wuhan')&(reported.model_label=='DLAN')&(reported.seed==seed)&(reported.split==split)].iloc[0]
  np.testing.assert_allclose(rmse,row.rmse,rtol=1e-12)
  checks.append(dict(seed=seed,split=split,rmse=rmse))
 state=torch.load(folder/f'seed_{seed}_checkpoint.pt',map_location='cpu',weights_only=False)['state']['model_state']
 assert state['blocks.0.q_proj.weight'].shape==(48,12)
 assert state['blocks.0.k_proj.weight'].shape==(48,12)
 for key,value in state.items():
  if key.endswith('raw_bandwidth'):
   b=torch.nn.functional.softplus(value).numpy()+1e-6
   for axis,name in enumerate(['Origin x','Origin y','Destination x','Destination y']):
    bands.append(dict(case='OD-Wuhan',seed=seed,axis=name,bandwidth=float(b[axis,0])))
pd.DataFrame(checks).to_csv(OUT/'od_k512_reaudit.csv',index=False)
expected=565.2762909669982
np.testing.assert_allclose(reported[(reported.case=='OD-Wuhan')&(reported.model_label=='DLAN')&(reported.split=='test')].rmse.mean(),expected,rtol=1e-14)
sp=pd.read_csv(OUT/'supplement_predictions.csv')
selected_pred=sp[((sp.experiment=='OD capacity')&sp.model.isin(['GeoAggregator','GNNWR']))|((sp.experiment=='OD neighbours')&(sp.model=='K512'))].copy()
selected_pred.loc[selected_pred.model=='K512','model']='DLAN'
for (label,seed),g in selected_pred.groupby(['model','seed']):
 actual=g.actual.to_numpy();prediction=g.predicted.to_numpy();tail=actual>=np.quantile(actual,.99)
 mask=(reported.case=='OD-Wuhan')&(reported.model_label==label)&(reported.seed==seed)&(reported.split=='test')
 reported.loc[mask,'upper_1pct_sse_share']=np.sum((prediction[tail]-actual[tail])**2)/np.sum((prediction-actual)**2)
 reported.loc[mask,'body_rmse']=np.sqrt(np.mean((prediction[~tail]-actual[~tail])**2))
summary=reported.groupby(['case','model_label','split']).agg(runs=('seed','nunique'),parameters=('parameters','first'),
 rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),rmse_median=('rmse','median'),mae_mean=('mae','mean'),r2_mean=('r2','mean'),
 seconds_mean=('seconds','mean'),epochs_mean=('epochs','mean'),equal_group_rmse=('equal_group_rmse','mean')).reset_index()
reported.to_csv(OUT/'reported_metrics.csv',index=False);summary.to_csv(OUT/'reported_summary.csv',index=False)
pred=pd.read_csv(OUT/'matched_predictions.csv');sp=pd.read_csv(OUT/'supplement_predictions.csv')
odpred=sp[((sp.experiment=='OD capacity')&sp.model.isin(['GeoAggregator','GNNWR']))|((sp.experiment=='OD neighbours')&(sp.model=='K512'))].copy();odpred.loc[odpred.model=='K512','model']='DLAN'
pd.concat([pred[pred.case!='OD-Wuhan'],odpred],ignore_index=True).to_csv(OUT/'reported_predictions.csv',index=False)
band=pd.read_csv(OUT/'matched_bandwidths.csv');pd.concat([band[band.case!='OD-Wuhan'],pd.DataFrame(bands)],ignore_index=True).to_csv(OUT/'reported_bandwidths.csv',index=False)
manifest={'OD-Wuhan':{'DLAN':str(folder.relative_to(ROOT)),'GeoAggregator':'share/OD-Wuhan/outputs/wuhan_od_attention_capacity10_v1/geoaggregator','GNNWR':'share/OD-Wuhan/outputs/gnnwr_upstream_attention10_v1/gnnwr','parameters':7069,'attention_width':12,'head':[10,8],'DLAN_candidates':512,'cross_machine':True},'other_cases':{'source':'matched_metrics.csv','parameters':6957},'selection':'K512 reported at user request; K1024, K2048 and old allocation remain separate configuration comparisons'}
(OUT/'reported_configuration_manifest.json').write_text(json.dumps(manifest,indent=2))
print(summary[summary.split=='test'].to_string(index=False))
