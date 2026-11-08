"""Audit saved predictions and export the immutable tables consumed by R.

No fitting, model selection or checkpoint inference is performed.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'article/evidence'
SOURCES={}
def read(path):
 p=ROOT/path;SOURCES[path]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
def save(df,name):df.to_csv(OUT/name,index=False)
co=read('share/CO2/outputs/co2_temporal_13k_v1/combined_metrics.csv')
co=co[co.model.isin(['product_knn','deepgwr','geoaggregator','ols'])].copy()
co['model_label']=co.model.map({'product_knn':'DLAN','deepgwr':'Joint attention','geoaggregator':'GeoAggregator','ols':'OLS'})
assert len(co)==62 and all(co[co.model!='ols'].groupby(['model','split']).size()==10)
save(co,'co2_metrics.csv')
rows=[]; audits=[];bands=[]
for kind in ['product_knn','deepgwr','geoaggregator']:
 for seed in range(1,11):
  p=ROOT/f'share/CO2/outputs/co2_temporal_13k_v1/{kind}/seed_{seed}_predictions.npz'
  SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();a=np.load(p)
  for split in ['validation','test']:
   actual=a[f'{split}_actual'];pred=a[f'{split}_predicted'];ids=a[f'{split}_row_id'];err=pred-actual
   rmse=float(np.sqrt(np.mean(err**2)));mae=float(np.mean(abs(err)));r2=float(1-np.sum(err**2)/np.sum((actual-actual.mean())**2))
   m=co[(co.model==kind)&(co.seed==seed)&(co.split==split)].iloc[0]
   assert np.allclose([rmse,mae,r2],[m.rmse,m.mae,m.r2],rtol=2e-5,atol=1e-6)
   audits.append(dict(case='CO2',model=kind,seed=seed,split=split,rows=len(ids),rmse=rmse,mae=mae,r2=r2))
   if split=='test' and seed==1:
    rows.extend(dict(case='CO2',model=m.model_label,row_id=int(i),actual=float(y),predicted=float(z),residual=float(e)) for i,y,z,e in zip(ids,actual,pred,err))
  if kind=='product_knn':
   p=ROOT/f'share/CO2/outputs/co2_temporal_13k_v1/{kind}/seed_{seed}_checkpoint.pt'
   s=torch.load(p,map_location='cpu',weights_only=False)['state']['model_state']
   for key,value in s.items():
    if key.endswith('raw_bandwidth'):
     b=torch.nn.functional.softplus(value).numpy()+1e-6
     for axis,axisname in enumerate(['Space x','Space y','Time']):
      for head in range(b.shape[1]):bands.append(dict(seed=seed,axis=axisname,head=head,bandwidth=float(b[axis,head])))
save(pd.DataFrame(rows),'co2_predictions_seed1.csv');save(pd.DataFrame(bands),'co2_bandwidths.csv')
hpbase='share/WuhanHP/outputs/location_holdout_three_seeds/'
hp=read(hpbase+'test_metric_summary.csv');hp=hp[hp.model.isin(['DeepGWRTransformer','GeoAggregator','GNNWR','GCNNWR'])].copy();hp['model']=hp.model.replace({'DeepGWRTransformer':'DLAN-P'});save(hp,'hp_summary.csv')
data=np.load(ROOT/'share/WuhanHP/data/processed.npz');groups=data['group'];all_y=data['y_raw'].ravel();coords=data['u_raw']
hr=[];group_rows=[]
for kind,label in [('deepgwr','DLAN-P'),('geoaggregator','GeoAggregator'),('gnnwr','GNNWR'),('gcnnwr','GCNNWR')]:
 for seed in [10,20,30]:
  p=ROOT/(hpbase+f'{kind}_seed{seed}_predictions.npz');a=np.load(p);ids=a['test_row_id'].astype(int);y=all_y[ids];pred=a['test_predicted'];e=pred-y
  metrics=read(hpbase+f'{kind}_seed{seed}_metrics.csv');m=metrics[metrics.split=='test'].iloc[0]
  rmse=np.sqrt(np.mean(e**2));assert np.isclose(rmse,m.rmse,rtol=2e-5)
  audits.append(dict(case='WuhanHP',model=label,seed=seed,split='test',rows=len(ids),rmse=rmse))
  if seed==10:
   d=pd.DataFrame(dict(group=groups[ids],actual=y,predicted=pred,residual=e,x=coords[ids,0],y=coords[ids,1]))
   for group,g in d.groupby('group'):
    group_rows.append(dict(model=label,group=group,n=len(g),rmse=np.sqrt(np.mean(g.residual**2)),mean_residual=g.residual.mean(),x=g.x.iloc[0],y=g.y.iloc[0]))
save(pd.DataFrame(group_rows),'hp_group_errors_seed10.csv')
od=read('share/OD-Wuhan/outputs/gtdr_reference_support_10seeds_v1/ordinary/product_knn/metrics.csv')
od=od[(od.seed.isin([1,2,3,4]))&(od.split=='test')].copy();assert len(od)==4
od['model_label']='DLAN (partial)'
for _,m in od.iterrows():
 p=ROOT/f'share/OD-Wuhan/outputs/gtdr_reference_support_10seeds_v1/ordinary/product_knn/seed_{int(m.seed)}_predictions.npz';a=np.load(p)
 print('OD keys',a.files) if m.seed==1 else None
 y=a['test_actual'];pred=a['test_predicted'];rmse=np.sqrt(np.mean((pred-y)**2));assert np.isclose(rmse,m.rmse,rtol=2e-5)
 audits.append(dict(case='WuhanOD',model='DLAN',seed=m.seed,split='test',rows=len(y),rmse=rmse))
save(od,'od_partial_metrics.csv')
odhist=read('share/OD-Wuhan/outputs/gtdr_product256_10seeds_v1/test_comparison.csv');save(odhist,'od_historical_summary.csv')
ab=read('output/deepgwr_kernel_ablation/gaussian_vs_learned/analysis/comparison.csv');save(ab[(ab.split=='test')&ab.dataset.isin(['WuhanHP','OD-Wuhan'])],'gaussian_ablation.csv')
save(pd.DataFrame(audits),'prediction_recalculation.csv')
(OUT/'source_hashes.json').write_text(json.dumps(SOURCES,indent=2))
print('Audited',len(audits),'prediction sets; no model trained.')
