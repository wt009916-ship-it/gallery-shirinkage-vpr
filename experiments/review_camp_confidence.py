"""Direct threshold-count inversion review, independent of kth-boundary shortcut."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import hashlib,json,time,math
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-CONFIDENCE-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
 start=time.perf_counter();c=read('configs/A-CAMP-CONFIDENCE-001.json');orig=read('configs/A-CAMP-COHORTS-001.json');rows=read('results/A-CAMP-CONFIDENCE-001/metrics.json');th=read('results/A-CAMP-CONFIDENCE-001/thresholds.json');checks=0
 for p,h in read('results/A-CAMP-CONFIDENCE-001/STARTED.json')['inputs'].items():assert sha(p)==h;checks+=1
 for p,h in read('results/A-CAMP-CONFIDENCE-001/execution.json')['output_sha256'].items():assert sha('results/A-CAMP-CONFIDENCE-001/'+p)==h;checks+=1
 for ds in c['datasets']:
  dc=orig['datasets'][ds]
  with np.load(R/dc['features'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
  with np.load(O/ds/'query_metrics.npz',allow_pickle=False) as z:vec=z['vectors']
  rr=[x for x in rows if x['dataset']==ds];offset=0
  for role in ['calibration','evaluation']:
   ix=d['query_role']==role;D=cdist(d['query_f'][ix].astype(float),d['gallery_f'].astype(float));ids=d['query_identity'][ix]
   if role=='calibration':
    order=np.argsort(D,axis=1,kind='stable');dd=np.take_along_axis(D,order,1);cs=np.cumsum(1/(1+dd),1);truth=(d['gallery_id'][order]==ids[:,None]).argmax(1)
    for method,mat in [('distance',dd),('aps_raw',cs)]:
     scores=sorted(mat[np.arange(len(ids)),truth]);n=len(scores)
     for ai,alpha in enumerate(c['alpha_grid']):
      k=math.ceil((n+1)*(1-alpha));tau=scores[k-1] if k<=n else None;stored=th[ds+'/'+method][ai];assert (tau is None)==(stored is None)
      if tau is not None:assert abs(tau-stored)<1e-9
    continue
   for case in dc['cases']:
    active=np.flatnonzero(~np.isin(d['gallery_id'],case['removed']));order=np.argsort(D[:,active],axis=1,kind='stable');dd=np.take_along_axis(D[:,active],order,1);cs=np.cumsum(1/(1+dd),1);legal=d['gallery_id'][active[order]]==ids[:,None]
    for method,mat in [('distance',dd),('aps_raw',cs)]:
     tau=[x if x is not None else np.inf for x in th[ds+'/'+method]];counts=np.array([[np.searchsorted(v,t,side='left') for t in tau] for v in mat])
     for k in c['k_values']:
      u=[];empty=[];found=[]
      for counts_i in counts:
       possible=[j for j,v in enumerate(counts_i) if v<=k];has=bool(possible);j=possible[0] if has else 0;u.append(c['alpha_grid'][j] if has else 1.0);empty.append(has and counts_i[j]==0);found.append(has)
      u=np.array(u);empty=np.array(empty);found=np.array(found);correct=legal[:,:min(k,len(active))].any(1);assert np.array_equal(np.stack([u,correct,empty,found]),vec[offset]),(ds,case['case_id'],method,k)
      row=rr[offset];assert row['correct']==correct.sum() and row['empty_at_selected_alpha']==empty.sum() and row['grid_unresolved']==(~found).sum();prob=1-u;bid=np.minimum((prob*10).astype(int),9);ece=sum((bid==b).mean()*abs(prob[bid==b].mean()-correct[bid==b].mean()) for b in range(10) if (bid==b).any());assert abs(ece-row['ece10'])<1e-12
      for budgetrow in [x for x in read('results/A-CAMP-CONFIDENCE-001/top1_budgets.json') if k==1 and x['dataset']==ds and x['case_id']==case['case_id'] and x['method']==method]:
       take=sorted(range(len(u)),key=lambda j:(u[j],j))[:budgetrow['answered']];assert budgetrow['errors']==int((~correct[take]).sum()) and budgetrow['selected_empty']==int(empty[take].sum())
      offset+=1;checks+=6
  assert offset==len(rr)
 result={'status':'PASS','checks':checks,'runtime_seconds':time.perf_counter()-start,'rows':len(rows),'scope':'Numeric implementation of declared alpha-grid and strict inequality; no original-code or probabilistic guarantee certification','script_sha256':sha('experiments/review_camp_confidence.py')};(O/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
