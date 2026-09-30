"""Literal finite-grid fixed-k uncertainty adaptation; descriptive calibration only."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import csv,hashlib,json,math,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-CONFIDENCE-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
 start=time.perf_counter();c=read('configs/A-CAMP-CONFIDENCE-001.json');orig=read('configs/A-CAMP-COHORTS-001.json');assert read('results/A-CAMP-ROBUSTNESS-001/review.json')['status']=='PASS';git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)]
 paths=['configs/A-CAMP-CONFIDENCE-001.json','experiments/run_camp_confidence.py','experiments/review_camp_confidence.py']
 for p in paths:subprocess.run([*git,'diff','HEAD','--exit-code','--',p],check=True,capture_output=True);subprocess.run([*git,'ls-files','--error-unmatch',p],check=True,capture_output=True)
 for p,h in c['inputs'].items():assert sha(p)==h
 O.mkdir(parents=True,exist_ok=True)
 with (O/'STARTED.json').open('x') as f:json.dump({'epoch':time.time(),'commit':subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip(),'inputs':{p:sha(p) for p in paths}|c['inputs']},f,indent=2)
 rows=[];thresholds={};budgets=[];bins=[];grid=np.array(c['alpha_grid'])
 for ds in c['datasets']:
  dc=orig['datasets'][ds]
  with np.load(R/dc['features'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
  contexts={};vectors=[]
  for role in ['calibration','evaluation']:
   ix=d['query_role']==role;q=d['query_f'][ix].astype(float);g=d['gallery_f'].astype(float);D=np.sqrt(np.maximum(0,(q*q).sum(1)[:,None]+(g*g).sum(1)[None,:]-2*q@g.T));contexts[role]=(D,d['query_identity'][ix])
  D,ids=contexts['calibration'];order=np.argsort(D,axis=1,kind='stable');dd=np.take_along_axis(D,order,axis=1);cs=np.cumsum(1/(1+dd),axis=1);true=(d['gallery_id'][order]==ids[:,None]).argmax(1)
  for method,mat in [('distance',dd),('aps_raw',cs)]:
   scores=np.sort(mat[np.arange(len(ids)),true]);nn=len(scores);thresholds[ds+'/'+method]=[float(scores[math.ceil((nn+1)*(1-alpha))-1]) if math.ceil((nn+1)*(1-alpha))<=nn else None for alpha in grid]
  D,ids=contexts['evaluation']
  for case in dc['cases']:
   active=np.flatnonzero(~np.isin(d['gallery_id'],case['removed']));order=np.argsort(D[:,active],axis=1,kind='stable');dd=np.take_along_axis(D[:,active],order,axis=1);cs=np.cumsum(1/(1+dd),axis=1);legal=d['gallery_id'][active[order]]==ids[:,None];reach=legal.any(1);n=len(ids);m=len(active)
   for method,mat in [('distance',dd),('aps_raw',cs)]:
    tau=np.array([x if x is not None else np.inf for x in thresholds[ds+'/'+method]])
    for k in c['k_values']:
     cap=min(k,m);can=np.ones((n,len(grid)),bool) if k>=m else mat[:,k,None]>=tau[None,:];found=can.any(1);j=can.argmax(1);u=np.where(found,grid[j],1.0);empty=found&(mat[:,0]>=tau[j]);correct=legal[:,:cap].any(1);prob=1-u;bid=np.minimum((prob*10).astype(int),9);ece=0
     row=dict(dataset=ds,case_id=case['case_id'],arm=case['arm'],fraction=case['fraction'],method=method,k=k,effective_k=cap,queries=n,reachable=int(reach.sum()),correct=int(correct.sum()),retrieval_coverage=float(correct.mean()),mean_confidence=float(prob.mean()),grid_unresolved=int((~found).sum()),empty_at_selected_alpha=int(empty.sum()),wrong_topk_with_empty=int((empty&~correct).sum()))
     for b in range(10):
      take=bid==b
      if take.any():
       acc=float(correct[take].mean());conf=float(prob[take].mean());ece+=take.mean()*abs(acc-conf);bins.append(dict(dataset=ds,case_id=case['case_id'],method=method,k=k,bin=b,count=int(take.sum()),accuracy=acc,confidence=conf))
     row['ece10']=float(ece);rows.append(row);vectors.append(np.stack([u,correct,empty,found]))
     if k==1:
      for frac in [.1,.25,.5]:
       budget=int(n*frac);take=np.argsort(u,kind='stable')[:budget];budgets.append(dict(dataset=ds,case_id=case['case_id'],arm=case['arm'],fraction=case['fraction'],method=method,answer_fraction=frac,answered=budget,errors=int((~correct[take]).sum()),selected_empty=int(empty[take].sum())))
  target=O/ds;target.mkdir();np.savez_compressed(target/'query_metrics.npz',vectors=np.stack(vectors),query_identity=ids)
 for name,obj in [('metrics',rows),('thresholds',thresholds),('bins',bins),('top1_budgets',budgets)]:
  (O/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
  if isinstance(obj,list):
   with (O/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(obj[0]));w.writeheader();w.writerows(obj)
 ex={'status':'EXECUTED_PENDING_REVIEW','runtime_seconds':time.perf_counter()-start,'rows':len(rows),'output_sha256':{p.relative_to(O).as_posix():sha(p.relative_to(R)) for p in O.rglob('*') if p.is_file() and p.name!='STARTED.json'}};(O/'execution.json').write_text(json.dumps(ex,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in ex.items() if k!='output_sha256'}))
if __name__=='__main__':main()
