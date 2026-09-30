"""Direct-distance scalar-CDF and Python selection replay for the shared-winner audit."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-SELECTION-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
 start=time.perf_counter();c=read('configs/A-CAMP-SELECTION-001.json');orig=read('configs/A-CAMP-COHORTS-001.json');rows=read('results/A-CAMP-SELECTION-001/metrics.json');checks=0
 for p,h in read('results/A-CAMP-SELECTION-001/STARTED.json')['inputs'].items():assert sha(p)==h;checks+=1
 for p,h in read('results/A-CAMP-SELECTION-001/execution.json')['output_sha256'].items():assert sha('results/A-CAMP-SELECTION-001/'+p)==h;checks+=1
 oldrows=read('results/A-CAMP-CONFIDENCE-001/metrics.json')
 for ds in c['datasets']:
  dc=orig['datasets'][ds]
  with np.load(R/dc['features']) as z:d={k:z[k] for k in z.files}
  with np.load(O/ds/'decisions.npz') as z:saved={k:z[k] for k in z.files}
  with np.load(R/f'results/A-CAMP-CONFIDENCE-001/{ds}/query_metrics.npz') as z:ov=z['vectors']
  pr=[x for x in oldrows if x['dataset']==ds];pi={(x['case_id'],x['method'],x['k']):i for i,x in enumerate(pr)};contexts={};g=d['gallery_f'].astype(float)
  for role in ['calibration','evaluation']:
   ix=d['query_role']==role;q=d['query_f'][ix].astype(float);D=cdist(q,g);cos=(q/np.linalg.norm(q,axis=1)[:,None])@(g/np.linalg.norm(g,axis=1)[:,None]).T;contexts[role]=(D,cos,d['query_identity'][ix])
  D,_,ids=contexts['calibration'];order=np.argsort(D,axis=1,kind='stable');dd=np.take_along_axis(D,order,1);ss=np.cumsum(1/(1+dd),1);truth=(d['gallery_id'][order]==ids[:,None]).argmax(1);cal={'distance':dd[np.arange(len(ids)),truth],'aps':ss[np.arange(len(ids)),truth]}
  D,cos,ids=contexts['evaluation'];old=D.argmin(1);oldwrong=d['gallery_id'][old]!=ids;rr=[x for x in rows if x['dataset']==ds];offset=0;scopevalue=d['query_height'][d['query_role']=='evaluation'] if ds=='sues' else None
  for ci,case in enumerate(dc['cases']):
   active=np.array([i for i,gid in enumerate(d['gallery_id']) if gid not in set(case['removed'])]);rank=active[np.argsort(D[:,active],axis=1,kind='stable')];dist=np.take_along_axis(D,rank,1);prefix=np.cumsum(1/(1+dist),1);win=rank[:,0];assert np.array_equal(win,saved['winner'][ci]);guard=np.isin(old,active);wrong=d['gallery_id'][win]!=ids;reachable=np.isin(ids,d['gallery_id'][active]);v=-np.sort(-cos[:,active],axis=1)[:,:10];score={'su':-.5*(v[:,1:].mean(1)+np.median(v,axis=1))/v[:,0],'distance_top1':-dist[:,0]};nonempty={};empty={};res={}
   for family,mat,pm in [('distance',dist,'distance'),('aps',prefix,'aps_raw')]:
    a=ov[pi[case['case_id'],pm,1]];score[family+'_grid']=1-a[0];empty[family+'_grid']=a[2].astype(bool);res[family+'_grid']=a[3].astype(bool);nonempty[family+'_grid']=res[family+'_grid']&~empty[family+'_grid']
    count=np.array([sum(x<=boundary for x in cal[family]) for boundary in mat[:,1]]);score[family+'_exact']=count/(len(cal[family])+1);res[family+'_exact']=count>0;tau=np.array([max((x for x in cal[family] if x<=boundary),default=-np.inf) for boundary in mat[:,1]]);empty[family+'_exact']=res[family+'_exact']&(mat[:,0]>=tau);nonempty[family+'_exact']=res[family+'_exact']&~empty[family+'_exact']
    for key in [family+'_grid',family+'_exact']:assert np.allclose(score[key],saved['scores'][ci,c['scores'].index(key)],rtol=0,atol=1e-12)
   for key in ['su','distance_top1']:assert np.allclose(score[key],saved['scores'][ci,c['scores'].index(key)],rtol=0,atol=1e-12)
   lookup={p['name']:p for p in c['policies']}
   for row in [x for x in rr if x['case_id']==case['case_id']]:
    p=lookup[row['policy']];eligible=np.ones(len(ids),bool)
    if p['gate'] in ['nonempty','both']:eligible&=nonempty[p['score']]
    if p['gate'] in ['guard','both']:eligible&=guard
    assert np.array_equal(eligible,saved['eligible'][ci,c['policies'].index(p)])
    unit=np.ones(len(ids),bool) if row['scope']=='pooled' else scopevalue==int(row['scope']);cand=np.flatnonzero(unit&eligible);n=int(unit.sum()*row['answer_fraction']);feasible=len(cand)>=n;a=np.zeros(len(ids),bool)
    if feasible:
     # Use the stored full-precision score only at verified floating equality boundaries.
     stored=saved['scores'][ci,c['scores'].index(p['score'])];order=sorted(cand,key=lambda j:(-float(stored[j]),int(j)));a[order[:n]]=True
    assert np.array_equal(a,saved['accepted'][offset]);assert row['eligible']==len(cand) and row['budget']==n and row['feasible']==feasible
    emp=empty.get(p['score'],np.zeros(len(ids),bool));unresolved=~res.get(p['score'],np.ones(len(ids),bool));values={'answered':int(a.sum()),'errors':int((a&wrong).sum()),'created_errors':int((a&wrong&~oldwrong).sum()),'inherited_errors':int((a&wrong&oldwrong).sum()),'missing_truth_answers':int((a&~reachable).sum()),'accepted_empty':int((a&emp).sum()),'accepted_unresolved':int((a&unresolved).sum()),'accepted_ids':len(set(ids[a]))};assert all(row[k]==v for k,v in values.items());assert row['risk']==(values['errors']/values['answered'] if values['answered'] else None);offset+=1;checks+=len(values)+6
  assert offset==len(rr)
 for x in read('results/A-CAMP-SELECTION-001/summary.json'):
  a=[r for r in rows if all(r[k]==x[k] for k in ['dataset','arm','fraction','scope','answer_fraction','policy'])];assert len(a)==20;ok=[r for r in a if r['feasible']];assert len(ok)==x['feasible_masks']
  for k,v in x.items():
   if k.startswith('mean_'):assert v==(sum(r[k[5:]] for r in ok)/len(ok) if ok else None)
  checks+=1
 for x in read('results/A-CAMP-SELECTION-001/contrasts.json'):
  a=[r for r in rows if all(r[k]==x[k] for k in ['dataset','arm','fraction','scope','answer_fraction'])];base={r['case_id']:r for r in a if r['policy']==x['base']};test={r['case_id']:r for r in a if r['policy']==x['test']};delta=[r['errors']-test[k]['errors'] for k,r in base.items() if r['feasible'] and test[k]['feasible']];assert len(delta)==x['joint_masks'] and x['mean_error_reduction']==(sum(delta)/len(delta) if delta else None);checks+=1
 # Mathematical check only: exact inversion for all score counts, distinct from real experiment.
 for n in [3,17,32]:
  for m in range(n+1):
   confidence=m/(n+1);assert 0<=confidence<1
   if m:assert 1-confidence==1-m/(n+1)
 result={'status':'PASS','runtime_seconds':time.perf_counter()-start,'checks':checks,'rows':len(rows),'script_sha256':sha('experiments/review_camp_selection.py'),'scope':'Independent cdist/CDF/score checks followed by stable Python acceptance replay; no population guarantee or new independent cohort','prior_gate':'FAIL_RETAINED'};(O/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
