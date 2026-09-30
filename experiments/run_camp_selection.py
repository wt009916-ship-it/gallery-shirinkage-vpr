"""Shared-prediction fixed-answer audit of empty sets, eligibility and alpha discretization."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-SELECTION-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
 start=time.perf_counter();c=read('configs/A-CAMP-SELECTION-001.json');orig=read('configs/A-CAMP-COHORTS-001.json');git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)]
 paths=['configs/A-CAMP-SELECTION-001.json','experiments/run_camp_selection.py','experiments/review_camp_selection.py']
 for p in paths:subprocess.run([*git,'ls-files','--error-unmatch',p],check=True,capture_output=True);subprocess.run([*git,'diff','HEAD','--exit-code','--',p],check=True,capture_output=True)
 for p,h in c['inputs'].items():assert sha(p)==h
 assert read('results/A-CAMP-CONFIDENCE-001/review.json')['status']=='PASS'
 dependencies=['results/A-CAMP-CONFIDENCE-001/'+p for p in ['metrics.json','review.json','university/query_metrics.npz','sues/query_metrics.npz']]
 c['inputs'].update({p:sha(p) for p in dependencies})
 O.mkdir(parents=True,exist_ok=True)
 with (O/'STARTED.json').open('x') as f:json.dump({'epoch':time.time(),'commit':subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip(),'inputs':{p:sha(p) for p in paths}|c['inputs']},f,indent=2)
 rows=[];diag=[]
 for ds in c['datasets']:
  dc=orig['datasets'][ds]
  with np.load(R/dc['features']) as z:d={k:z[k] for k in z.files}
  q=d['query_f'].astype(float);g=d['gallery_f'].astype(float);roles=d['query_role'];contexts={}
  for role in ['calibration','evaluation']:
   ix=roles==role;qr=q[ix];D=np.sqrt(np.maximum(0,(qr*qr).sum(1)[:,None]+(g*g).sum(1)[None,:]-2*qr@g.T));cos=(qr@g.T)/(np.linalg.norm(qr,axis=1)[:,None]*np.linalg.norm(g,axis=1)[None,:]);contexts[role]=(D,cos,d['query_identity'][ix])
  D,_,ids=contexts['calibration'];rank=np.argsort(D,axis=1,kind='stable');dd=np.take_along_axis(D,rank,1);cs=np.cumsum(1/(1+dd),1);tr=(d['gallery_id'][rank]==ids[:,None]).argmax(1);cal={'distance':np.sort(dd[np.arange(len(ids)),tr]),'aps':np.sort(cs[np.arange(len(ids)),tr])}
  previous=read('results/A-CAMP-CONFIDENCE-001/metrics.json');pr=[x for x in previous if x['dataset']==ds];pi={(x['case_id'],x['method'],x['k']):i for i,x in enumerate(pr)}
  with np.load(R/f'results/A-CAMP-CONFIDENCE-001/{ds}/query_metrics.npz') as z:oldv=z['vectors']
  D,cos,ids=contexts['evaluation'];n=len(ids);oldwin=D.argmin(1);oldwrong=d['gallery_id'][oldwin]!=ids;scope_values=d['query_height'][roles=='evaluation'] if ds=='sues' else None
  accepted=[];scores_all=[];eligible_all=[];winners=[];diagnostic_vectors=[]
  for ci,case in enumerate(dc['cases']):
   active=np.flatnonzero(~np.isin(d['gallery_id'],case['removed']));rank=active[np.argsort(D[:,active],axis=1,kind='stable')];dist=np.take_along_axis(D,rank,1);sums=np.cumsum(1/(1+dist),1);win=rank[:,0];winners.append(win);guard=np.isin(oldwin,active);assert np.array_equal(win[guard],oldwin[guard]);wrong=d['gallery_id'][win]!=ids;reachable=np.isin(ids,d['gallery_id'][active])
   vv=np.sort(cos[:,active],axis=1)[:,-10:];assert vv[:,9].min()>0;su=-(vv[:,:9].mean(1)+(vv[:,4]+vv[:,5])/2)/(2*vv[:,9]);score={'su':su,'distance_top1':-dist[:,0]};nonempty={};empty={};resolved={}
   dv=[]
   for family,mat,pm in [('distance',dist,'distance'),('aps',sums,'aps_raw')]:
    uv=oldv[pi[case['case_id'],pm,1]];assert np.array_equal(uv[1].astype(bool),~wrong)
    score[family+'_grid']=1-uv[0];empty[family+'_grid']=uv[2].astype(bool);resolved[family+'_grid']=uv[3].astype(bool);nonempty[family+'_grid']=resolved[family+'_grid']&~empty[family+'_grid']
    count=np.searchsorted(cal[family],mat[:,1],side='right');exact=count/(len(cal[family])+1);tau=cal[family][np.maximum(0,count-1)];res=count>0;emp=res&(mat[:,0]>=tau)
    score[family+'_exact']=exact;empty[family+'_exact']=emp;resolved[family+'_exact']=res;nonempty[family+'_exact']=res&~emp
    dv.extend([count,emp,res])
    diag.append(dict(dataset=ds,case_id=case['case_id'],arm=case['arm'],fraction=case['fraction'],family=family,queries=n,grid_empty=int(empty[family+'_grid'].sum()),exact_empty=int(emp.sum()),grid_unresolved=int((~resolved[family+'_grid']).sum()),exact_unresolved=int((~res).sum()),grid_unique_scores=len(np.unique(score[family+'_grid'])),exact_unique_scores=len(np.unique(exact)),guard_eligible=int(guard.sum()),missing_truth=int((~reachable).sum()),l2_vs_normalized_cosine_prediction_disagreements=int((win!=active[cos[:,active].argmax(1)]).sum())))
   diagnostic_vectors.append(np.stack(dv));scores_all.append(np.stack([score[m] for m in c['scores']]))
   elig={}
   for policy in c['policies']:
    gate=policy['gate'];key=policy['name'];a=np.ones(n,bool)
    if gate in ['nonempty','both']:a&=nonempty[policy['score']]
    if gate in ['guard','both']:a&=guard
    elig[key]=a
   eligible_all.append(np.stack([elig[p['name']] for p in c['policies']]))
   for scope in dc['scopes']:
    unit=np.ones(n,bool) if scope=='pooled' else scope_values==int(scope)
    for policy in c['policies']:
     key=policy['name'];cand=np.flatnonzero(unit&elig[key]);order=cand[np.argsort(-score[policy['score']][cand],kind='stable')]
     for af in c['answer_fractions']:
      budget=int(unit.sum()*af);feasible=len(cand)>=budget;a=np.zeros(n,bool)
      if feasible:a[order[:budget]]=True
      e=int((a&wrong).sum());an=int(a.sum());ne=int((a&wrong&~oldwrong).sum());emp=empty.get(policy['score'],np.zeros(n,bool));unres=~resolved.get(policy['score'],np.ones(n,bool))
      rows.append(dict(dataset=ds,case_id=case['case_id'],arm=case['arm'],fraction=case['fraction'],scope=scope,policy=key,score=policy['score'],gate=policy['gate'],answer_fraction=af,queries=int(unit.sum()),budget=budget,eligible=len(cand),feasible=feasible,answered=an,errors=e,risk=e/an if an else None,created_errors=ne,inherited_errors=e-ne,missing_truth_answers=int((a&~reachable).sum()),accepted_empty=int((a&emp).sum()),accepted_unresolved=int((a&unres).sum()),accepted_ids=len(np.unique(ids[a]))));accepted.append(a)
      if policy['gate'] in ['guard','both']:assert ne==0
  target=O/ds;target.mkdir();np.savez_compressed(target/'decisions.npz',accepted=np.stack(accepted),winner=np.stack(winners),scores=np.stack(scores_all),eligible=np.stack(eligible_all),diagnostic_vectors=np.stack(diagnostic_vectors),query_identity=ids);print(json.dumps({'dataset':ds,'rows':len(accepted)}),flush=True)
 summary=[];contrasts=[]
 for ds in c['datasets']:
  rr=[x for x in rows if x['dataset']==ds];lookup={(x['case_id'],x['scope'],x['answer_fraction'],x['policy']):x for x in rr}
  for arm,frac in [('target',.25),('control',.25),('target',.5)]:
   for scope in orig['datasets'][ds]['scopes']:
    for af in c['answer_fractions']:
     for p in c['policies']:
      aa=[x for x in rr if (x['arm'],x['fraction'],x['scope'],x['answer_fraction'],x['policy'])==(arm,frac,scope,af,p['name'])];assert len(aa)==20;ok=[x for x in aa if x['feasible']];summary.append(dict(dataset=ds,arm=arm,fraction=frac,scope=scope,answer_fraction=af,policy=p['name'],feasible_masks=len(ok),**{'mean_'+k:sum(x[k] for x in ok)/len(ok) if ok else None for k in ['answered','errors','created_errors','missing_truth_answers','accepted_empty','accepted_unresolved']}))
     for base,test in c['contrasts']:
      aa=[x for x in rr if (x['arm'],x['fraction'],x['scope'],x['answer_fraction'],x['policy'])==(arm,frac,scope,af,base)];pair=[(x,lookup[x['case_id'],scope,af,test]) for x in aa];joint=[(a,b) for a,b in pair if a['feasible'] and b['feasible']];delta=[a['errors']-b['errors'] for a,b in joint];contrasts.append(dict(dataset=ds,arm=arm,fraction=frac,scope=scope,answer_fraction=af,base=base,test=test,joint_masks=len(joint),mean_error_reduction=sum(delta)/len(delta) if delta else None,min_reduction=min(delta) if delta else None,max_reduction=max(delta) if delta else None))
 for name,obj in [('metrics',rows),('summary',summary),('contrasts',contrasts),('diagnostics',diag)]:
  (O/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
  with (O/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(obj[0]));w.writeheader();w.writerows(obj)
 ex={'status':'EXECUTED_PENDING_REVIEW','rows':len(rows),'runtime_seconds':time.perf_counter()-start,'output_sha256':{p.relative_to(O).as_posix():sha(p.relative_to(R)) for p in O.rglob('*') if p.is_file() and p.name!='STARTED.json'}};(O/'execution.json').write_text(json.dumps(ex,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in ex.items() if k!='output_sha256'}))
if __name__=='__main__':main()
