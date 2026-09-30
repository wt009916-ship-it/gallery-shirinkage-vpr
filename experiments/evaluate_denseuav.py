"""Fixed two-region / temporal-gallery experiment. Identity is primary.

All geographic secondary labels rescore the identical accepted sets. For
old->plain, a surviving historical winner need NOT remain the current winner.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,time,csv
from pathlib import Path
import numpy as np
from evaluate_camp import su_margin,fit
from run_update_memory import encode
from freeze_denseuav import distance
R=Path(__file__).resolve().parents[1];O=R/'results/A-DENSEUAV-EXTERNAL-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 c=json.loads((R/'configs/A-DENSEUAV-EXTERNAL-001.json').read_text());inf=json.loads((O/'inference.json').read_text());assert sha(O/'features.npz')==inf['features_sha256']
 for p,h in inf['input_sha256'].items():assert sha(R/p)==h,p
 with (O/'EVALUATION_STARTED.json').open('x') as f:json.dump(dict(epoch=time.time(),config_sha256=sha(R/'configs/A-DENSEUAV-EXTERNAL-001.json')),f)
 start=time.perf_counter();z=np.load(O/'features.npz');qid=z['query_id'];gid=z['gallery_id'];q=z['query_f'];geo=distance(np.array([c['coords'][str(i)] for i in qid]),np.array([c['coords'][str(i)] for i in gid]));rows=[];accepted=[];pred=[];raw=[];thresholds=[];full=[];scores={v:q@z['gallery_'+v].T for v in ['old','plain']}
 fullfits={}
 for condition in c['conditions']:
  name=condition['name'];h=scores[condition['history']];s=scores[condition['current']];hr=np.argsort(-h,axis=1,kind='stable');old=hr[:,0];oldsu,oldmargin=su_margin(np.take_along_axis(h,hr[:,:10],axis=1));oldwrong=gid[old]!=qid
  _,_,dec,res=encode(z['gallery_'+condition['history']],8);upper=(q@dec.T).astype(float)+np.linalg.norm(q.astype(float),axis=1)[:,None]*res[None,:]+1e-6;assert (upper>=h).all();fixed={}
  for case in c['cases']:
   active=np.flatnonzero(~np.isin(gid,case['removed']));dead=np.flatnonzero(np.isin(gid,case['removed']));rank=active[np.argsort(-s[:,active],axis=1,kind='stable')];win=rank[:,0];v=np.take_along_axis(s,rank[:,:10],axis=1);su,margin=su_margin(v);guard=np.isin(old,active);bound=upper[:,dead].max(1) if len(dead) else np.full(len(q),-np.inf);sqvalid=v[:,0]>bound;allok=np.ones(len(q),bool);ss=[(su,allok),(su,guard),(oldsu,allok),(oldsu,guard),(margin,allok),(margin,guard),(oldmargin,guard),(v[:,0]-np.maximum(v[:,1],bound),sqvalid)];wrong=gid[win]!=qid;meters=geo[np.arange(len(qid)),win];created=wrong&~oldwrong;inherited=wrong&oldwrong
   if condition['history']==condition['current']:
    assert np.array_equal(win[guard],old[guard]) and guard[sqvalid].all()
   pi=len(pred);pred.append(win)
   for fold in c['folds']:
    ca=np.isin(qid,fold['calibration']);ev=np.isin(qid,fold['evaluation']);n=int(ev.sum());assert not (ca&ev).any() and (ca|ev).all();full.append(dict(condition=name,fold=fold['name'],queries=n,correct=int((ev&~wrong).sum()),reference_distance_median=float(np.median(meters[ev])),reference_within20=int((ev&(meters<=20)).sum()),reference_within50=int((ev&(meters<=50)).sum()))) if case['arm']=='full' else None
    raw.append(dict(condition=name,fold=fold['name'],case_id=case['case_id'],wrong=int((ev&wrong).sum()),created=int((ev&created).sum()),inherited=int((ev&inherited).sum()),historical_wrong=int((ev&oldwrong).sum()),repaired=int((ev&oldwrong&~wrong).sum()),identity_reachable=int((ev&np.isin(qid,gid[active])).sum()),reference20_reachable=int((ev&(geo[:,active].min(1)<=20)).sum())))
    for m,(conf,elig) in zip(c['methods'],ss):
     fitted=fit(conf[ca],elig[ca],(~wrong)[ca],qid[ca],8);key=(fold['name'],m)
     if case['arm']=='full':fixed[key]=fitted;fullfits[name,fold['name'],m]=fitted
     thresholds.append(dict(condition=name,fold=fold['name'],case_id=case['case_id'],method=m,fit=fitted))
     for regime in c['regimes']:
      a=np.zeros(len(q),bool);tau=None;budget=None;possible=int((ev&elig).sum())
      if regime.startswith('budget'):
       budget=int(n*int(regime[6:])/100);feasible=possible>=budget
       if feasible:
        ix=np.flatnonzero(ev&elig);a[ix[np.argsort(-conf[ix],kind='stable')[:budget]]]=True
      else:
       frozen_condition='old_same' if name=='old_to_plain' else name
       ft=fitted if regime=='matched' else fullfits[frozen_condition,fold['name'],m];feasible=ft is not None
       if feasible:tau=ft['threshold'];a=ev&elig&(conf>=tau)
      an=int(a.sum());err=int((a&wrong).sum());new=int((a&created).sum());ni=len(set(qid[a]));r={k:v for k,v in case.items() if k!='removed'}|dict(condition=name,fold=fold['name'],method=m,regime=regime,queries=n,budget=budget,eligible=possible,feasible=feasible,threshold=tau,answered=an,errors=err,correct=an-err,risk=err/an if an else None,created_errors=new,inherited_errors=err-new,accepted_ids=ni,empirical_useful10=bool(an and 10*err<=an and ni>=8),prediction_row=pi)
      for t in [20,50,100]:r[f'reference_errors_gt{t}m']=int((a&(meters>t)).sum())
      if condition['history']==condition['current'] and m.endswith('guard'):assert new==0
      rows.append(r);accepted.append(a)
  print(json.dumps(dict(condition=name,rows=len(rows),seconds=round(time.perf_counter()-start,1))),flush=True)
 summary=[]
 for condition in c['conditions']:
  for fold in c['folds']:
   for arm,fraction in [('target',.25),('target',.5),('control',.25)]:
    for m in c['methods']:
     for regime in c['regimes']:
      key=dict(condition=condition['name'],fold=fold['name'],arm=arm,fraction=fraction,method=m,regime=regime);rr=[r for r in rows if all(r[k]==v for k,v in key.items())];assert len(rr)==20;ok=[r for r in rr if r['feasible']];used=ok if regime.startswith('budget') else rr;summary.append(key|dict(feasible_masks=len(ok),useful10_masks=sum(r['empirical_useful10'] for r in rr),**{'mean_'+k:float(np.mean([r[k] for r in used])) if used else None for k in ['answered','errors','correct','created_errors','inherited_errors','reference_errors_gt20m','reference_errors_gt50m','reference_errors_gt100m']}))
 p=c['primary'];sel=lambda m:next(r for r in summary if r['method']==m and all(r[k]==p[k] for k in ['condition','fold','arm','fraction','regime']));base=sel(p['base']);hist=sel(p['historical']);delta=base['mean_errors']-hist['mean_errors'] if base['mean_errors'] is not None and hist['mean_errors'] is not None else None;rel=delta/base['mean_errors'] if delta is not None and base['mean_errors']>0 else None;passed=base['feasible_masks']==hist['feasible_masks']==20 and delta is not None and delta>=1 and rel is not None and rel>=.1
 for n,d in [('metrics',rows),('summary',summary),('thresholds',thresholds),('raw_errors',raw),('full_accuracy',full),('decision',dict(status='UTILITY_RULE_PASS' if passed else 'UTILITY_RULE_FAIL',base=base,historical=hist,absolute_reduction=delta,relative_reduction=rel,original_SUES_failure_overturned=False))]:save(O/(n+'.json'),d)
 np.savez_compressed(O/'decisions.npz',accepted=np.stack(accepted),winner=np.stack(pred));save(O/'evaluation.json',dict(status='EXECUTED_PENDING_REVIEW',rows=len(rows),seconds=time.perf_counter()-start,output_sha256={p.name:sha(p) for p in O.iterdir() if p.name in ['metrics.json','summary.json','thresholds.json','raw_errors.json','full_accuracy.json','decision.json','decisions.npz']},features_sha256=sha(O/'features.npz')))
if __name__=='__main__':main()
