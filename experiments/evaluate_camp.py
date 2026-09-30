"""Fixed CAMP robustness evaluation; no encoder or score selection."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from run_update_memory import encode
from audit_u1652_fair import serialize
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-ROBUSTNESS-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def fit(conf,eligible,correct,ids,minids):
    order=np.flatnonzero(eligible);order=order[np.argsort(-conf[order],kind='stable')];seen=set();err=0;out=None
    for n,j in enumerate(order,1):
        seen.add(ids[j]);err+=int(not correct[j])
        if n<len(order) and conf[order[n]]==conf[j]:continue
        if len(seen)>=minids and 10*err<=n:out={'threshold':float(conf[j]),'answered':n,'errors':err,'accepted_ids':len(seen)}
    return out
def su_margin(v):
    v=v.astype(float);assert np.isfinite(v).all() and v.shape[1]==10 and v[:,0].min()>1e-8,'SU undefined for nonpositive first score; do not shift scores'
    return -.5*(v[:,1:10].mean(1)+np.median(v,axis=1))/v[:,0],v[:,0]-v[:,1]
def main():
    c=read('configs/A-CAMP-ROBUSTNESS-001.json');inf=read('results/A-CAMP-ROBUSTNESS-001/inference.json')
    for p,h in inf['input_sha256'].items():assert sha(p)==h
    with (O/'EVALUATION_STARTED.json').open('x') as f:json.dump({'epoch':time.time(),'config_sha256':sha('configs/A-CAMP-ROBUSTNESS-001.json')},f)
    allout=[]
    for name,dc in c['datasets'].items():
        start=time.perf_counter();target=O/name;assert sha(f'results/A-CAMP-ROBUSTNESS-001/{name}/features.npz')==next(x['features_sha256'] for x in inf['datasets'] if x['dataset']==name)
        with np.load(target/'features.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        g=d['gallery_f'];gid=d['gallery_id'];co,sc,dec,res=encode(g,8);enc={8:(co,sc,dec,res)};cases=dc['cases'];methods=c['methods']
        def context(role):
            ix=d['query_role']==role;q=d['query_f'][ix];s=q@g.T;rank=np.argsort(-s,axis=1,kind='stable');su,margin=su_margin(np.take_along_axis(s,rank[:,:10],axis=1));upper=(q@dec.T).astype(float)+np.linalg.norm(q.astype(float),axis=1)[:,None]*res[None,:]+1e-6;assert (upper>=s).all()
            return dict(ids=d['query_identity'][ix],scope=d['query_height'][ix] if name=='sues' else d['query_block'][ix],s=s,old=rank[:,0],oldsu=su,oldmargin=margin,upper=upper)
        def specs(ctx,case):
            active=np.flatnonzero(~np.isin(gid,case['removed']));dead=np.flatnonzero(np.isin(gid,case['removed']));rank=active[np.argsort(-ctx['s'][:,active],axis=1,kind='stable')];win=rank[:,0];v=np.take_along_axis(ctx['s'],rank[:,:10],axis=1);su,margin=su_margin(v);guard=np.isin(ctx['old'],active);assert np.array_equal(win[guard],ctx['old'][guard]);upper=ctx['upper'][:,dead].max(1) if len(dead) else np.full(len(win),-np.inf);sqvalid=v[:,0]>upper;assert guard[sqvalid].all();allok=np.ones(len(win),bool)
            scores=[(su,allok),(su,guard),(ctx['oldsu'],allok),(ctx['oldsu'],guard),(margin,allok),(margin,guard),(ctx['oldmargin'],guard),(v[:,0]-np.maximum(v[:,1],upper),sqvalid)]
            return win,scores
        def subset(ctx,scope):return np.ones(len(ctx['ids']),bool) if scope=='pooled' else ctx['scope']==int(scope)
        cal=context('calibration');fits=[]
        for case in cases:
            win,ss=specs(cal,case);correct=gid[win]==cal['ids']
            for scope in dc['scopes']:
                ix=subset(cal,scope)
                for m,(conf,elig) in zip(methods,ss):fits.append(dict(case_id=case['case_id'],scope=scope,method=m,fit=fit(conf[ix],elig[ix],correct[ix],cal['ids'][ix],dc['min_calibration_accepted_ids'])))
        save(target/'thresholds.json',fits);fhash=sha(f'results/A-CAMP-ROBUSTNESS-001/{name}/thresholds.json');fm={(r['case_id'],r['scope'],r['method']):r['fit'] for r in fits}
        ev=context('evaluation');ids=ev['ids'];assert len(ids)==dc['expected_eval_queries'];oldwrong=gid[ev['old']]!=ids;uniq=sorted(set(ids));units=np.array([ids==u for u in uniq],np.int32);rows=[];answers=[];pred=[];costs=[];accuracy=[];raw=[]
        with (target/'identity_metrics.csv').open('w',newline='',encoding='utf8') as f:
            iw=csv.writer(f);iw.writerow(['row','identity','queries','answered','correct','errors','created_errors'])
            for ci,case in enumerate(cases):
                win,ss=specs(ev,case);pred.append(win);wrong=gid[win]!=ids;created=wrong&~oldwrong;inherited=wrong&oldwrong;repaired=~wrong&oldwrong;active=np.flatnonzero(~np.isin(gid,case['removed']));dead=np.flatnonzero(np.isin(gid,case['removed']));raw.append(dict(case_id=case['case_id'],wrong=int(wrong.sum()),created=int(created.sum()),inherited=int(inherited.sum()),repaired=int(repaired.sum()),oldwrong=int(oldwrong.sum())))
                for m in ['current_frozen','history_fp32','history_sq8_bound']:costs.append(dict(case_id=case['case_id'],method=m,serialized_bytes=len(serialize(g,gid,active,dead,m,enc))))
                for scope in dc['scopes']:
                    ix=subset(ev,scope);n=int(ix.sum())
                    if ci==0:accuracy.append(dict(scope=scope,queries=n,correct=int((ix&~wrong).sum()),recall1=float((~wrong)[ix].mean())))
                    for m,(conf,eligible) in zip(methods,ss):
                        for regime in c['regimes']:
                            a=np.zeros(len(ids),bool);budget=None;tau=None;possible=int((ix&eligible).sum())
                            if regime.startswith('budget'):
                                budget=int(n*int(regime[6:])/100);feasible=possible>=budget
                                if feasible:
                                    cand=np.flatnonzero(ix&eligible);a[cand[np.argsort(-conf[cand],kind='stable')[:budget]]]=True
                            else:
                                fitted=fm[case['case_id'] if regime=='matched' else 'full',scope,m];feasible=fitted is not None
                                if feasible:tau=fitted['threshold'];a=ix&eligible&(conf>=tau)
                            an=int(a.sum());err=int((a&wrong).sum());new=int((a&created).sum());ni=len(set(ids[a]));row={k:v for k,v in case.items() if k!='removed'}|dict(scope=scope,method=m,regime=regime,queries=n,budget=budget,eligible=possible,feasible=feasible,threshold=tau,answered=an,errors=err,correct=an-err,risk=err/an if an else None,created_errors=new,inherited_errors=err-new,accepted_ids=ni,empirical_useful10=bool(an and 10*err<=an and ni>=dc['min_calibration_accepted_ids']))
                            if m.endswith('guard'):assert new==0
                            rows.append(row);answers.append(a)
                            if scope=='pooled':
                                totals=units@np.stack([np.ones(len(ids),bool),a,a&~wrong,a&wrong,a&created],1).astype(np.int32)
                                for u,v in zip(uniq,totals):iw.writerow([len(rows)-1,u,*v.tolist()])
                if ci%20==0:print(json.dumps({'dataset':name,'cases':ci+1,'seconds':round(time.perf_counter()-start,1)}),flush=True)
        summary=[]
        for arm,frac in [('target',.25),('control',.25),('target',.5)]:
            for scope in dc['scopes']:
                for m in methods:
                    for regime in c['regimes']:
                        key=dict(arm=arm,fraction=frac,scope=scope,method=m,regime=regime);rr=[r for r in rows if all(r[k]==v for k,v in key.items())];assert len(rr)==20;ok=[r for r in rr if r['feasible']];chosen=ok if regime.startswith('budget') else rr;risks=[r['risk'] for r in rr if r['risk'] is not None];summary.append(key|dict(feasible_masks=len(ok),useful10_masks=sum(r['empirical_useful10'] for r in rr),nonempty_masks=len(risks),**{'mean_'+k:float(np.mean([r[k] for r in chosen])) if chosen else None for k in ['answered','errors','correct','created_errors','inherited_errors']},mean_risk=float(np.mean(risks)) if risks else None))
        p=c['primary'];select=lambda m:next(x for x in summary if all(x[k]==v for k,v in p.items() if k in ['arm','fraction','scope','regime']) and x['method']==m)
        base,hist=select(p['base']),select(p['historical']);delta=base['mean_errors']-hist['mean_errors'] if base['mean_errors'] is not None and hist['mean_errors'] is not None else None;rel=delta/base['mean_errors'] if delta is not None and base['mean_errors']>0 else None;passed=base['feasible_masks']==hist['feasible_masks']==20 and rel is not None and rel>=.1 and delta>=1
        decision={'status':'DESCRIPTIVE_COMPARISON_RULE_PASS' if passed else 'DESCRIPTIVE_COMPARISON_RULE_FAIL','base':base,'historical':hist,'absolute_reduction':delta,'relative_reduction':rel,'independent_confirmation':False,'not_overturning_previous_gate':True}
        for fname,obj in [('metrics',rows),('summary',summary),('full_accuracy',accuracy),('index_costs',costs),('raw_errors',raw),('decision',decision)]:save(target/(fname+'.json'),obj)
        np.savez_compressed(target/'decisions.npz',accepted=np.stack(answers),winner=np.stack(pred));assert sha(f'results/A-CAMP-ROBUSTNESS-001/{name}/thresholds.json')==fhash
        out=dict(dataset=name,rows=len(rows),thresholds=len(fits),runtime_seconds=time.perf_counter()-start,output_sha256={x.name:sha(x.relative_to(R)) for x in target.iterdir() if x.is_file() and x.name!='features.npz'});allout.append(out);print(json.dumps({'dataset':name,'decision':decision,'full_accuracy':accuracy}))
    save(O/'evaluation.json',{'status':'EXECUTED_PENDING_REVIEW','datasets':allout,'input_sha256':{p:sha(p) for p in ['configs/A-CAMP-ROBUSTNESS-001.json','experiments/evaluate_camp.py','experiments/run_update_memory.py','experiments/audit_u1652_fair.py','results/A-CAMP-ROBUSTNESS-001/inference.json']}})
if __name__=='__main__':main()
