"""Independent ascending-score formula, Python stable selection and count replay."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-ROBUSTNESS-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
    start=time.perf_counter();c=read('configs/A-CAMP-ROBUSTNESS-001.json');inf=read('results/A-CAMP-ROBUSTNESS-001/inference.json');ex=read('results/A-CAMP-ROBUSTNESS-001/evaluation.json');checks=0;reports=[]
    for obj in [c,inf,ex]:
        for p,h in obj['input_sha256'].items():assert sha(p)==h,p;checks+=1
    for name,dc in c['datasets'].items():
        target=O/name
        for p,h in next(x['output_sha256'] for x in ex['datasets'] if x['dataset']==name).items():assert sha(target.relative_to(R)/p)==h;checks+=1
        with np.load(target/'features.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
        with np.load(R/dc['template'],allow_pickle=False) as z:
            for k in d:
                if k not in ['query_f','gallery_f']:assert np.array_equal(d[k],z[k])
        with np.load(target/'decisions.npz',allow_pickle=False) as z:answers=z['accepted'];pred=z['winner']
        for f in [d['query_f'],d['gallery_f']]:assert f.shape[1]==1024 and np.isfinite(f).all() and abs(np.linalg.norm(f,axis=1)-1).max()<1e-5
        g=d['gallery_f'];gid=d['gallery_id'];scale=np.max(abs(g),axis=1).astype(np.float32)/np.float32(127);co=np.rint(g/scale[:,None]).clip(-127,127).astype(np.int8);decoded=co.astype(np.float32)*scale[:,None];res=np.nextafter(np.linalg.norm(g.astype(float)-decoded.astype(float),axis=1).astype(np.float32),np.float32(np.inf));contexts={};cache={}
        def su(v):return -(v[:,:9].sum(1)/(9*v[:,9])+(v[:,4]+v[:,5])/(2*v[:,9]))/2
        for role in ['calibration','evaluation']:
            ix=d['query_role']==role;q=d['query_f'][ix];s=q@g.T;old=s.argmax(1);v=np.sort(s,axis=1)[:,-10:].astype(float);ctx=dict(ids=d['query_identity'][ix],scope=d['query_height'][ix] if name=='sues' else d['query_block'][ix],s=s,old=old,oldsu=su(v),oldmargin=v[:,9]-v[:,8]);contexts[role]=ctx;upper=(q@decoded.T).astype(float)+np.linalg.norm(q.astype(float),axis=1)[:,None]*res[None,:]+1e-6;assert (upper>=s).all()
            for ci,case in enumerate(dc['cases']):
                active=np.flatnonzero(~np.isin(gid,case['removed']));dead=np.flatnonzero(np.isin(gid,case['removed']));win=active[s[:,active].argmax(1)];v=np.sort(s[:,active],axis=1)[:,-10:].astype(float);guard=np.isin(old,active);allok=np.ones(len(win),bool);u=upper[:,dead].max(1) if len(dead) else np.full(len(win),-np.inf);ss=[(su(v),allok),(su(v),guard),(ctx['oldsu'],allok),(ctx['oldsu'],guard),(v[:,9]-v[:,8],allok),(v[:,9]-v[:,8],guard),(ctx['oldmargin'],guard),(v[:,9]-np.maximum(v[:,8],u),v[:,9]>u)]
                cache[role,case['case_id']]=(win,dict(zip(c['methods'],ss)))
                if role=='evaluation':assert np.array_equal(pred[ci],win)
        def scope(ctx,name):return np.ones(len(ctx['ids']),bool) if name=='pooled' else ctx['scope']==int(name)
        fits=read(target.relative_to(R)/'thresholds.json');fm={};ctx=contexts['calibration']
        for r in fits:
            win,ss=cache['calibration',r['case_id']];conf,eligible=ss[r['method']];ix=scope(ctx,r['scope']);order=sorted(np.flatnonzero(ix&eligible),key=lambda j:(-float(conf[j]),int(j)));seen=set();err=0;best=None
            for n,j in enumerate(order,1):
                seen.add(ctx['ids'][j]);err+=int(gid[win[j]]!=ctx['ids'][j])
                if n<len(order) and conf[order[n]]==conf[j]:continue
                if len(seen)>=dc['min_calibration_accepted_ids'] and err*10<=n:best=(float(conf[j]),n,err,len(seen))
            fit=r['fit'];assert (best is None)==(fit is None)
            if best is not None:assert abs(best[0]-fit['threshold'])<2e-14 and best[1:]==(fit['answered'],fit['errors'],fit['accepted_ids'])
            fm[r['case_id'],r['scope'],r['method']]=fit
        rows=read(target.relative_to(R)/'metrics.json');ctx=contexts['evaluation'];ids=ctx['ids'];oldwrong=gid[ctx['old']]!=ids;cases={r['case_id']:r for r in dc['cases']}
        for j,r in enumerate(rows):
            win,ss=cache['evaluation',r['case_id']];conf,eligible=ss[r['method']];ix=scope(ctx,r['scope']);a=np.zeros(len(ids),bool);possible=int((eligible&ix).sum());assert possible==r['eligible']
            if r['regime'].startswith('budget'):
                n=int(ix.sum()*int(r['regime'][6:])/100);feasible=possible>=n;assert n==r['budget']
                if feasible:a[sorted(np.flatnonzero(eligible&ix),key=lambda k:(-float(conf[k]),int(k)))[:n]]=True
            else:
                fit=fm[r['case_id'] if r['regime']=='matched' else 'full',r['scope'],r['method']];feasible=fit is not None
                if feasible:
                    tau=fit['threshold'];a=eligible&ix&(conf>=tau);boundary=eligible&ix&(abs(conf-tau)<2e-14)
                    if boundary.any() and 'su' in r['method']:
                        active=np.arange(len(g)) if r['method'].startswith('old_') else np.flatnonzero(~np.isin(gid,cases[r['case_id']]['removed']));rank=active[np.argsort(-ctx['s'][:,active],axis=1,kind='stable')];v=np.take_along_axis(ctx['s'],rank[:,:10],axis=1).astype(float);exact=-.5*(v[:,1:].mean(1)+np.median(v,axis=1))/v[:,0];a[boundary]=exact[boundary]>=tau
            assert feasible==r['feasible'] and np.array_equal(a,answers[j]),(name,j)
            wrong=gid[win]!=ids;an=int(a.sum());err=int((a&wrong).sum());new=int((a&wrong&~oldwrong).sum());ni=len(set(ids[a]));assert (an,err,new,err-new,an-err,ni)==(r['answered'],r['errors'],r['created_errors'],r['inherited_errors'],r['correct'],r['accepted_ids']);assert r['risk']==(err/an if an else None);assert r['empirical_useful10']==bool(an and 10*err<=an and ni>=dc['min_calibration_accepted_ids'])
        identityrows=0;groupcounts={}
        with (target/'identity_metrics.csv').open(encoding='utf8',newline='') as f:
            for r in csv.DictReader(f):
                identityrows+=1;j=int(r['row']);unit=ids==r['identity'];a=answers[j]&unit;win=cache['evaluation',rows[j]['case_id']][0];wrong=gid[win]!=ids;assert [int(r[k]) for k in ['queries','answered','correct','errors','created_errors']]==[int(unit.sum()),int(a.sum()),int((a&~wrong).sum()),int((a&wrong).sum()),int((a&wrong&~oldwrong).sum())];groupcounts[j]=groupcounts.get(j,0)+1
        assert all(v==len(set(ids)) for v in groupcounts.values())
        for r in read(target.relative_to(R)/'summary.json'):
            rr=[x for x in rows if all(x[k]==r[k] for k in ['arm','fraction','scope','method','regime'])];assert len(rr)==20;ok=[x for x in rr if x['feasible']];chosen=ok if r['regime'].startswith('budget') else rr;assert len(ok)==r['feasible_masks'];assert r['useful10_masks']==sum(x['empirical_useful10'] for x in rr)
            for k in ['answered','errors','correct','created_errors','inherited_errors']:assert r['mean_'+k]==(sum(x[k] for x in chosen)/len(chosen) if chosen else None)
        for r in read(target.relative_to(R)/'raw_errors.json'):assert r['wrong']==r['inherited']+r['created'] and r['wrong']-r['oldwrong']==r['created']-r['repaired']
        for r in read(target.relative_to(R)/'index_costs.json'):
            nd=len(cases[r['case_id']]['removed']);expected=24+(len(g)-nd)*(1024*4+4)+(nd*(1024*4+4) if r['method']=='history_fp32' else nd*(1024+12) if r['method']=='history_sq8_bound' else 0);assert expected==r['serialized_bytes']
        decision=read(target.relative_to(R)/'decision.json');b,h=decision['base'],decision['historical'];delta=b['mean_errors']-h['mean_errors'] if b['mean_errors'] is not None and h['mean_errors'] is not None else None;rel=delta/b['mean_errors'] if delta is not None and b['mean_errors']>0 else None;passed=b['feasible_masks']==h['feasible_masks']==20 and rel is not None and rel>=.1 and delta>=1;assert decision['status']==('DESCRIPTIVE_COMPARISON_RULE_PASS' if passed else 'DESCRIPTIVE_COMPARISON_RULE_FAIL')
        reports.append(dict(dataset=name,rows=len(rows),fits=len(fits),identity_rows=identityrows,decision_slots=int(answers.size)))
    result={'status':'PASS','runtime_seconds':time.perf_counter()-start,'hash_checks':checks,'datasets':reports,'method':'separate ascending SU, Python stable sorting, calibration prefixes, exact full decision replay, identity contribution counts, serialized size arithmetic','script_sha256':sha('experiments/review_camp.py'),'not_verified':['novelty','geographic independence','pretraining membership','population risk guarantees']};(O/'review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
