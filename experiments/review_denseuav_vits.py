"""Read-only reconstruction of predictions, policy selections and counts."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/A-DENSEUAV-VITS-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 start=time.perf_counter();c=read(R/'configs/A-DENSEUAV-VITS-001.json');report=read(O/'evaluation.json');checks=0
 for p,h in report['output_sha256'].items():assert sha(O/p)==h;checks+=1
 assert sha(O/'features.npz')==report['features_sha256'];z=np.load(O/'features.npz');q=z['query_f'];qid=z['query_id'];gid=z['gallery_id'];d=np.load(O/'decisions.npz');rows=read(O/'metrics.json');fits=read(O/'thresholds.json');fm={(r['condition'],r['fold'],r['case_id'],r['method']):r['fit'] for r in fits};assert len(rows)==14640 and d['accepted'].shape==(14640,777) and d['winner'].shape==(183,777)
 # Direct unit-vector spherical distances, independently from evaluator haversine.
 xy=np.radians(np.array([c['coords'][str(k)] for k in gid]));qx=np.radians(np.array([c['coords'][str(k)] for k in qid]));vec=lambda x:np.column_stack((np.cos(x[:,1])*np.cos(x[:,0]),np.cos(x[:,1])*np.sin(x[:,0]),np.sin(x[:,1])));dist=np.arccos(np.clip(vec(qx)@vec(xy).T,-1,1))*6371008.8
 # Encoder reproduced here from the documented per-vector symmetric SQ8 rule.
 from run_update_memory import encode
 scores={v:q@z['gallery_'+v].T for v in ['plain','old']};cache={};pi=0
 for cond in c['conditions']:
  name=cond['name'];h=scores[cond['history']];s=scores[cond['current']];hr=np.argsort(-h,axis=1,kind='stable');hw=hr[:,0];hv=np.take_along_axis(h,hr[:,:10],1).astype(float);hs=-.5*(hv[:,1:].mean(1)+np.median(hv,axis=1))/hv[:,0];hm=hv[:,0]-hv[:,1]
  # Quantization is a shared library, but assert the stored reconstruction bound
  # directly against every true dot product, not against method names.
  _,_,de,res=encode(z['gallery_'+cond['history']],8);up=(q@de.T).astype(float)+np.linalg.norm(q.astype(float),axis=1)[:,None]*res[None,:]+1e-6;assert np.all(up>=h)
  for case in c['cases']:
   ix=np.where(~np.isin(gid,case['removed']))[0];dead=np.where(np.isin(gid,case['removed']))[0];r=ix[np.argsort(-s[:,ix],axis=1,kind='stable')];w=r[:,0];assert np.array_equal(w,d['winner'][pi]);pi+=1;v=np.take_along_axis(s,r[:,:10],1).astype(float);cs=-.5*(np.mean(v[:,1:],axis=1)+np.median(v,axis=1))/v[:,0];cm=v[:,0]-v[:,1];guard=~np.isin(gid[hw],case['removed']);bound=np.max(up[:,dead],axis=1) if len(dead) else np.full(777,-np.inf);sq=v[:,0]>bound;allok=np.ones(777,bool);spec=[(cs,allok),(cs,guard),(hs,allok),(hs,guard),(cm,allok),(cm,guard),(hm,guard),(v[:,0]-np.maximum(v[:,1],bound),sq)];cache[name,case['case_id']]=(w,hw,spec);checks+=1
 for f in fits:
  fold=next(x for x in c['folds'] if x['name']==f['fold']);ca=np.isin(qid,fold['calibration']);w,_,sp=cache[f['condition'],f['case_id']];score,elig=sp[c['methods'].index(f['method'])];ix=np.flatnonzero(ca&elig);ix=ix[np.argsort(-score[ix],kind='stable')];err=np.cumsum(gid[w[ix]]!=qid[ix]);n=np.arange(1,len(ix)+1);ends=np.r_[score[ix][:-1]!=score[ix][1:],True] if len(ix) else np.zeros(0,bool);valid=np.flatnonzero((err*10<=n)&(n>=8)&ends)
  # One H90 query per ID means accepted ID count equals prefix length.
  if len(valid):
   k=valid[-1];expected=dict(threshold=float(score[ix[k]]),answered=int(n[k]),errors=int(err[k]),accepted_ids=int(n[k]));assert f['fit']==expected
  else:assert f['fit'] is None
  checks+=1
 for j,r in enumerate(rows):
  fold=next(x for x in c['folds'] if x['name']==r['fold']);ev=np.isin(qid,fold['evaluation']);w,hw,sp=cache[r['condition'],r['case_id']];conf,elig=sp[c['methods'].index(r['method'])];a=np.zeros(777,bool)
  if r['regime'].startswith('budget'):
   n=int(ev.sum()*int(r['regime'][6:])/100);ix=np.flatnonzero(ev&elig);ok=len(ix)>=n
   if ok:a[ix[np.argsort(-conf[ix],kind='stable')[:n]]]=True
  else:
   fc='old_same' if r['condition']=='old_to_plain' and r['regime']=='full_frozen' else r['condition']
   fitted=fm[fc,r['fold'],r['case_id'] if r['regime']=='matched' else 'full',r['method']];ok=fitted is not None
   if ok:a=ev&elig&(conf>=fitted['threshold'])
  assert r['feasible']==ok and np.array_equal(a,d['accepted'][j]);wrong=gid[w]!=qid;created=wrong&(gid[hw]==qid);assert r['answered']==int(a.sum()) and r['errors']==int((a&wrong).sum()) and r['created_errors']==int((a&created).sum());assert r['inherited_errors']==r['errors']-r['created_errors'];assert not (a&~ev).any()
  for t in [20,50,100]:assert r[f'reference_errors_gt{t}m']==int((a&(dist[np.arange(777),w]>t)).sum())
  checks+=9
 for s in read(O/'summary.json'):
  key={k:s[k] for k in ['condition','fold','arm','fraction','method','regime']};rr=[r for r in rows if all(r[k]==v for k,v in key.items())];assert len(rr)==20;ok=[r for r in rr if r['feasible']];assert s['feasible_masks']==len(ok);used=ok if s['regime'].startswith('budget') else rr
  for k,v in s.items():
   if k.startswith('mean_'):
    e=float(np.mean([r[k[5:]] for r in used])) if used else None;assert v==e;checks+=1
 result=dict(status='PASS_INDEPENDENT_RECONSTRUCTION',checks=checks,rows=len(rows),thresholds=len(fits),seconds=time.perf_counter()-start,limitations=['Shared frozen encoder and SQ8 library; independent ranking/selection/count reconstruction, not independent implementation of neural network','Spherical reference-coordinate distances only; no certified physical GPS accuracy'],input_sha256={p:sha(O/p) for p in ['features.npz','metrics.json','decisions.npz','evaluation.json']});(O/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
