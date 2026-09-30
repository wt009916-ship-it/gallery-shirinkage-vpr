"""Posthoc eight-cell trajectory accounting on unchanged accepted sets.

This is ordered descriptive decomposition, not a causal attribution theorem.
"""
import json,csv,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/A-DENSEUAV-TRANSITIONS-001'
def main():
 O.mkdir(exist_ok=True);out=[];nchecks=0;hashes={}
 for exp in ['A-DENSEUAV-EXTERNAL-001','A-DENSEUAV-VITS-001']:
  p=R/'results'/exp;assert json.loads((p/'review.json').read_text())['status']=='PASS_INDEPENDENT_RECONSTRUCTION';c=json.loads((R/'configs'/f'{exp}.json').read_text());z=np.load(p/'features.npz');d=np.load(p/'decisions.npz');rows=json.loads((p/'metrics.json').read_text());gid=z['gallery_id'];qid=z['query_id'];nc=len(c['cases']);plain=d['winner'][0];old=d['winner'][nc];e0=gid[old]!=qid;ev=gid[plain]!=qid
  for i,r in enumerate(rows):
   if r['condition']!='old_to_plain' or r['arm']!='target' or r['fraction']!=.25 or r['regime']!='budget50':continue
   a=d['accepted'][i];ed=gid[d['winner'][r['prediction_row']]]!=qid;v={}
   for x in [0,1]:
    for y in [0,1]:
     for w in [0,1]:v[f't{x}{y}{w}']=int((a&(e0==x)&(ev==y)&(ed==w)).sum())
   assert sum(v.values())==r['answered'];assert sum(v[k] for k in v if k[-1]=='1')==r['errors'];assert v['t001']+v['t011']==r['created_errors'];nchecks+=3
   out.append(dict(experiment=exp,fold=r['fold'],method=r['method'],case_id=r['case_id'],feasible=r['feasible'],answered=r['answered'],errors=r['errors'],new_vs_old=r['created_errors'],version_created_persisting=v['t011'],deletion_created_after_version_correct=v['t001'],inherited_never_repaired=v['t111'],recreated_after_version_repair=v['t101'],**v))
  for f in ['metrics.json','decisions.npz','review.json']:hashes[f'{exp}/{f}']=hashlib.sha256((p/f).read_bytes()).hexdigest()
 with (O/'trajectories.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 sums=[]
 for exp in ['A-DENSEUAV-EXTERNAL-001','A-DENSEUAV-VITS-001']:
  for fold in ['west_to_east','east_to_west']:
   for m in sorted({r['method'] for r in out}):
    rr=[r for r in out if r['experiment']==exp and r['fold']==fold and r['method']==m];assert len(rr)==20;ok=[r for r in rr if r['feasible']];sums.append(dict(experiment=exp,fold=fold,method=m,feasible_masks=len(ok),**{k:float(np.mean([r[k] for r in ok])) if ok else None for k in out[0] if k not in ['experiment','fold','method','case_id','feasible']}))
 (O/'summary.json').write_text(json.dumps(sums,indent=2));(O/'review.json').write_text(json.dumps(dict(status='PASS_EXHAUSTIVE_ACCOUNTING',checks=nchecks,rows=len(out),posthoc=True,source_inputs=hashes,meaning='tXYZ: wrongness before replacement / after replacement without deletion / after replacement plus deletion. Ordered algebra, not unique causal attribution. No predictions/acceptance changed.'),indent=2));print(json.dumps(dict(rows=len(out),checks=nchecks)))
if __name__=='__main__':main()
