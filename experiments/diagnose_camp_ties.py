"""Post hoc analytic tie envelope; no score tuning or new efficacy test.

For every feasible Stage14 row, retain all eligible scores above the budget
boundary and choose the remaining r of m exactly tied queries. Report minimum,
maximum and uniform-tie expected errors. No approximate ties, random seeds,
changed predictions, labels or budgets. Results are descriptive reused-data
diagnostics. Compare predeclared pairs on jointly feasible masks only.
"""
import csv,hashlib,itertools,json,subprocess,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-SELECTION-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def dump(name,rows):
 (O/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
 with (O/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 assert read('results/A-CAMP-SELECTION-001/review.json')['status']=='PASS'
 c=read('configs/A-CAMP-SELECTION-001.json');orig=read('configs/A-CAMP-COHORTS-001.json');metrics=read('results/A-CAMP-SELECTION-001/metrics.json')
 inputs={f'results/A-CAMP-SELECTION-001/{p}':h for p,h in read('results/A-CAMP-SELECTION-001/execution.json')['output_sha256'].items()}
 inputs['experiments/diagnose_camp_ties.py']=sha('experiments/diagnose_camp_ties.py')
 for p,h in inputs.items():assert sha(p)==h
 git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)]
 subprocess.run([*git,'diff','HEAD','--exit-code','--','experiments/diagnose_camp_ties.py'],check=True,capture_output=True)
 with (O/'ties_STARTED.json').open('x') as f:json.dump({'epoch':time.time(),'commit':subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip(),'inputs':inputs,'classification':'POSTHOC_DIAGNOSTIC_NO_NEW_CONFIRMATION'},f,indent=2)
 # Exhaustive small mathematical tests, separate from actual data evidence.
 checks=0
 for m in range(1,9):
  for w in range(m+1):
   labels=[1]*w+[0]*(m-w)
   for take in range(m+1):
    vals=[sum(labels[i] for i in comb) for comb in itertools.combinations(range(m),take)]
    assert min(vals)==max(0,take-(m-w)) and max(vals)==min(take,w)
    assert abs(sum(vals)/len(vals)-take*w/m)<1e-12;checks+=1
 rows=[]
 for ds in c['datasets']:
  dc=orig['datasets'][ds]
  with np.load(R/dc['features']) as z:gid=z['gallery_id'];heights=z['query_height'][z['query_role']=='evaluation'] if ds=='sues' else None
  with np.load(O/ds/'decisions.npz') as z:d={k:z[k] for k in z.files}
  ci={case['case_id']:i for i,case in enumerate(dc['cases'])};pi={p['name']:i for i,p in enumerate(c['policies'])}
  for i,x in enumerate([x for x in metrics if x['dataset']==ds]):
   if not x['feasible']:continue
   j=ci[x['case_id']];score=d['scores'][j,c['scores'].index(x['score'])];unit=np.ones(len(score),bool) if x['scope']=='pooled' else heights==int(x['scope']);eligible=unit&d['eligible'][j,pi[x['policy']]];wrong=gid[d['winner'][j]]!=d['query_identity'];budget=x['budget'];boundary=np.sort(score[eligible])[-budget];above=eligible&(score>boundary);tied=eligible&(score==boundary);take=budget-int(above.sum());m=int(tied.sum());w=int((wrong&tied).sum());fixed=int((wrong&above).sum());lo=fixed+max(0,take-(m-w));hi=fixed+min(take,w);expected=fixed+take*w/m
   # Independent label-best/worst sorting replays the two extremal choices.
   candidates=np.flatnonzero(eligible)
   best=sorted(candidates,key=lambda q:(-float(score[q]),bool(wrong[q]),int(q)))[:budget]
   worst=sorted(candidates,key=lambda q:(-float(score[q]),not bool(wrong[q]),int(q)))[:budget]
   assert int(wrong[best].sum())==lo and int(wrong[worst].sum())==hi
   assert lo<=x['errors']<=hi and lo<=expected<=hi
   rows.append({k:x[k] for k in ['dataset','case_id','arm','fraction','scope','answer_fraction','policy','budget','errors']}|dict(boundary_ties=m,take_from_tie=take,tied_wrong=w,min_errors=lo,max_errors=hi,expected_errors=expected));checks+=3
 summary=[];contrasts=[]
 keys=['dataset','arm','fraction','scope','answer_fraction','policy'];groups={tuple(x[k] for k in keys) for x in rows if x['arm']!='full'}
 for key in sorted(groups):
  a=[x for x in rows if tuple(x[k] for k in keys)==key]
  summary.append(dict(zip(keys,key))|{'feasible_masks':len(a)}|{'mean_'+k:sum(x[k] for x in a)/len(a) for k in ['errors','min_errors','max_errors','expected_errors','boundary_ties','take_from_tie']})
 index={(x['dataset'],x['case_id'],x['scope'],x['answer_fraction'],x['policy']):x for x in rows}
 for x in read('results/A-CAMP-SELECTION-001/contrasts.json'):
  candidates=[a for a in rows if all(a[k]==x[k] for k in keys[:-1]) and a['policy']==x['base']];pairs=[(a,index.get((a['dataset'],a['case_id'],a['scope'],a['answer_fraction'],x['test']))) for a in candidates];pairs=[(a,b) for a,b in pairs if b is not None];assert len(pairs)==x['joint_masks']
  out={k:x[k] for k in ['dataset','arm','fraction','scope','answer_fraction','base','test','joint_masks']}
  for name,ak,bk in [('expected_reduction','expected_errors','expected_errors'),('min_reduction','min_errors','max_errors'),('max_reduction','max_errors','min_errors')]:out[name]=sum(a[ak]-b[bk] for a,b in pairs)/len(pairs) if pairs else None
  contrasts.append(out)
 for name,obj in [('ties_metrics',rows),('ties_summary',summary),('ties_contrasts',contrasts)]:dump(name,obj)
 result={'status':'PASS_ANALYTIC_ENVELOPE_AND_EXTREMAL_REPLAY','rows':len(rows),'checks':checks,'scope':'Post hoc exact-score tie envelopes, not confidence intervals or independent trials','outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('ties_*.*') if p.name!='ties_review.json'}}
 (O/'ties_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='outputs'}))
if __name__=='__main__':main()
