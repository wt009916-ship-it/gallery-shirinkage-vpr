import csv,hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent
m=json.loads((R/'ARTIFACT_MANIFEST.json').read_text());checks=0
for name,r in m['files'].items():
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==r['sha256'],name
 checks+=1
for p in (R/'tables').rglob('*.csv'):
 for r in csv.DictReader(p.open(encoding='utf-8-sig',newline='')):
  for keys in [('answered','correct','errors'),('mean_answered','mean_correct','mean_errors')]:
   if all(r.get(k,'') not in ('','None','nan') for k in keys):
    a,b,c=map(lambda k:float(r[k]),keys);assert math.isclose(a,b+c,rel_tol=1e-9,abs_tol=1e-7),(p.name,r);checks+=1
  if all(r.get(k,'') not in ('','None','nan') for k in ['min_errors','errors','max_errors']):
   assert float(r['min_errors'])<=float(r['errors'])<=float(r['max_errors']),(p.name,r);checks+=1
print(json.dumps({'status':'PASS','checks':checks,'files':len(m['files']),'scope':'hashes, count arithmetic and tie bounds only'}))
