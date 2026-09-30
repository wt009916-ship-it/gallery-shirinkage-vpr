"""Fixed metadata-only deletion feasibility, with matched entry counts."""
import json,csv,hashlib,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1];O=R/'results/A-VILD-COVERAGE-001';D=R/'data/vild_metadata/ViLD_dataset'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=True);cp=R/'configs/A-VILD-COVERAGE-001.json';c=json.loads(cp.read_text());protocol={'config_sha256':sha(cp),'script_sha256':sha(Path(__file__)),'started_epoch':time.time(),'model_scored':False};(O/'STARTED.json').open('x').write(json.dumps(protocol));records=[];summaries=[];hashes={};checks=0
 for folder in ['vild','vild_09']:
  p=D/folder/f'{folder}_coordinates__test.csv';rr=list(csv.DictReader(p.open()));hashes[p.relative_to(R).as_posix()]=sha(p);q=np.array([[float(r['utm_east']),float(r['utm_north'])] for r in rr if r['image_type']=='query']);g=np.array([[float(r['utm_east']),float(r['utm_north'])] for r in rr if r['image_type']=='reference']);tree=cKDTree(g);radii=sorted(set([c['primary_radius_m']]+c['secondary_radius_m']));neighbours={t:tree.query_ball_point(q,t) for t in radii}
  cells,inverse=np.unique(np.floor(g/c['spatial_cell_m']).astype(np.int64),axis=0,return_inverse=True);sizes=np.bincount(inverse)
  for fraction in c['target_entry_fractions']:
   for seed in range(c['seed_start'],c['seed_start']+c['seeds']):
    rng=np.random.default_rng(seed);order=rng.permutation(len(cells));cut=np.searchsorted(np.cumsum(sizes[order]),int(np.ceil(fraction*len(g))))+1;dead_spatial=np.isin(inverse,order[:cut]);k=int(dead_spatial.sum());dead_random=np.zeros(len(g),bool);dead_random[rng.choice(len(g),size=k,replace=False)]=True;assert dead_random.sum()==dead_spatial.sum();checks+=1
    for name,dead in [('spatial',dead_spatial),('matched_random',dead_random)]:
     for radius,near in neighbours.items():
      missing=np.array([dead[ix].all() for ix in near]);active_tree=cKDTree(g[~dead]);independent=active_tree.query(q)[0]>radius;assert np.array_equal(missing,independent);checks+=1
      records.append(dict(dataset=folder,target_fraction=fraction,seed=seed,arm=name,radius_m=radius,query_count=len(q),gallery_count=len(g),removed_entries=k,actual_entry_fraction=k/len(g),removed_cells=cut if name=='spatial' else None,uncovered_queries=int(missing.sum()),uncovered_fraction=float(missing.mean())))
   for name in ['spatial','matched_random']:
    for radius in radii:
     rows=[r for r in records if r['dataset']==folder and r['target_fraction']==fraction and r['arm']==name and r['radius_m']==radius];assert len(rows)==c['seeds'];summaries.append(dict(dataset=folder,target_fraction=fraction,arm=name,radius_m=radius,masks=len(rows),query_count=len(q),gallery_count=len(g),mean_removed_entries=float(np.mean([r['removed_entries'] for r in rows])),mean_uncovered=float(np.mean([r['uncovered_queries'] for r in rows])),min_uncovered=min(r['uncovered_queries'] for r in rows),max_uncovered=max(r['uncovered_queries'] for r in rows)))
  print(json.dumps({'completed':folder,'rows':len(records)}),flush=True)
 with (O/'metrics.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 (O/'summary.json').write_text(json.dumps(summaries,indent=2));(O/'review.json').write_text(json.dumps(dict(status='PASS_TWO_GEOMETRY_COMPUTATIONS',checks=checks,rows=len(records),source_sha256=hashes,config_sha256=protocol['config_sha256'],script_sha256=protocol['script_sha256'],model_scored=False,limits='Same reference coordinates and masks; second calculation validates radius coverage implementation, not dataset truth or scientific novelty.'),indent=2));print(json.dumps({'checks':checks,'rows':len(records)}))
if __name__=='__main__':main()
