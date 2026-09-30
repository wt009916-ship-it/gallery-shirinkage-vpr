"""Metadata-only protocol. Never reads image descriptors or accuracy."""
import hashlib,json,datetime,zipfile,re
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
R=Path(__file__).resolve().parents[1]; I=R/'results/DENSEUAV-INTAKE-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf8')
def distance(a,b):
 a=np.radians(a);b=np.radians(b);dl=a[:,None,0]-b[None,:,0];dt=a[:,None,1]-b[None,:,1]
 return 6371008.8*2*np.arcsin(np.sqrt(np.clip(np.sin(dt/2)**2+np.cos(a[:,None,1])*np.cos(b[None,:,1])*np.sin(dl/2)**2,0,1)))
def main():
 out=R/'configs/A-DENSEUAV-EXTERNAL-001.json';assert not out.exists()
 coords={}
 for s in (I/'metadata/DenseUAV/Dense_GPS_ALL.txt').read_text().splitlines():
  p,e,n,*_=s.split();k=p.split('/')[-2];v=[float(e[1:]),float(n[1:])]
  if k in coords:assert coords[k]==v
  coords[k]=v
 index=json.loads((I/'archive_index.json').read_text());gallery=sorted({x['name'].split('/')[-2] for x in index if '/test/gallery_satellite/' in x['name'] and x['bytes']});query=sorted({x['name'].split('/')[-2] for x in index if '/test/query_drone/' in x['name'] and x['bytes']})
 assert len(gallery)==3033 and len(query)==777 and set(query)<=set(gallery)<=set(coords)
 xy=np.array([coords[k] for k in query]);dm=distance(xy,xy);ng,groups=connected_components(dm<=500,directed=False);assert ng==2
 # Larger western component is calibration, distant eastern component evaluation;
 # reverse fold is reported as well. This is two regions, not 777 independent sites.
 order=sorted(range(ng),key=lambda g:float(xy[groups==g,0].mean()));west=[query[i] for i in np.flatnonzero(groups==order[0])];east=[query[i] for i in np.flatnonzero(groups==order[1])]
 assert len(west)==577 and len(east)==200
 gap=float(distance(np.array([coords[x] for x in west]),np.array([coords[x] for x in east])).min());assert gap>2000
 # Inspect all published U1652 coordinates, without selecting/excluding by results.
 u=[]
 with zipfile.ZipFile(R/'results/U1652-INTAKE-001/official_coordinates.kml') as z:
  for name in z.namelist():
   if name.endswith('.kml'):
    for s in re.findall(rb'<coordinates>(.*?)</coordinates>',z.read(name),re.S):
     for v in s.decode('ascii').split():
      a=v.split(',')
      if len(a)>=2:u.append([float(a[0]),float(a[1])])
 ud=distance(xy,np.array(u)).min(1) if u else None
 selected=[x for x in index if x['bytes'] and (x['name'].startswith('DenseUAV/test/query_drone/') and x['name'].endswith('/H90.JPG') or x['name'].startswith('DenseUAV/test/gallery_satellite/') and x['name'].split('/')[-1] in ['H90.tif','H90_old.tif'])]
 assert len(selected)==6843
 cases=[dict(case_id='full',arm='full',fraction=0.,seed=None,removed=[])]
 nonquery=sorted(set(gallery)-set(query))
 for seed in range(300930,300950):
  rng=np.random.default_rng(seed);w=rng.permutation(west);e=rng.permutation(east)
  for f in [.25,.5]:
   removed=sorted(list(w[:int(len(w)*f)])+list(e[:int(len(e)*f)]));cases.append(dict(case_id=f'target_{int(f*100)}_{seed}',arm='target',fraction=f,seed=seed,removed=removed))
  removed=sorted(rng.choice(nonquery,int(len(w)*.25)+int(len(e)*.25),replace=False));cases.append(dict(case_id=f'control_25_{seed}',arm='control',fraction=.25,seed=seed,removed=removed))
 c=dict(exp_id='A-DENSEUAV-EXTERNAL-001',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='Metadata only; all H90 queries, all gallery identities, both filename variants. No model scores observed.',source=dict(repo='Dmmm997/DenseUAV',revision=json.loads((I/'hf_info.json').read_text())['sha'],archive='DenseUAV.zip',archive_bytes=16738945339,lfs_sha256='21d3f592d8102be8cabafaaf98d94994b6d58fb8c35a3b6dcae0ce4d100c3cb8',index_sha256=sha(I/'archive_index.json')),budget=dict(compressed_member_bytes=sum(x['compressed_bytes'] for x in selected),download_ceiling_bytes=2900000000,max_workers=4,gpu_minutes_estimate=[3,10],batch_size=4,precision='FP32',peak_allocated_limit_mib=4096,new_training=False),members=selected,coords=coords,query_ids=query,gallery_ids=gallery,folds=[dict(name='west_to_east',calibration=west,evaluation=east),dict(name='east_to_west',calibration=east,evaluation=west)],spatial_audit=dict(min_fold_separation_m=gap,coordinate_reference='published nominal coordinates; no independently verified physical truth',query_components_100m=int(connected_components(dm<=100,directed=False)[0]),u1652_coordinate_count=len(u),u1652_nearest_min_m=float(ud.min()) if ud is not None else None,u1652_queries_within_500m=int((ud<=500).sum()) if ud is not None else None,pretraining_overlap_certified=False),conditions=[dict(name='plain_same',history='plain',current='plain'),dict(name='old_same',history='old',current='old'),dict(name='old_to_plain',history='old',current='plain')],conditions_scope='Paper confirms 2020/2022 satellite and 2021 UAV. Per-file year mapping not separately verified: retain old/plain labels. Not query-weather experiment.',model='Frozen official University1652 CAMP checkpoint and existing camp_model.py; no adaptation',methods=['current_su','current_su_guard','old_su','old_su_guard','current_margin','current_margin_guard','old_margin_guard','sq8_margin_guard'],regimes=['budget10','budget25','budget50','full_frozen','matched'],min_calibration_accepted_ids=8,alpha=.1,cases=cases,primary=dict(condition='plain_same',fold='west_to_east',arm='target',fraction=.25,regime='budget50',base='current_su_guard',historical='old_su_guard',utility_rule='all 20 masks feasible, at least 1 mean fewer error and 10% relative decrease; internal utility threshold, not universal academic threshold',primary_label='identity',independent_confirmation=False),secondary=dict(reference_distance_thresholds_m=[20,50,100],rule='Re-score exactly the same predictions and accepted queries; no geographic reranking or tuned thresholds; adjacent locations may remain legal after identity deletion.'),limitations=['H90 slice, not full DenseUAV benchmark','Two spatial components, masks are repeated interventions not independent new sites','Satellite crops overlap; identity errors do not imply physical localization failure','Old-to-plain changes gallery descriptors: deleted-winner guard no longer mathematically ensures zero new error','University-pretrained encoder can have geographic overlap; quantify nearest published coordinates but do not certify absent training overlap'],input_sha256={p:sha(R/p) for p in ['experiments/camp_model.py','results/CAMP-INTAKE-001/checkpoint.json','manifests/CAMP_SOURCE.json','experiments/evaluate_camp.py','experiments/run_update_memory.py']})
 c['threshold_transfer_rule']='full_frozen: unchanged gallery uses its own full predeletion calibration; old_to_plain uses old_same full calibration without any plain calibration. matched: fit current condition and deletion using disjoint-region calibration. No evaluation labels in either.'
 save(out,c);save(I/'geometry_and_budget.json',{k:c[k] for k in ['frozen_utc','budget','spatial_audit','limitations']});print(json.dumps({k:c[k] for k in ['budget','spatial_audit']}))
if __name__=='__main__':main()
