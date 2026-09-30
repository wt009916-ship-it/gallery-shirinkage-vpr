"""Frozen CAMP inference on the complete predeclared H90 slice."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,time,hashlib,subprocess,collections
from pathlib import Path
import numpy as np,torch
from denseuav_vits_model import build,tensor,descriptor
R=Path(__file__).resolve().parents[1];O=R/'results/A-DENSEUAV-VITS-001';CP='configs/A-DENSEUAV-VITS-001.json'
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 c=json.loads((R/CP).read_text());O.mkdir(exist_ok=True);d=json.loads((R/'results/A-DENSEUAV-EXTERNAL-001/download.json').read_text());assert d['status']=='PASS_DOWNLOAD';assert sorted(x['member'] for x in d['files'])==sorted(x['name'] for x in c['members'])
 for p,h in c['input_sha256'].items():assert sha(p)==h,p
 git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)];paths=[CP,'experiments/infer_denseuav_vits.py','experiments/evaluate_denseuav_vits.py','experiments/review_denseuav_vits.py']
 for p in paths:
  subprocess.run([*git,'ls-files','--error-unmatch',p],check=True,capture_output=True);subprocess.run([*git,'diff','HEAD','--exit-code','--',p],check=True,capture_output=True)
 lookup={x['member']:x for x in d['files']};items=[lookup[f'DenseUAV/test/query_drone/{i}/H90.JPG'] for i in c['query_ids']]
 for suffix in ['H90.tif','H90_old.tif']:items += [lookup[f'DenseUAV/test/gallery_satellite/{i}/{suffix}'] for i in c['gallery_ids']]
 assert len(items)==6843
 pixels=collections.defaultdict(list)
 for r in items:pixels[r['pixel_sha256']].append(r['member'])
 duplicates=[v for v in pixels.values() if len(v)>1];cross=[v for v in duplicates if len({s.split('/')[-2] for s in v})>1]
 save(O/'image_quality.json',dict(images=len(items),duplicate_pixel_groups=duplicates,cross_identity_duplicate_groups=cross,policy='No outcome-dependent exclusions; duplicate groups exposed. Stop if any query/calibration identity has exact duplicate pixels under different IDs.'))
 assert not any(any('/query_drone/' in p for p in v) for v in cross),'Cross-identity query duplicate: amendment required before scoring'
 with (O/'INFERENCE_STARTED.json').open('x') as f:json.dump(dict(epoch=time.time(),commit=subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip(),input_sha256={p:sha(p) for p in paths},budget=c['budget']),f,indent=2)
 torch.set_num_threads(4);torch.manual_seed(300930);torch.cuda.manual_seed_all(300930);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();model,load=build();model.cuda();values=[]
 with torch.inference_mode():
  for i in range(0,len(items),4):
   x=torch.stack([tensor(r) for r in items[i:i+4]]).cuda();v=descriptor(model,x);assert torch.isfinite(v).all() and v.shape==(len(x),512);
   if i==0:assert torch.equal(model.raw(x),model.manual_raw(x))
   values.append(v.cpu().numpy());assert torch.cuda.max_memory_allocated()/1024**2<4096
   if i%400==0:print(json.dumps(dict(images=i+len(x),total=len(items),seconds=round(time.perf_counter()-start,1))),flush=True)
  f=np.concatenate(values);repeat=descriptor(model,tensor(items[0])[None].cuda()).cpu().numpy()[0];delta=float(abs(repeat-f[0]).max());assert delta<1e-5 and abs(np.linalg.norm(f,axis=1)-1).max()<1e-5
 np.savez_compressed(O/'features.npz',query_id=np.array(c['query_ids']),gallery_id=np.array(c['gallery_ids']),query_f=f[:777],gallery_plain=f[777:3810],gallery_old=f[3810:]);save(O/'inference.json',dict(status='PASS_FROZEN_INFERENCE',images=len(items),seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,repeat_max_abs=delta,load=load,features_sha256=sha('results/A-DENSEUAV-VITS-001/features.npz'),download_sha256=sha('results/A-DENSEUAV-EXTERNAL-001/download.json'),input_sha256={p:sha(p) for p in paths}));print('PASS_FROZEN_INFERENCE',flush=True)
if __name__=='__main__':main()
