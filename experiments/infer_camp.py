"""Frozen official-CAMP global-descriptor follow-up on the existing cohorts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,subprocess,time
from pathlib import Path
import numpy as np,torch
from camp_model import build,tensor,descriptor
R=Path(__file__).resolve().parents[1];O=R/'results/A-CAMP-ROBUSTNESS-001'
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def main():
 c=read('configs/A-CAMP-ROBUSTNESS-001.json');O.mkdir(parents=True,exist_ok=True);git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)]
 paths=['configs/A-CAMP-ROBUSTNESS-001.json','experiments/infer_camp.py','experiments/evaluate_camp.py','experiments/review_camp.py','experiments/camp_model.py']
 for p in paths:
  subprocess.run([*git,'ls-files','--error-unmatch',p],check=True,capture_output=True);subprocess.run([*git,'diff','HEAD','--exit-code','--',p],check=True,capture_output=True)
 for p,h in c['input_sha256'].items():assert sha(p)==h,p
 with (O/'INFERENCE_STARTED.json').open('x') as f:json.dump({'epoch':time.time(),'commit':subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip(),'budget':c['budget'],'inputs':{p:sha(p) for p in paths}},f,indent=2)
 torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats();began=time.perf_counter();model,load=build();model.cuda();audits=[];compat=[]
 for name,dc in c['datasets'].items():
  target=O/name;target.mkdir(exist_ok=True);intake=read(dc['intake']);lookup={x['member']:x for x in intake['files']};gallery={x['identity']:x for x in intake['files'] if x['view']=='satellite' and (name=='sues' or x['role']=='gallery')}
  with np.load(R/dc['template']) as z:meta={k:z[k] for k in z.files if k not in ['query_f','gallery_f']}
  items=[lookup[str(q)] for q in meta['query_id']]+[gallery[str(g)] for g in meta['gallery_id']];assert len(items)==dc['expected_images'];start=time.perf_counter();values=[]
  with torch.inference_mode():
   for i in range(0,len(items),c['budget']['batch_size']):
    x=torch.stack([tensor(r) for r in items[i:i+c['budget']['batch_size']]]).cuda();f=descriptor(model,x);assert f.shape==(len(x),1024) and torch.isfinite(f).all();values.append(f.cpu().numpy());assert torch.cuda.max_memory_allocated()/1024**2<c['budget']['peak_allocated_limit_mib']
    if i==0:
     direct=torch.nn.functional.normalize(model.model_1.convnext(x)[0],dim=-1);assert torch.equal(f,direct)
     for scalar in [-7.,5.]:
      model.model_1.pos_scale=scalar;assert torch.equal(f,descriptor(model,x))
     model.model_1.pos_scale=.6;compat.append({'dataset':name,'images':len(x),'full_model_vs_backbone_exact':True,'unused_scalar_perturbations':[-7.,5.],'descriptor_invariant_exact':True})
    if i%400==0:print(json.dumps({'dataset':name,'images':i+len(x),'total':len(items),'seconds':round(time.perf_counter()-start,1)}),flush=True)
   features=np.concatenate(values);rep=descriptor(model,tensor(items[0])[None].cuda()).cpu().numpy()[0];delta=float(abs(rep-features[0]).max());assert delta<1e-5 and abs(np.linalg.norm(features,axis=1)-1).max()<1e-5
  nq=len(meta['query_id']);np.savez_compressed(target/'features.npz',**meta,query_f=features[:nq],gallery_f=features[nq:]);audits.append({'dataset':name,'images':len(items),'query_count':nq,'gallery_count':len(items)-nq,'runtime_seconds':time.perf_counter()-start,'repeat_max_abs':delta,'features_sha256':sha(f'results/A-CAMP-ROBUSTNESS-001/{name}/features.npz'),'queries_and_gallery_order_equal_to_template':True})
 result={'status':'PASS_SOURCE_COMPATIBLE_CAMP_INFERENCE','datasets':audits,'load':load,'compatibility':compat,'total_seconds':time.perf_counter()-began,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'new_training':False,'pretraining_overlap_known':False,'precision_difference_from_author':'FP32 without autocast; no flip; author gap descriptor and resize/normalization preserved','input_sha256':{p:sha(p) for p in paths}};save(O/'inference.json',result);print(json.dumps(result))
if __name__=='__main__':main()
