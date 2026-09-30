"""Isolated-history access benchmark. Worker has no labels or previous-score input
unless explicitly running the cached-history mode. Review happens separately.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import csv
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

R=Path(__file__).resolve().parents[1]
C=R/'configs/A-HISTORY-ACCESS-001.json'
O=R/'results/A-HISTORY-ACCESS-001'
P=R/'data/history_access_001'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    x=Counters();x.cb=ctypes.sizeof(x)
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    fn=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    fn.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    assert fn(kernel.GetCurrentProcess(),ctypes.byref(x),x.cb)
    return dict(working_set_bytes=x.WorkingSetSize,process_peak_working_set_bytes=x.PeakWorkingSetSize)
def su(values):
    v=np.asarray(values,np.float64)
    assert len(v)==10 and np.isfinite(v).all() and v[0]>1e-8
    return float(-.5*(v[1:].mean()+np.median(v))/v[0])
def retrieval(q,g,ids,active,mode,history=None):
    scores=g@q
    rank=np.argsort(-scores,kind='stable')
    if mode=='archive_recompute':
        old=int(rank[0]);old_su=su(scores[rank[:10]])
        top=rank[active[rank]][:10]
        winner=int(top[0]);guard=bool(active[old])
    else:
        top=rank[:10];winner=int(ids[top[0]])
        old,old_su,guard=None,None,None
        if mode=='cached_history':
            old,old_su=history
            guard=bool(active[old])
    return dict(winner=winner,current_su=su(scores[top]),historical_winner=old,historical_su=old_su,guard=guard,
                top10_scores=scores[top].astype(float).tolist())

def freeze():
    from vild_common import check_freeze,load_layout
    check_freeze();c=read(C)
    assert not (O/'freeze.json').exists()
    records=[];inputs={C.relative_to(R).as_posix():sha(C),Path(__file__).relative_to(R).as_posix():sha(Path(__file__))}
    for ds in c['datasets']:
        layout=load_layout(ds);nq=len(layout['qpath'])
        ev=np.flatnonzero(layout['qrole']=='evaluation')
        chosen=ev[np.linspace(0,len(ev)-1,c['queries_per_dataset'],dtype=int)]
        base=R/'data/vild_protocol_001';src=base/f'{ds}_features.npy'
        inf=read(R/f'results/A-VILD-VISUAL-001/{ds}/inference.json')
        assert sha(src)==inf['features_sha256']
        feat=np.load(src,mmap_mode='r');d=P/ds;d.mkdir(parents=True,exist_ok=True)
        np.save(d/'query.npy',feat[chosen]);np.save(d/'archive.npy',feat[nq:])
        dead=np.unpackbits(layout['deleted_packed'],axis=1,count=len(layout['gpath'])).astype(bool)
        cases=read(R/f'results/A-VILD-VISUAL-001/{ds}_cases.json')
        for ci in c['case_indices']:
            ids=np.flatnonzero(~dead[ci]).astype(np.int32)
            np.save(d/f'active_{ci}.npy',feat[nq+ids]);np.save(d/f'ids_{ci}.npy',ids)
            np.save(d/f'alive_{ci}.npy',~dead[ci])
        with (base/f'{ds}_input_images.csv').open(encoding='utf8',newline='') as f:manifest=list(csv.DictReader(f))
        save(d/'images.json',[manifest[int(i)] for i in chosen[:c['end_to_end_images_per_dataset']]])
        # Cache is packaged for the cache-only worker; archive worker cannot open it.
        with np.load(base/f'{ds}_historical_cache.npz',allow_pickle=False) as h:
            save(d/'history_cache.json',{str(k):[int(h['winner'][i]),float(h['su'][i])] for k,i in enumerate(chosen)})
        rec=dict(dataset=ds,source_query_indices=chosen.tolist(),queries=len(chosen),gallery=len(layout['gpath']),cases=[dict(index=ci,**cases[ci]) for ci in c['case_indices']])
        save(d/'selection.json',rec);records.append(rec)
        for p in d.iterdir():inputs[p.relative_to(R).as_posix()]=sha(p)
        for p in [src,base/f'{ds}.npz',base/f'{ds}_historical_cache.npz',R/f'results/A-VILD-VISUAL-001/{ds}/review.json']:
            inputs[p.relative_to(R).as_posix()]=sha(p)
    save(O/'freeze.json',dict(status='FIXED_BEFORE_DEPLOYMENT_TIMING',post_result=True,datasets=records,input_sha256=inputs,config=read(C)))
    print('FROZEN',flush=True)

def worker(ds,ci,mode,image_mode=False):
    c=read(C);d=P/ds
    # Log all opened project data and deny access outside the explicit input set.
    allowed={'query.npy',f'alive_{ci}.npy'}
    allowed |= {'archive.npy'} if mode=='archive_recompute' else {f'active_{ci}.npy',f'ids_{ci}.npy'}
    if mode=='cached_history':allowed.add('history_cache.json')
    image_files=set()
    if image_mode:
        images=read(d/'images.json');allowed.add('images.json')
        image_files={str((R/r['path']).resolve()).casefold() for r in images}
    opened=set()
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if str(p).casefold().startswith(str(R/'data').casefold()+os.sep):
            if not ((p.parent==d and p.name in allowed) or str(p).casefold() in image_files):
                raise RuntimeError('Forbidden data access: '+str(p))
            opened.add(p.relative_to(R).as_posix())
    sys.addaudithook(hook)
    started=time.perf_counter();before=memory()
    q=np.load(d/'query.npy');active=np.load(d/f'alive_{ci}.npy')
    ids=None
    if mode=='archive_recompute':g=np.load(d/'archive.npy')
    else:g=np.load(d/f'active_{ci}.npy');ids=np.load(d/f'ids_{ci}.npy')
    cache=read(d/'history_cache.json') if mode=='cached_history' else None
    if cache is not None:assert 'unseen_query_key' not in cache
    load_seconds=time.perf_counter()-started
    loaded=memory()
    samples=[];predictions=[]
    encoder=None;torch=None
    if image_mode:
        import torch
        from camp_model import build,descriptor,tensor
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        encoder,load_record=build();encoder.cuda();torch.cuda.reset_peak_memory_stats()
    def call(k):
        hist=cache[str(k)] if cache is not None else None
        if hist is not None:hist=(int(hist[0]),float(hist[1]))
        if image_mode:
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.inference_mode():
                x=tensor(images[k])[None].cuda();f=descriptor(encoder,x).cpu().numpy()[0]
            torch.cuda.synchronize();encoded=time.perf_counter()
            out=retrieval(f,g,ids,active,mode,hist);end=time.perf_counter()
            return out,dict(total_ms=(end-begin)*1000,encode_decode_transfer_ms=(encoded-begin)*1000,retrieval_ms=(end-encoded)*1000,feature_max_abs=float(abs(f-q[k]).max()))
        begin=time.perf_counter();out=retrieval(q[k],g,ids,active,mode,hist)
        return out,dict(total_ms=(time.perf_counter()-begin)*1000)
    count=len(images) if image_mode else len(q)
    for k in range(c['warmup_queries_per_case']):call(k%count)
    for repeat in range(c['timed_repeats']):
        for k in range(count):
            out,t=call(k);samples.append(dict(repeat=repeat,query_index=k,**t))
            if repeat==0:predictions.append(out)
            else:assert out==predictions[k]
    peak=memory()
    if image_mode:assert torch.cuda.max_memory_allocated()/1024**2<c['gpu_peak_limit_mib']
    result=dict(dataset=ds,case_index=ci,mode=mode,image_mode=image_mode,status='EXECUTED_PENDING_REVIEW',
        samples=samples,predictions=predictions,opened_data=sorted(opened),input_load_seconds=load_seconds,
        memory_before=before,memory_loaded=loaded,memory_end=peak,
        array_bytes=dict(gallery=g.nbytes,query_slice=q.nbytes,active_bitmap=active.nbytes,active_ids=0 if ids is None else ids.nbytes),
        cache_serialized_bytes=0 if cache is None else (d/'history_cache.json').stat().st_size,
        gpu_peak_allocated_mib=None if not image_mode else torch.cuda.max_memory_allocated()/1024**2,
        environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),blas_threads=1))
    name=f'{ds}_{ci}_{mode}'+('_images' if image_mode else '')
    save(O/'runs'/(name+'.json'),result)
    print(name,'DONE',flush=True)

def run():
    f=read(O/'freeze.json');c=read(C)
    for p,h in f['input_sha256'].items():assert sha(R/p)==h,p
    # Freeze/config/source must be committed before timing.
    git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)]
    for p in [C,Path(__file__),O/'freeze.json']:
        subprocess.run([*git,'ls-files','--error-unmatch',p.relative_to(R).as_posix()],check=True,capture_output=True)
        subprocess.run([*git,'diff','HEAD','--exit-code','--',p.relative_to(R).as_posix()],check=True,capture_output=True)
    for ds in c['datasets']:
        for ci in c['case_indices']:
            for mode in c['modes']:
                dest=O/'runs'/f'{ds}_{ci}_{mode}.json'
                assert not dest.exists(),'Do not overwrite timing observations'
                subprocess.run([sys.executable,__file__,'worker','--dataset',ds,'--case',str(ci),'--mode',mode],check=True)
        subprocess.run([sys.executable,__file__,'worker','--dataset',ds,'--case',str(c['end_to_end_case_index']),'--mode','archive_recompute','--images'],check=True)

def review():
    f=read(O/'freeze.json');c=read(C);rows=[];checks=0;disagreements=[]
    for p,h in f['input_sha256'].items():assert sha(R/p)==h,p
    for rec in f['datasets']:
        ds=rec['dataset'];idx=rec['source_query_indices'];base=R/'data/vild_protocol_001'
        winners=np.load(base/f'{ds}_winners.npy',mmap_mode='r');tops=np.load(base/f'{ds}_top10.npy',mmap_mode='r')
        for p in sorted((O/'runs').glob(ds+'_*.json')):
            z=read(p);assert z['dataset']==ds;ci=z['case_index'];mode=z['mode']
            assert len(z['samples'])==len(z['predictions'])*c['timed_repeats']
            active=np.load(P/ds/f'alive_{ci}.npy')
            if mode!='cached_history':assert not any('cache' in x or 'winner' in x for x in z['opened_data'])
            for k,out in enumerate(z['predictions']):
                i=idx[k];assert active[out['winner']]
                err=float(np.max(abs(np.array(out['top10_scores'])-tops[ci,i])))
                assert err<c['numeric_atol']
                assert abs(out['current_su']-su(tops[ci,i]))<c['numeric_atol']
                if out['winner']!=int(winners[ci,i]):disagreements.append(dict(dataset=ds,case_index=ci,mode=mode,image_mode=z['image_mode'],query_index=k,type='current_winner'))
                if mode!='active_only':
                    assert abs(out['historical_su']-su(tops[0,i]))<c['numeric_atol']
                    assert out['guard']==bool(active[out['historical_winner']])
                    if out['historical_winner']!=int(winners[0,i]):disagreements.append(dict(dataset=ds,case_index=ci,mode=mode,image_mode=z['image_mode'],query_index=k,type='historical_winner'))
                    if out['guard']:assert out['winner']==out['historical_winner']
                checks+=1
            ms=[x['total_ms'] for x in z['samples']]
            if z['image_mode']:assert max(x['feature_max_abs'] for x in z['samples'])<c['numeric_atol']
            rows.append(dict(dataset=ds,case_index=ci,mode=mode,image_mode=z['image_mode'],queries=len(z['predictions']),observations=len(ms),
                median_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),min_ms=min(ms),max_ms=max(ms),
                gallery_bytes=z['array_bytes']['gallery'],array_payload_bytes=sum(z['array_bytes'].values()),
                process_peak_working_set_bytes=z['memory_end']['process_peak_working_set_bytes'],gpu_peak_allocated_mib=z['gpu_peak_allocated_mib'],
                encode_median_ms=None if not z['image_mode'] else float(np.median([x['encode_decode_transfer_ms'] for x in z['samples']]))))
    assert len(rows)==len(c['datasets'])*(len(c['case_indices'])*len(c['modes'])+1)
    save(O/'summary.json',rows)
    save(O/'review.json',dict(status='PASS_DEPLOYMENT_ACCESS_AND_NUMERIC_REPLAY' if not disagreements else 'MODIFY_RANK_DISAGREEMENTS',checks=checks,winner_disagreements=disagreements,
        scientific_scope='Engineering access/cost replay only, not new accuracy evidence, arbitrary future-query performance or risk guarantee',
        freeze_sha256=sha(O/'freeze.json'),summary_sha256=sha(O/'summary.json'),runs_sha256={p.name:sha(p) for p in sorted((O/'runs').glob('*.json'))}))
    print('REVIEW',checks,'rank disagreements',len(disagreements),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','run','worker','review']);a.add_argument('--dataset');a.add_argument('--case',type=int);a.add_argument('--mode');a.add_argument('--images',action='store_true');args=a.parse_args()
    if args.command=='worker':worker(args.dataset,args.case,args.mode,args.images)
    else:globals()[args.command]()
