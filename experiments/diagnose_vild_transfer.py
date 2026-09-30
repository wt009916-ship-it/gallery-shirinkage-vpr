"""Post-result engineering diagnosis; no tuning or replacement of frozen outcomes."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from camp_model import build, descriptor, tensor
from vild_common import R, O, P, config, sha, save, check_freeze, load_layout


def main():
    start=time.perf_counter()
    check_freeze()
    target=R/'results/A-VILD-TRANSFER-DIAGNOSTIC-001'
    target.mkdir(exist_ok=True)
    assert not (target/'diagnostic.json').exists()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    model, loaded=build()
    model.cuda()
    rows=[]
    for dataset in config()['datasets']:
        d=load_layout(dataset)
        nq=len(d['qpath'])
        paths=np.concatenate([d['qpath'],d['gpath']])
        f=np.load(P/f'{dataset}_features.npy',mmap_mode='r')
        inf=json.loads((O/dataset/'inference.json').read_text())
        assert sha(P/f'{dataset}_features.npy')==inf['features_sha256']
        with (P/f'{dataset}_input_images.csv').open(newline='',encoding='utf8') as h:
            manifest=list(csv.DictReader(h))
        assert [x['path'] for x in manifest]==paths.tolist()
        # Fixed evenly spaced images, chosen without score/error inspection.
        selected=np.unique(np.r_[np.linspace(0,nq-1,8,dtype=int),np.linspace(nq,len(paths)-1,8,dtype=int)])
        with torch.inference_mode():
            z=descriptor(model,torch.stack([tensor(manifest[i]) for i in selected]).cuda()).cpu().numpy()
        delta=float(np.max(abs(z-f[selected])))
        assert delta<1e-5
        norms=np.linalg.norm(f,axis=1)
        assert np.isfinite(f).all() and float(abs(norms-1).max())<1e-5
        csvpath=R/f'data/vild_full/ViLD_dataset/{dataset}/{dataset}_coordinates__test.csv'
        with csvpath.open(newline='',encoding='utf-8-sig') as h:
            metadata=list(csv.DictReader(h))
        references={x['filename']:x for x in metadata if x['image_type']=='reference'}
        testmask=np.array([Path(p).name in references for p in d['gpath']])
        gi=np.flatnonzero(testmask)
        for j in gi:
            r=references[Path(d['gpath'][j]).name]
            assert np.array_equal(d['gxy'][j],np.array([float(r['utm_east']),float(r['utm_north'])]))
        queries={x['filename']:x for x in metadata if x['image_type']=='query'}
        ids=np.flatnonzero(d['qrole']=='evaluation')
        for j in ids:
            r=queries[Path(d['qpath'][j]).name]
            assert np.array_equal(d['qxy'][j],np.array([float(r['utm_east']),float(r['utm_north'])]))
        prior=np.load(P/f'{dataset}_winners.npy',mmap_mode='r')[0]
        winner=np.empty(nq,np.int32)
        checks=0
        # Recompute the same 64-row blocks as the main run, then mask val references.
        for i in range(0,nq,64):
            scores=f[i:min(i+64,nq)]@f[nq:].T
            full=np.argmax(scores,axis=1)
            assert np.array_equal(full,prior[i:i+len(full)])
            winner[i:i+len(full)]=gi[np.argmax(scores[:,gi],axis=1)]
            checks+=len(full)
        raw=[]
        for name,indices,pred in [('shared_val_test',np.arange(len(d['gpath'])),prior),('test_only',gi,winner)]:
            distance=np.linalg.norm(d['qxy'][ids]-d['gxy'][pred[ids]],axis=1)
            tree=cKDTree(d['gxy'][indices])
            for radius in [25,50,100,150]:
                count=tree.query_ball_point(d['qxy'][ids],radius,return_length=True)
                raw.append(dict(gallery=name,entries=len(indices),queries=len(ids),radius_m=radius,
                    correct=int((distance<=radius).sum()),recall1=float((distance<=radius).mean()),
                    covered_queries=int((count>0).sum()),uniform_random_expected_recall=float(np.mean(count/len(indices)))))
        rows.append(dict(dataset=dataset,status='PASS_SAMPLED_FEATURES_ALL_ORDER_AND_FULL_WINNERS',
            sampled_feature_rows=selected.tolist(),feature_repeat_max_abs=delta,full_winner_checks=checks,
            published_test_coordinate_checks=len(ids)+len(gi),scores=raw,
            input_sha256={f'data/vild_protocol_001/{dataset}_features.npy':inf['features_sha256'],
                csvpath.relative_to(R).as_posix():sha(csvpath)}))
        print(json.dumps(rows[-1]),flush=True)
    save(target/'diagnostic.json',dict(status='PASS_ENGINEERING_DIAGNOSTIC_NOT_MODEL_VALIDATION',
        post_result=True,no_threshold_fitting=True,no_original_protocol_changed=True,
        scope='Same buffered queries; test-only gallery sensitivity is not exact author evaluation. 16 repeated images per flight do not audit every forward pass.',
        checkpoint=loaded,results=rows,seconds=time.perf_counter()-start,script_sha256=sha(Path(__file__))))


if __name__=='__main__':
    main()
