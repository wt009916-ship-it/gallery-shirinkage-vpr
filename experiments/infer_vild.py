"""Resumable score-blind CAMP extraction on a frozen ViLD layout."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import concurrent.futures as cf
import hashlib
import json
import subprocess
import time

import cv2
import numpy as np
import torch

from camp_model import build, descriptor, tensor as reference_tensor
from vild_common import R, O, P, CP, config, sha, save, check_freeze, load_layout


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    save(tmp, data)
    tmp.replace(path)


def image_tensor(path):
    raw = (R/path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert im is not None, path
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (384,384), interpolation=cv2.INTER_LINEAR_EXACT).astype(np.float32)
    im -= np.array([.485,.456,.406], np.float32)*255
    im *= 1/(np.array([.229,.224,.225], np.float32)*255)
    return torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))), digest


def main():
    freeze = check_freeze()
    c = config()
    git = ['git', '-c', 'safe.directory='+R.as_posix(), '-C', str(R)]
    for name in ['experiments/infer_vild.py','experiments/evaluate_vild.py','experiments/review_vild.py',
                 'experiments/vild_common.py','configs/A-VILD-VISUAL-001.json','results/A-VILD-VISUAL-001/freeze.json']:
        subprocess.run([*git, 'ls-files','--error-unmatch',name],check=True,capture_output=True)
        subprocess.run([*git, 'diff','HEAD','--exit-code','--',name],check=True,capture_output=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(300930)
    torch.cuda.manual_seed_all(300930)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    model, loaded = build()
    assert loaded['checkpoint_sha256'] == c['checkpoint_sha256']
    model.cuda()
    batch = c['batch_size']
    frozen_hash = sha(O/'freeze.json')
    for dataset in c['datasets']:
        target = O/dataset
        target.mkdir(exist_ok=True)
        final = target/'inference.json'
        if final.exists():
            prior = json.loads(final.read_text())
            assert prior['status']=='PASS_FROZEN_INFERENCE' and prior['freeze_sha256']==frozen_hash
            assert sha(P/f'{dataset}_features.npy')==prior['features_sha256']
            continue
        d = load_layout(dataset)
        paths = np.concatenate([d['qpath'], d['gpath']]).tolist()
        fn = P/f'{dataset}_features.npy'
        log = P/f'{dataset}_input_images.csv'
        progress = target/'inference_progress.json'
        done, previous_seconds = 0, 0.
        if progress.exists():
            old = json.loads(progress.read_text())
            assert old['freeze_sha256']==frozen_hash and old['total']==len(paths)
            done = old['completed']
            previous_seconds = old['seconds']
            features = np.load(fn, mmap_mode='r+')
            assert features.shape==(len(paths),c['descriptor_dim']) and features.dtype==np.float32
            handle = log.open('r+b')
            handle.truncate(old['manifest_offset'])
            handle.seek(old['manifest_offset'])
        else:
            assert not fn.exists() and not log.exists(), 'Unowned partial output'
            features = np.lib.format.open_memmap(fn, mode='w+', dtype=np.float32, shape=(len(paths),c['descriptor_dim']))
            handle = log.open('xb')
            handle.write(b'path,sha256\n')
            handle.flush()
            atomic(progress, dict(status='INFERENCE_RUNNING', total=len(paths), completed=0,
                seconds=0., manifest_offset=handle.tell(), freeze_sha256=frozen_hash))
        start = time.perf_counter()
        with handle, cf.ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            futures = [pool.submit(image_tensor, p) for p in paths[done:done+batch]]
            for i in range(done,len(paths),batch):
                results = [f.result() for f in futures]
                if i == done:
                    assert torch.equal(results[0][0], reference_tensor({'path':paths[i], 'sha256':results[0][1]})), 'CAMP preprocessing adapter mismatch'
                futures = [pool.submit(image_tensor,p) for p in paths[i+batch:i+2*batch]]
                x = torch.stack([r[0] for r in results]).cuda()
                value = descriptor(model,x)
                assert value.shape==(len(results),c['descriptor_dim']) and torch.isfinite(value).all()
                array = value.cpu().numpy()
                assert float(abs(np.linalg.norm(array,axis=1)-1).max())<1e-5
                assert torch.cuda.max_memory_allocated()/1024**2 < c['peak_allocated_limit_mib']
                features[i:i+len(results)] = array
                for path, (_, digest) in zip(paths[i:i+len(results)], results):
                    assert ',' not in path and '\n' not in path
                    handle.write(f'{path},{digest}\n'.encode('utf8'))
                completed = i+len(results)
                if completed%512==0 or completed==len(paths):
                    features.flush()
                    handle.flush()
                    elapsed = previous_seconds+time.perf_counter()-start
                    atomic(progress, dict(status='INFERENCE_RUNNING', total=len(paths), completed=completed,
                        seconds=elapsed, manifest_offset=handle.tell(), freeze_sha256=frozen_hash))
                    print(json.dumps(dict(dataset=dataset, images=completed, total=len(paths), seconds=round(elapsed,1))),flush=True)
            repeat = descriptor(model,image_tensor(paths[0])[0][None].cuda()).cpu().numpy()[0]
            delta = float(abs(repeat-features[0]).max())
            assert delta < 1e-5
        features.flush()
        # Exact byte duplicates across calibration/evaluation would invalidate the
        # intended split even when coordinate metadata claims separation.
        import csv
        with log.open(encoding='utf8',newline='') as f:
            hashes = [r['sha256'] for r in csv.DictReader(f)]
        nq = len(d['qpath'])
        cal = {h for h, role in zip(hashes[:nq], d['qrole']) if role=='calibration'}
        ev = {h for h, role in zip(hashes[:nq], d['qrole']) if role=='evaluation'}
        cross = sorted(cal&ev)
        save(target/'duplicate_query_check.json', dict(cross_calibration_evaluation_byte_duplicates=len(cross), hashes=cross))
        assert not cross, 'Byte-identical queries cross the calibration/evaluation split'
        save(final, dict(status='PASS_FROZEN_INFERENCE', images=len(paths),
            seconds=previous_seconds+time.perf_counter()-start,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            repeat_max_abs=delta, load=loaded, freeze_sha256=frozen_hash,
            features_sha256=sha(fn), image_manifest_sha256=sha(log),
            exact_input_byte_unique=len(set(hashes)), cross_split_query_duplicates=len(cross)))
        print(json.dumps(dict(dataset=dataset,status='PASS_FROZEN_INFERENCE')),flush=True)


if __name__=='__main__':
    main()
