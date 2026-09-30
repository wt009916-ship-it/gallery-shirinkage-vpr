"""Build score-blind lists after verified extraction/image intake, before GPU work."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from vild_common import R, O, P, CP, config, sha, save, masks


def main():
    c = config()
    assert json.loads((R/'results/VILD-INTAKE-001/image_audit.json').read_text())['status'] == 'PASS_ALL_SPLIT_PATHS_AND_FIXED_IMAGE_SAMPLE'
    O.mkdir(exist_ok=True)
    P.mkdir(exist_ok=True)
    assert not (O/'freeze.json').exists(), 'Existing freeze must not be overwritten'
    results = []
    paths = [CP, Path(__file__), R/'experiments/vild_common.py',
        R/'experiments/infer_vild.py', R/'experiments/evaluate_vild.py',
        R/'experiments/review_vild.py', R/'experiments/camp_model.py',
        R/'results/CAMP-INTAKE-001/checkpoint.json', R/'manifests/CAMP_SOURCE.json',
        R/'results/VILD-INTAKE-001/image_audit.json']
    for dataset in c['datasets']:
        flight = 'flight10' if dataset == 'vild' else 'flight09'
        rows = {}
        for split in ['val', 'test']:
            path = R/f'data/vild_full/ViLD_dataset/{dataset}/{dataset}_coordinates__{split}.csv'
            with path.open(encoding='utf-8-sig', newline='') as f:
                rows[split] = list(csv.DictReader(f))
            paths.append(path)
        query = {s: sorted([r for r in rr if r['image_type']=='query'], key=lambda r:r['filename']) for s, rr in rows.items()}
        xy = lambda rr: np.array([[float(r['utm_east']), float(r['utm_north'])] for r in rr], np.float64)
        separation = cKDTree(xy(query['val'])).query(xy(query['test']))[0]
        keep = separation >= c['buffer_m']
        ev = [r for r, ok in zip(query['test'], keep) if ok]
        qq = query['val'] + ev
        gg = sorted([r for rr in rows.values() for r in rr if r['image_type']=='reference'], key=lambda r:r['filename'])
        assert len({r['filename'] for r in gg}) == len(gg)
        assert len({r['filename'] for r in qq}) == len(qq)
        qxy, gxy = xy(qq), xy(gg)
        near = cKDTree(gxy).query(qxy)[0]
        assert (near <= min(c['secondary_radius_m'])).all()
        case, dead = masks(gxy, c)
        root = f'data/vild_full/ViLD_dataset/{flight}'
        qp = np.array([root+'_uav/'+r['filename'] for r in qq])
        gp = np.array([root+'_satellite/'+r['filename'] for r in gg])
        qrole = np.array(['calibration']*len(query['val'])+['evaluation']*len(ev))
        _, qcell = np.unique(np.floor(qxy/c['spatial_cell_m']).astype(np.int64), axis=0, return_inverse=True)
        np.savez_compressed(P/f'{dataset}.npz', qpath=qp, gpath=gp, qxy=qxy, gxy=gxy,
            qrole=qrole, qcell=qcell, deleted_packed=np.packbits(dead, axis=1),
            excluded_test_names=np.array([r['filename'] for r, ok in zip(query['test'], keep) if not ok]))
        save(O/f'{dataset}_cases.json', case)
        paths += [P/f'{dataset}.npz', O/f'{dataset}_cases.json']
        results.append(dict(dataset=dataset, calibration_queries=len(query['val']), evaluation_queries=len(ev),
            original_test_queries=len(query['test']), buffer_excluded=int((~keep).sum()), gallery_images=len(gg),
            total_images=len(qq)+len(gg), minimum_retained_separation_m=float(separation[keep].min()),
            calibration_cells=len(set(qcell[qrole=='calibration'])), evaluation_cells=len(set(qcell[qrole=='evaluation'])),
            cases=len(case), feature_bytes=(len(qq)+len(gg))*c['descriptor_dim']*4,
            conservative_gpu_minutes_at_25_images_per_second=(len(qq)+len(gg))/25/60))
        print(json.dumps(results[-1]), flush=True)
    save(O/'freeze.json', dict(status='LAYOUT_FIXED_BEFORE_INFERENCE', created_epoch=time.time(),
        datasets=results, input_sha256={p.relative_to(R).as_posix():sha(p) for p in paths},
        model_scored=False, independent_confirmation=False))


if __name__ == '__main__':
    main()
