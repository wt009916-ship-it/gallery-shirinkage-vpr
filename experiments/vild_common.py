"""Shared data layout and deterministic masks; no model fitting here."""
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
O = R / 'results/A-VILD-VISUAL-001'
P = R / 'data/vild_protocol_001'
CP = R / 'configs/A-VILD-VISUAL-001.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def config():
    return json.loads(CP.read_text(encoding='utf8'))


def masks(gxy, c):
    cells, inverse = np.unique(np.floor(gxy/c['spatial_cell_m']).astype(np.int64), axis=0, return_inverse=True)
    sizes = np.bincount(inverse)
    cases = [dict(case_id='full', arm='full', target_fraction=0., seed=0, removed_entries=0, actual_entry_fraction=0.)]
    deleted = [np.zeros(len(gxy), bool)]
    for fraction in c['target_entry_fractions']:
        for seed in range(c['seed_start'], c['seed_start']+c['seeds']):
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(cells))
            cut = np.searchsorted(np.cumsum(sizes[order]), int(np.ceil(fraction*len(gxy))))+1
            spatial = np.isin(inverse, order[:cut])
            count = int(spatial.sum())
            random = np.zeros(len(gxy), bool)
            random[rng.choice(len(gxy), size=count, replace=False)] = True
            assert count == int(random.sum()) and count < len(gxy)-10
            for arm, dead in [('spatial', spatial), ('matched_random', random)]:
                cases.append(dict(case_id=f'{arm}_{int(fraction*100)}_{seed}', arm=arm,
                    target_fraction=fraction, seed=seed, removed_entries=count,
                    actual_entry_fraction=count/len(gxy)))
                deleted.append(dead)
    return cases, np.stack(deleted)


def load_layout(dataset):
    with np.load(P/f'{dataset}.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_freeze():
    freeze = json.loads((O/'freeze.json').read_text(encoding='utf8'))
    for path, digest in freeze['input_sha256'].items():
        assert sha(R/path) == digest, path
    return freeze
