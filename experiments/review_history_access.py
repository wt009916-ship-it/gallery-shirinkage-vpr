"""Post-run reviewer; exact run enumeration fixes vild/vild_09 prefix collision.

The frozen timing implementation and all observations remain unchanged.
"""
from history_access import R, C, O, P, read, save, sha, su
import numpy as np


def main():
    frozen = read(O / 'freeze.json')
    c = read(C)
    for p, h in frozen['input_sha256'].items():
        assert sha(R / p) == h, p
    expected = []
    rows, disagreements = [], []
    checks = 0
    for rec in frozen['datasets']:
        ds, idx = rec['dataset'], rec['source_query_indices']
        base = R / 'data/vild_protocol_001'
        winners = np.load(base / f'{ds}_winners.npy', mmap_mode='r')
        tops = np.load(base / f'{ds}_top10.npy', mmap_mode='r')
        cases = [(ci, mode, False) for ci in c['case_indices'] for mode in c['modes']]
        cases.append((c['end_to_end_case_index'], 'archive_recompute', True))
        for ci, mode, image_mode in cases:
            name = f'{ds}_{ci}_{mode}' + ('_images' if image_mode else '') + '.json'
            expected.append(name)
            z = read(O / 'runs' / name)
            assert (z['dataset'], z['case_index'], z['mode'], z['image_mode']) == (ds, ci, mode, image_mode)
            n = c['end_to_end_images_per_dataset'] if image_mode else c['queries_per_dataset']
            assert len(z['predictions']) == n
            assert len(z['samples']) == n * c['timed_repeats']
            assert {(s['repeat'], s['query_index']) for s in z['samples']} == {(t,k) for t in range(c['timed_repeats']) for k in range(n)}
            active = np.load(P / ds / f'alive_{ci}.npy')
            if mode != 'cached_history':
                assert not any('cache' in x or 'winner' in x for x in z['opened_data'])
            max_score_diff = max_su_diff = 0.
            for k, out in enumerate(z['predictions']):
                i = idx[k]
                assert active[out['winner']]
                err = float(np.max(abs(np.array(out['top10_scores']) - tops[ci,i])))
                serr = abs(out['current_su'] - su(tops[ci,i]))
                max_score_diff = max(max_score_diff, err)
                max_su_diff = max(max_su_diff, serr)
                assert err < c['numeric_atol'] and serr < c['numeric_atol']
                if out['winner'] != int(winners[ci,i]):
                    disagreements.append(dict(run=name, query_index=k, type='current_winner'))
                if mode != 'active_only':
                    assert abs(out['historical_su'] - su(tops[0,i])) < c['numeric_atol']
                    assert out['guard'] == bool(active[out['historical_winner']])
                    if out['historical_winner'] != int(winners[0,i]):
                        disagreements.append(dict(run=name, query_index=k, type='historical_winner'))
                    if out['guard']:
                        assert out['winner'] == out['historical_winner']
                checks += 1
            ms = [x['total_ms'] for x in z['samples']]
            if image_mode:
                assert max(x['feature_max_abs'] for x in z['samples']) < c['numeric_atol']
            rows.append(dict(dataset=ds, case_index=ci, mode=mode, image_mode=image_mode,
                queries=n, observations=len(ms), median_ms=float(np.median(ms)),
                p95_ms=float(np.percentile(ms,95)), min_ms=min(ms), max_ms=max(ms),
                gallery_bytes=z['array_bytes']['gallery'], array_payload_bytes=sum(z['array_bytes'].values()),
                cache_serialized_bytes=z['cache_serialized_bytes'],
                input_load_seconds=z['input_load_seconds'],
                process_peak_working_set_bytes=z['memory_end']['process_peak_working_set_bytes'],
                gpu_peak_allocated_mib=z['gpu_peak_allocated_mib'],
                max_score_abs_diff= max_score_diff, max_su_abs_diff=max_su_diff,
                encode_median_ms=None if not image_mode else float(np.median([x['encode_decode_transfer_ms'] for x in z['samples']]))))
    assert set(expected) == {p.name for p in (O/'runs').glob('*.json')}
    save(O/'summary.json', rows)
    save(O/'review.json', dict(
        status='PASS_DEPLOYMENT_ACCESS_AND_NUMERIC_REPLAY' if not disagreements else 'MODIFY_RANK_DISAGREEMENTS',
        checks=checks, runs=len(rows), winner_disagreements=disagreements,
        reviewer_fix='Frozen reviewer glob vild_*.json also matched vild_09 files; this reviewer enumerates exact protocol tuples. Timing worker, freeze and raw observations unchanged.',
        scientific_scope='Engineering access/cost replay only, not new accuracy evidence, arbitrary future-query performance or risk guarantee',
        reviewer_sha256=sha(__file__), freeze_sha256=sha(O/'freeze.json'), summary_sha256=sha(O/'summary.json'),
        runs_sha256={name:sha(O/'runs'/name) for name in sorted(expected)}))
    print('REVIEW', checks, 'rank disagreements', len(disagreements))


if __name__ == '__main__':
    main()
