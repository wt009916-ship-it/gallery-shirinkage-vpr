"""Fixed 25m sensitivity; never refit the 100m parent acceptance rules."""
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from vild_common import R,O,P,sha,save,load_layout,config


def main():
    cp=R/'configs/A-VILD-25M-SENSITIVITY-001.json'
    c=json.loads(cp.read_text())
    assert c['radius_m']==25 and c['primary_changed'] is False
    source=R/c['source_file']
    assert 'positive_dist_threshold=[25]' in source.read_text()
    out=R/'results/A-VILD-25M-SENSITIVITY-001'
    out.mkdir(exist_ok=True)
    records=[]
    checks=0
    inputs={cp.relative_to(R).as_posix():sha(cp),c['source_file']:sha(source),
            'experiments/rescore_vild_25m.py':sha(__file__)}
    for dataset in config()['datasets']:
        target=O/dataset
        review=json.loads((target/'review.json').read_text())
        assert review['status']=='PASS_RECONSTRUCTED_COUNTS_COVERAGE_AND_SAMPLED_RANKINGS'
        inputs[(target/'review.json').relative_to(R).as_posix()]=sha(target/'review.json')
        rows=json.loads((target/'metrics.json').read_text())
        d=load_layout(dataset)
        winner=np.load(P/f'{dataset}_winners.npy')
        nearest=np.load(P/f'{dataset}_nearest.npy')
        with np.load(P/f'{dataset}_acceptance.npz',allow_pickle=False) as z:
            masks=z['accepted']
        dead=np.unpackbits(d['deleted_packed'],axis=1,count=len(d['gpath'])).astype(bool)
        delta=d['gxy'][winner]-d['qxy'][None,:,:]
        correct=(delta*delta).sum(-1)<=25**2
        check_correct=np.linalg.norm(delta,axis=-1)<=25
        assert np.array_equal(correct,check_correct)
        neighbors=cKDTree(d['gxy']).query_ball_point(d['qxy'],25)
        for ci in range(len(dead)):
            supported=np.array([np.any(~dead[ci,ix]) for ix in neighbors])
            assert np.array_equal(supported,nearest[ci]<=25)
            checks+=1
        ev=d['qrole']=='evaluation'
        _,cells=np.unique(d['qcell'][ev],return_inverse=True)
        cell_count=np.bincount(cells)
        for row in rows:
            if row['radius_m']!=100:
                continue
            ci,ai=row['prediction_row'],row['acceptance_row']
            a=masks[ai]
            wrong=~correct[ci]
            created=wrong&correct[0]
            covered=nearest[ci]<=25
            errors=int(np.count_nonzero(a&wrong))
            inherited=int(np.count_nonzero(a&wrong&~correct[0]))
            new=int(np.count_nonzero(a&created))
            assert errors==inherited+new and errors>=row['errors']
            if row['method'].endswith('guard'):
                assert new==0
            r={k:row[k] for k in ['case_id','arm','target_fraction','seed','method','regime','feasible','queries','answered','acceptance_row','prediction_row']}
            r.update(dataset=dataset,radius_m=25,parent_calibration_radius_m=100,errors=errors,
                correct=row['answered']-errors,risk=errors/row['answered'] if row['answered'] else None,
                created_errors=new,inherited_errors=inherited,
                created_coverage_present=int((a&created&covered).sum()),
                created_coverage_absent=int((a&created&~covered).sum()),
                cell_macro_error_return_rate=float(np.mean(np.bincount(cells,weights=(a&wrong)[ev])/cell_count)))
            assert r['created_errors']==r['created_coverage_present']+r['created_coverage_absent']
            records.append(r)
            checks+=5
    save(out/'metrics.json',records)
    summaries=[]
    for dataset in config()['datasets']:
        for arm in ['spatial','matched_random']:
            for fraction in [.25,.5]:
                for method in config()['methods']:
                    for regime in config()['regimes']:
                        key=dict(dataset=dataset,arm=arm,target_fraction=fraction,method=method,regime=regime)
                        rr=[r for r in records if all(r[k]==v for k,v in key.items())]
                        assert len(rr)==20
                        selected=[r for r in rr if r['feasible']] if regime=='budget50' else rr
                        summaries.append(key|dict(feasible_masks=sum(r['feasible'] for r in rr),
                            mean_answered=float(np.mean([r['answered'] for r in selected])) if selected else None,
                            mean_errors=float(np.mean([r['errors'] for r in selected])) if selected else None,
                            mean_created_errors=float(np.mean([r['created_errors'] for r in selected])) if selected else None))
    save(out/'summary.json',summaries)
    save(out/'review.json',dict(status='PASS_IDENTICAL_DECISIONS_STRICTER_LABEL_RESCORING',checks=checks,rows=len(records),
        input_sha256=inputs,output_sha256={p.name:sha(p) for p in [out/'metrics.json',out/'summary.json']},
        new_thresholds=0,new_predictions=0,primary_changed=False,
        label_scope='25m in published reference coordinates; parent thresholds were fitted at 100m'))
    print(json.dumps(dict(status='PASS_25M_SENSITIVITY',checks=checks,rows=len(records))),flush=True)


if __name__=='__main__':
    main()
