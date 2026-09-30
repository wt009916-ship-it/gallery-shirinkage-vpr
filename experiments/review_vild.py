"""Reconstruct counts/acceptance and audit sampled rankings independently."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from vild_common import R, O, P, config, sha, save, check_freeze, load_layout, masks


def independent_fit(scores,eligible,correct,cells,minimum):
    ids=np.flatnonzero(eligible)
    ids=ids[np.argsort(-scores[ids],kind='stable')]
    if not len(ids):
        return None
    errors=np.cumsum(~correct[ids])
    sizes=np.arange(1,len(ids)+1)
    _,first=np.unique(cells[ids],return_index=True)
    new_cell=np.zeros(len(ids),np.int32)
    new_cell[first]=1
    distinct=np.cumsum(new_cell)
    ends=np.r_[scores[ids[:-1]]!=scores[ids[1:]],True]
    feasible=ends & (errors*10<=sizes) & (np.asarray(distinct)>=minimum)
    if not feasible.any():
        return None
    k=np.flatnonzero(feasible)[-1]
    return dict(threshold=float(scores[ids[k]]),answered=int(sizes[k]),errors=int(errors[k]),cells=int(distinct[k]))


def main():
    check_freeze()
    c=config()
    for dataset in c['datasets']:
        start=time.perf_counter()
        target=O/dataset
        record=json.loads((target/'evaluation.json').read_text())
        for p,h in record['output_sha256'].items():
            assert sha(R/p)==h,p
        d=load_layout(dataset)
        nq,ng=len(d['qpath']),len(d['gpath'])
        dead=np.unpackbits(d['deleted_packed'],axis=1,count=ng).astype(bool)
        cases=json.loads((O/f'{dataset}_cases.json').read_text())
        recases,redead=masks(d['gxy'],c)
        assert cases==recases and np.array_equal(dead,redead)
        del redead
        w=np.load(P/f'{dataset}_winners.npy',mmap_mode='r')
        top=np.load(P/f'{dataset}_top10.npy',mmap_mode='r')
        near=np.load(P/f'{dataset}_nearest.npy',mmap_mode='r')
        f=np.load(P/f'{dataset}_features.npy',mmap_mode='r')
        checks=1
        # Fixed evenly-spaced queries, full direct stable sort inside each active
        # gallery. Same FP32 dot-product block shape avoids changing score rounding.
        samples=np.unique(np.linspace(0,nq-1,8,dtype=int))
        for j in samples:
            b=int(j)//64*64
            score=(f[b:min(b+64,nq)]@f[nq:].T)[int(j)-b]
            for ci in range(len(cases)):
                active=np.flatnonzero(~dead[ci])
                direct=active[np.argsort(-score[active],kind='stable')[:10]]
                assert direct[0]==w[ci,j]
                assert np.array_equal(score[direct],top[ci,j])
                assert abs(np.linalg.norm(d['gxy'][active]-d['qxy'][j],axis=1).min()-near[ci,j])<1e-8
                checks+=3
        ca=d['qrole']=='calibration'
        ev=~ca
        radii=[c['primary_radius_m']]+c['secondary_radius_m']
        coverage={}
        tree=cKDTree(d['gxy'])
        for radius in radii:
            neighbors=tree.query_ball_point(d['qxy'],radius)
            independent=np.empty((len(cases),nq),bool)
            for ci in range(len(cases)):
                independent[ci]=[np.any(~dead[ci,ix]) for ix in neighbors]
                assert np.array_equal(independent[ci],near[ci]<=radius)
                checks+=1
            coverage[radius]=independent
        # Algebraically equivalent SU computed using the two middle order values.
        values=top.astype(float)
        su=-(values[:,:,1:].sum(2)/9+(values[:,:,4]+values[:,:,5])/2)/(2*values[:,:,0])
        exact_su=-.5*(values[:,:,1:].mean(2)+np.median(values,axis=2))/values[:,:,0]
        assert np.allclose(su,exact_su,rtol=0,atol=1e-12)
        dist=np.hypot(d['gxy'][w,0]-d['qxy'][None,:,0],d['gxy'][w,1]-d['qxy'][None,:,1])
        guard=~dead[:,w[0]]
        assert np.array_equal(w[guard],np.broadcast_to(w[0],w.shape)[guard])
        thresholds=json.loads((target/'thresholds.json').read_text())
        fits={}
        for row in thresholds:
            ci=next(i for i,case in enumerate(cases) if case['case_id']==row['case_id'])
            m=row['method']
            score=exact_su[0] if m=='old_su_guard' else exact_su[ci]
            eligible=np.ones(nq,bool) if m=='current_su' else guard[ci]
            expected=independent_fit(score[ca],eligible[ca],dist[ci,ca]<=100,d['qcell'][ca],c['min_calibration_accepted_cells'])
            actual=row['fit']
            assert (expected is None)==(actual is None)
            if expected:
                for key in ['answered','errors','cells']:
                    assert expected[key]==actual[key]
                assert abs(expected['threshold']-actual['threshold'])<1e-12
            fits[ci,m]=actual
            checks+=5
        metrics=json.loads((target/'metrics.json').read_text())
        with np.load(P/f'{dataset}_acceptance.npz',allow_pickle=False) as z:
            accepted=z['accepted']
        _, cell=np.unique(d['qcell'][ev],return_inverse=True)
        totals=np.bincount(cell)
        for row in metrics:
            ci,ai=row['prediction_row'],row['acceptance_row']
            m=row['method']
            score=su[0] if m=='old_su_guard' else su[ci]
            eligible=np.ones(nq,bool) if m=='current_su' else guard[ci]
            a=accepted[ai]
            expect=np.zeros(nq,bool)
            if row['regime']=='budget50':
                ids=np.flatnonzero(ev&eligible)
                feasible=len(ids)>=int(ev.sum())//2
                if feasible:
                    # Use original top-value expression for exact tie ordering;
                    # alternate algebra above may differ by a final rounding bit.
                    tv=values[0] if m=='old_su_guard' else values[ci]
                    exact=-.5*(tv[:,1:].mean(1)+np.median(tv,axis=1))/tv[:,0]
                    expect[ids[np.argsort(-exact[ids],kind='stable')[:int(ev.sum())//2]]]=True
            else:
                ft=fits[0 if row['regime']=='full_frozen' else ci,m]
                feasible=ft is not None
                if feasible:
                    tv=values[0] if m=='old_su_guard' else values[ci]
                    exact=-.5*(tv[:,1:].mean(1)+np.median(tv,axis=1))/tv[:,0]
                    expect=ev&eligible&(exact>=ft['threshold'])
            assert bool(row['feasible'])==bool(feasible) and np.array_equal(a,expect)
            radius=row['radius_m']
            wrong=dist[ci]>radius
            oldwrong=dist[0]>radius
            created=wrong&~oldwrong
            covered=coverage[radius][ci]
            wanted=dict(queries=int(ev.sum()),available=int((ev&eligible).sum()),answered=int(a.sum()),
                errors=int((a&wrong).sum()),correct=int((a&~wrong).sum()),
                created_errors=int((a&created).sum()),inherited_errors=int((a&wrong&oldwrong).sum()),
                created_coverage_present=int((a&created&covered).sum()),created_coverage_absent=int((a&created&~covered).sum()),
                accepted_uncovered=int((a&~covered).sum()))
            for key,value in wanted.items():
                assert row[key]==value,(key,ai,radius)
                checks+=1
            assert row['errors']==row['created_errors']+row['inherited_errors']
            assert row['created_errors']==row['created_coverage_absent']+row['created_coverage_present']
            assert not (a&~ev).any() and not (a&~eligible).any()
            answered_by_cell=np.bincount(cell,weights=a[ev],minlength=len(totals))
            errors_by_cell=np.bincount(cell,weights=(a&wrong)[ev],minlength=len(totals))
            active_cells=answered_by_cell>0
            assert row['answered_cells']==int(active_cells.sum())
            expected_risk=wanted['errors']/wanted['answered'] if wanted['answered'] else None
            assert row['risk']==expected_risk
            useful=bool(wanted['answered'] and wanted['errors']*10<=wanted['answered'] and active_cells.sum()>=8)
            assert row['empirical_useful10']==useful
            if row['regime']=='budget50':
                assert row['answered']==(int(ev.sum())//2 if feasible else 0)
            expected_macro=float(np.mean(errors_by_cell[active_cells]/answered_by_cell[active_cells])) if active_cells.any() else None
            if expected_macro is None:
                assert row['cell_macro_selective_risk'] is None
            else:
                assert abs(row['cell_macro_selective_risk']-expected_macro)<1e-12
            checks+=5
            if m.endswith('guard'):
                assert row['created_errors']==0
            for field, flags in [('answer',a),('error_return',a&wrong),('correct_return',a&~wrong)]:
                fraction=np.bincount(cell,weights=flags[ev],minlength=len(totals))/totals
                assert abs(row['cell_macro_'+field+'_rate']-fraction.mean())<1e-12
                checks+=1
            checks+=4
        for row in json.loads((target/'raw_errors.json').read_text()):
            ci=next(i for i,case in enumerate(cases) if case['case_id']==row['case_id'])
            radius=row['radius_m']
            before=dist[0]>radius
            after=dist[ci]>radius
            cover=coverage[radius][ci]
            assert row['created_coverage_present']==int((ev&~before&after&cover).sum())
            assert row['created_coverage_absent']==int((ev&~before&after&~cover).sum())
            assert row['created']==row['created_coverage_present']+row['created_coverage_absent']
            assert row['errors']==row['created']+row['inherited']
            assert row['errors']-row['historical_errors']==row['created']-row['repaired']
            assert row['rejected_correct_fallback_after_old_correct']==int((ev&~guard[ci]&~after&~before).sum())
            assert row['rejected_correct_fallback_after_old_wrong']==int((ev&~guard[ci]&~after&before).sum())
            checks+=7
        for item in json.loads((target/'summary.json').read_text()):
            keys=['arm','target_fraction','method','regime','radius_m']
            rr=[r for r in metrics if all(r[k]==item[k] for k in keys)]
            assert len(rr)==c['seeds']
            assert item['feasible_masks']==sum(r['feasible'] for r in rr)
            assert item['useful10_masks']==sum(r['empirical_useful10'] for r in rr)
            selected=[r for r in rr if r['feasible']] if item['regime']=='budget50' else rr
            for key,value in item.items():
                if key.startswith('mean_'):
                    expected=float(np.mean([r[key[5:]] for r in selected])) if selected else None
                    assert expected==value
                    checks+=1
        save(target/'review.json',dict(status='PASS_RECONSTRUCTED_COUNTS_COVERAGE_AND_SAMPLED_RANKINGS',
            checks=checks,metric_rows=len(metrics),rank_queries_checked=len(samples),rank_cases_per_query=len(cases),
            coverage_all_queries_and_cases=True,seconds=time.perf_counter()-start,
            scope='All acceptance/count/summary reconstruction; sampled direct ranking, not every full dot product recomputed; no physical-coordinate or risk-guarantee certification.',
            evaluation_sha256=sha(target/'evaluation.json')))
        print(json.dumps(dict(dataset=dataset,status='PASS_REVIEW',checks=checks)),flush=True)


if __name__=='__main__':
    main()
