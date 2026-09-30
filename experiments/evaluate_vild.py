"""Pure-deletion ranking and fixed descriptive calibration; no score tuning."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json
import time

import numpy as np
from scipy.spatial import cKDTree

from vild_common import R, O, P, config, sha, save, check_freeze, load_layout

BLOCK = 64


def confidence(top):
    v = top.astype(np.float64)
    assert v.shape[-1]==10 and np.isfinite(v).all() and (v[...,0]>1e-8).all()
    return -.5*(v[...,1:].mean(-1)+np.median(v,axis=-1))/v[...,0]


def fit(score, eligible, correct, cells, minimum):
    order = np.flatnonzero(eligible)
    order = order[np.argsort(-score[order],kind='stable')]
    seen, errors, answer = set(), 0, None
    for n,j in enumerate(order,1):
        seen.add(int(cells[j]))
        errors += int(not correct[j])
        if n<len(order) and score[order[n]]==score[j]:
            continue
        if len(seen)>=minimum and 10*errors<=n:
            answer = dict(threshold=float(score[j]), answered=n, errors=errors, cells=len(seen))
    return answer


def rank_dataset(dataset,d,dead,features):
    nq, ng = len(d['qpath']), len(d['gpath'])
    nc = len(dead)
    base = P/dataset
    winpath, toppath, nearpath = [base.with_name(dataset+'_'+n+'.npy') for n in ['winners','top10','nearest']]
    target = O/dataset
    donefile = target/'ranking.json'
    if donefile.exists():
        record = json.loads(donefile.read_text())
        for p,h in record['files'].items():
            assert sha(R/p)==h,p
        return np.load(winpath),np.load(toppath),np.load(nearpath)
    progress = target/'ranking_progress.json'
    done = 0
    mode = 'w+'
    if progress.exists():
        state = json.loads(progress.read_text())
        assert state['freeze_sha256']==sha(O/'freeze.json')
        done = state['completed_queries']
        mode = 'r+'
    else:
        assert not any(p.exists() for p in [winpath,toppath,nearpath])
    winner = np.lib.format.open_memmap(winpath,mode=mode,dtype=np.int32,shape=(nc,nq))
    top = np.lib.format.open_memmap(toppath,mode=mode,dtype=np.float32,shape=(nc,nq,10))
    nearest = np.lib.format.open_memmap(nearpath,mode=mode,dtype=np.float64,shape=(nc,nq))
    start = time.perf_counter()
    # Coverage is computed from coordinates independently of visual rankings.
    for ci in range(nc):
        nearest[ci] = cKDTree(d['gxy'][~dead[ci]]).query(d['qxy'])[0]
    nearest.flush()
    checks, slow = 0, 0
    g = features[nq:]
    for i in range(done,nq,BLOCK):
        s = features[i:min(i+BLOCK,nq)]@g.T
        full = np.argsort(-s,axis=1,kind='stable')
        for k, rank in enumerate(full):
            for ci in range(nc):
                probe = min(128,ng)
                while True:
                    retained = rank[:probe][~dead[ci,rank[:probe]]]
                    if len(retained)>=10:
                        break
                    assert probe<ng
                    probe = min(ng,probe*2)
                chosen = retained[:10]
                winner[ci,i+k] = chosen[0]
                top[ci,i+k] = s[k,chosen]
                slow += int(probe>128)
                if not dead[ci,rank[0]]:
                    assert chosen[0]==rank[0]
                checks += 1
        winner.flush()
        top.flush()
        save(progress,dict(status='RANKING_RUNNING',completed_queries=min(i+BLOCK,nq),total_queries=nq,freeze_sha256=sha(O/'freeze.json')))
        if i%(BLOCK*8)==0 or i+BLOCK>=nq:
            print(json.dumps(dict(dataset=dataset, ranked_queries=min(i+BLOCK,nq), total=nq,seconds=round(time.perf_counter()-start,1))),flush=True)
    save(donefile,dict(status='PASS_FIXED_SCORE_DELETION_RANKING',queries=nq,cases=nc,seconds=time.perf_counter()-start,
        block=BLOCK,guard_checks_this_run=checks,expanded_prefix_cases_this_run=slow,
        files={p.relative_to(R).as_posix():sha(p) for p in [winpath,toppath,nearpath]}))
    return winner,top,nearest


def evaluate(dataset,c,d,dead,cases,winner,top,nearest):
    target = O/dataset
    ca = d['qrole']=='calibration'
    ev = ~ca
    n = int(ev.sum())
    old = winner[0]
    su = confidence(top)
    meters = np.linalg.norm(d['gxy'][winner]-d['qxy'][None,:,:],axis=-1)
    radii = [c['primary_radius_m']]+c['secondary_radius_m']
    rows, raw, thresholds, masks_out, fullfits = [], [], [], [], {}
    cells = np.unique(d['qcell'][ev])
    cell_ix = [ev&(d['qcell']==cell) for cell in cells]
    for ci,case in enumerate(cases):
        guard = ~dead[ci,old]
        assert np.array_equal(winner[ci,guard],old[guard])
        specs = [(su[ci],np.ones(len(ev),bool)),(su[ci],guard),(su[0],guard)]
        for radius in radii:
            wrong = meters[ci]>radius
            oldwrong = meters[0]>radius
            created = wrong&~oldwrong
            covered = nearest[ci]<=radius
            assert not (created&guard).any()
            raw.append(case|dict(radius_m=radius, queries=n, errors=int((ev&wrong).sum()),
                historical_errors=int((ev&oldwrong).sum()), created=int((ev&created).sum()),
                created_coverage_present=int((ev&created&covered).sum()),
                created_coverage_absent=int((ev&created&~covered).sum()),
                inherited=int((ev&wrong&oldwrong).sum()), repaired=int((ev&~wrong&oldwrong).sum()),
                uncovered=int((ev&~covered).sum()),
                rejected_correct_fallback_after_old_correct=int((ev&~guard&~wrong&~oldwrong).sum()),
                rejected_correct_fallback_after_old_wrong=int((ev&~guard&~wrong&oldwrong).sum())))
        for method,(score,eligible) in zip(c['methods'],specs):
            fitted = fit(score[ca],eligible[ca],meters[ci,ca]<=c['primary_radius_m'],d['qcell'][ca],c['min_calibration_accepted_cells'])
            thresholds.append(case|dict(method=method,fit=fitted))
            if ci==0:
                fullfits[method]=fitted
            for regime in c['regimes']:
                a = np.zeros(len(ev),bool)
                threshold = None
                available = int((ev&eligible).sum())
                budget = n//2 if regime=='budget50' else None
                if regime=='budget50':
                    feasible = available>=budget
                    if feasible:
                        ix = np.flatnonzero(ev&eligible)
                        a[ix[np.argsort(-score[ix],kind='stable')[:budget]]]=True
                else:
                    ft = fitted if regime=='matched' else fullfits[method]
                    feasible = ft is not None
                    if feasible:
                        threshold = ft['threshold']
                        a = ev&eligible&(score>=threshold)
                ai = len(masks_out)
                masks_out.append(a)
                answered = int(a.sum())
                for radius in radii:
                    wrong,oldwrong = meters[ci]>radius,meters[0]>radius
                    created = wrong&~oldwrong
                    covered = nearest[ci]<=radius
                    errors = int((a&wrong).sum())
                    nonempty = [ix for ix in cell_ix if (a&ix).any()]
                    rows.append(case|dict(method=method,regime=regime,radius_m=radius,queries=n,
                        available=available,budget=budget,feasible=feasible,threshold=threshold,
                        answered=answered,errors=errors,correct=answered-errors,risk=errors/answered if answered else None,
                        created_errors=int((a&created).sum()),inherited_errors=int((a&wrong&oldwrong).sum()),
                        created_coverage_present=int((a&created&covered).sum()),
                        created_coverage_absent=int((a&created&~covered).sum()),
                        accepted_uncovered=int((a&~covered).sum()),answered_cells=len(nonempty),
                        empirical_useful10=bool(answered and 10*errors<=answered and len(nonempty)>=8),
                        cell_macro_answer_rate=float(np.mean([a[ix].mean() for ix in cell_ix])),
                        cell_macro_error_return_rate=float(np.mean([(a&wrong)[ix].mean() for ix in cell_ix])),
                        cell_macro_correct_return_rate=float(np.mean([(a&~wrong)[ix].mean() for ix in cell_ix])),
                        cell_macro_selective_risk=float(np.mean([(a&wrong&ix).sum()/(a&ix).sum() for ix in nonempty])) if nonempty else None,
                        acceptance_row=ai,prediction_row=ci))
        if ci%20==0:
            print(json.dumps(dict(dataset=dataset, evaluated_cases=ci+1,rows=len(rows))),flush=True)
    summary=[]
    for arm in ['spatial','matched_random']:
        for fraction in c['target_entry_fractions']:
            for method in c['methods']:
                for regime in c['regimes']:
                    for radius in radii:
                        key=dict(arm=arm,target_fraction=fraction,method=method,regime=regime,radius_m=radius)
                        rr=[r for r in rows if all(r[k]==v for k,v in key.items())]
                        assert len(rr)==c['seeds']
                        feasible=[r for r in rr if r['feasible']]
                        used=feasible if regime=='budget50' else rr
                        item=key|dict(feasible_masks=len(feasible),useful10_masks=sum(r['empirical_useful10'] for r in rr),masks=len(rr))
                        for name in ['answered','errors','correct','created_errors','inherited_errors','created_coverage_present','created_coverage_absent','accepted_uncovered','cell_macro_answer_rate','cell_macro_error_return_rate','cell_macro_correct_return_rate']:
                            item['mean_'+name]=float(np.mean([r[name] for r in used])) if used else None
                        summary.append(item)
    p=c['primary']
    pick=lambda method:next(r for r in summary if r['method']==method and all(r[k]==p[k] for k in ['arm','target_fraction','regime','radius_m']))
    base,hist=pick(p['base']),pick(p['historical'])
    comparable=base['feasible_masks']==hist['feasible_masks']==c['seeds']
    delta=base['mean_errors']-hist['mean_errors'] if comparable else None
    relative=delta/base['mean_errors'] if comparable and base['mean_errors']>0 else None
    passed=comparable and delta>=1 and relative is not None and relative>=.1
    decision=dict(status='INTERNAL_DESCRIPTIVE_RULE_PASS' if passed else 'INTERNAL_DESCRIPTIVE_RULE_FAIL',
        dataset=dataset,is_predeclared_primary_dataset=dataset==p['dataset'],base=base,historical=hist,
        absolute_reduction=delta,relative_reduction=relative,all_masks_comparable=comparable,
        original_failures_overturned=False,finite_sample_guarantee=False)
    for name,data in [('metrics',rows),('raw_errors',raw),('thresholds',thresholds),('summary',summary),('decision',decision)]:
        save(target/(name+'.json'),data)
    np.savez_compressed(P/f'{dataset}_acceptance.npz',accepted=np.stack(masks_out))
    np.savez_compressed(P/f'{dataset}_historical_cache.npz',query_path=d['qpath'],winner=old.astype(np.int32),su=su[0].astype(np.float64))
    files=[target/(name+'.json') for name in ['metrics','raw_errors','thresholds','summary','decision']]+[P/f'{dataset}_acceptance.npz']
    save(target/'evaluation.json',dict(status='EXECUTED_PENDING_REVIEW',rows=len(rows),thresholds=len(thresholds),
        historical_cache_bytes=(P/f'{dataset}_historical_cache.npz').stat().st_size,
        historical_payload_bytes_without_query_mapping=int(old.astype(np.int32).nbytes+su[0].nbytes),
        cache_scope='All calibration and evaluation queries, includes stored query-path mapping. Requires same-query history; no system memory savings claim.',
        output_sha256={p.relative_to(R).as_posix():sha(p) for p in files}))
    print(json.dumps(decision),flush=True)


def main():
    check_freeze()
    c=config()
    for dataset in c['datasets']:
        inf=json.loads((O/dataset/'inference.json').read_text())
        assert inf['status']=='PASS_FROZEN_INFERENCE' and sha(P/f'{dataset}_features.npy')==inf['features_sha256']
        d=load_layout(dataset)
        cases=json.loads((O/f'{dataset}_cases.json').read_text())
        dead=np.unpackbits(d['deleted_packed'],axis=1,count=len(d['gpath'])).astype(bool)
        features=np.load(P/f'{dataset}_features.npy',mmap_mode='r')
        winner,top,nearest=rank_dataset(dataset,d,dead,features)
        evaluate(dataset,c,d,dead,cases,winner,top,nearest)


if __name__=='__main__':
    main()
