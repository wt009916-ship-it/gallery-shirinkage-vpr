"""Dependency-free aggregate reanalysis. Does not download data or run inference."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

GROUP=['source','dataset','model','scope','condition','fold','radius_m','arm','fraction','regime']
CONTRASTS={'candidate_check':('current_su','current_su_guard'),
           'historical_score':('current_su_guard','old_su_guard'),
           'total_history':('current_su','old_su_guard')}


def main():
    root=Path(__file__).resolve().parents[1]
    a=argparse.ArgumentParser()
    a.add_argument('--input-dir',type=Path,default=root/'results/paired_evidence')
    a.add_argument('--output-dir',type=Path,default=root/'derived')
    args=a.parse_args();d=args.input_dir
    rows=list(csv.DictReader((d/'normalized.csv').open(encoding='utf8',newline='')))
    buckets=defaultdict(dict)
    for z in rows:
        key=tuple(z[k] for k in GROUP)
        case=(z['case_id'],z['method'])
        assert case not in buckets[key]
        buckets[key][case]=z
        if z['feasible']=='True':
            assert int(z['answered'])==int(z['errors'])+int(z['correct'])
            if int(z['answered']):assert abs(float(z['risk'])-int(z['errors'])/int(z['answered']))<1e-12
        if int(z['answered'])==0:assert z['risk']==''
    expected=json.loads((d/'paired_summary.json').read_text(encoding='utf8'))
    primary=[];checks=len(rows)
    for s in expected:
        key=tuple('' if s[k] is None else str(s[k]) for k in GROUP)
        group=buckets[key];lm,rm=CONTRASTS[s['contrast']]
        lc={c for c,m in group if m==lm};rc={c for c,m in group if m==rm}
        assert lc==rc
        valid=[c for c in sorted(lc) if group[(c,lm)]['feasible']=='True' and group[(c,rm)]['feasible']=='True']
        used=[c for c in valid if int(group[(c,lm)]['answered'])>0 and int(group[(c,rm)]['answered'])>0]
        assert s['cases']==len(lc) and s['paired_feasible']==len(valid) and s['paired_nonempty']==len(used)
        assert set(s['paired_case_ids'])==set(used) and set(s['excluded_case_ids'])==lc-set(used)
        recomputed={}
        for side,method in [('left',lm),('right',rm)]:
            for k in ['answered','errors','correct','risk','created','inherited']:
                rawkey=k if k not in ['created','inherited'] else k+'_errors'
                value=mean(float(group[(c,method)][rawkey]) for c in used) if used else None
                name=side+'_'+k;recomputed[name]=value
                assert (s[name] is None and value is None) or abs(s[name]-value)<1e-10
        delta=None if not used else recomputed['right_errors']-recomputed['left_errors']
        assert (delta is None and s['error_delta_right_minus_left'] is None) or abs(delta-s['error_delta_right_minus_left'])<1e-10
        checks+=1
        if (s['scope']=='pooled' and s['condition'] in ['same','plain_same'] and s['radius_m'] in [None,100]
            and s['arm'] in ['target','spatial'] and s['fraction']==.25 and s['regime']=='budget50'):
            primary.append({k:s[k] for k in GROUP+['contrast']} | dict(cases=len(lc),paired_nonempty=len(used),**recomputed,error_delta_right_minus_left=delta))
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    with (out/'paired_primary.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(primary[0]));w.writeheader();w.writerows(primary)
    report=dict(status='PASS_REGENERATED_FROM_EXPORTED_AGGREGATES',checks=checks,source_rows=len(rows),groups=len(expected),primary_rows=len(primary),
        inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [d/'normalized.csv',d/'paired_summary.json']},
        scope='Recomputes case intersections, counts and primary means from aggregate records. Not image inference, raw-label verification or independent scientific confirmation.')
    (out/'reproduction_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report))


if __name__=='__main__':main()
