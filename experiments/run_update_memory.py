"""Frozen update-memory ablation. Every representation/threshold set before scoring."""
import csv
import hashlib
import json
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]
O=R/'results/A-UPDATE-001'
REQUIRED=['configs/A-UPDATE-001.json','experiments/run_update_memory.py','experiments/review_update_memory.py']
def read(p): return json.loads((R/p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256((R/p).read_bytes()).hexdigest()
def save(p,x): (O/p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')

def encode(g,bits):
    m=2**(bits-1)-1
    scale=np.max(np.abs(g),axis=1).astype(np.float32)/np.float32(m)
    assert np.all(scale>0)
    codes=np.rint(g/scale[:,None]).clip(-m,m).astype(np.int8)
    if bits==8:
        payload=codes.copy(); restored=payload
    else:
        u=(codes.astype(np.int16)+7).astype(np.uint8)
        payload=u[:,0::2] | (u[:,1::2]<<4)
        restored=np.empty_like(codes)
        restored[:,0::2]=(payload&15).astype(np.int8)-7
        restored[:,1::2]=(payload>>4).astype(np.int8)-7
    assert np.array_equal(restored,codes)
    decoded=restored.astype(np.float32)*scale[:,None]
    residual=np.linalg.norm(g.astype(np.float64)-decoded.astype(np.float64),axis=1)
    upper=np.nextafter(residual.astype(np.float32),np.float32(np.inf))
    assert np.all(upper.astype(float)>=residual)
    return payload,scale,decoded,upper

def storage(g,gid,active,deleted,method,enc):
    # Actual serializable arrays, not merely theoretical code length. No ANN.
    header=struct.pack('<6I',0x55504431,1,1536,len(active),len(deleted) if method.startswith('history') else 0,0)
    buf=header+g[active].astype('<f4').tobytes()+gid[active].astype('<u4').tobytes()
    per=0; scratch=0
    if method=='history_fp32':
        per=6148;buf+=g[deleted].astype('<f4').tobytes()+gid[deleted].astype('<u4').tobytes()
    elif method.startswith('history_sq'):
        bits=8 if 'sq8' in method else 4
        payload,scale,decoded,err=enc[bits]
        per=payload.shape[1]+8
        buf+=payload[deleted].tobytes()+scale[deleted].astype('<f4').tobytes()+gid[deleted].astype('<u4').tobytes()
        if method.endswith('bound'):
            per+=4;buf+=err[deleted].astype('<f4').tobytes()
        scratch=len(deleted)*1536*4
    expected=24+len(active)*6148+len(deleted)*per
    assert len(buf)==expected
    assert struct.unpack('<6I',buf[:24])[2]==1536
    return {'serialized_index_bytes':len(buf),'retired_bytes_per_item':per,
            'decode_float32_scratch_bytes':scratch,'serialization':'in_memory_real_bytes_not_latency_benchmark'}

def metric(labels,correct,supported,a):
    n=int(a.sum());e=int((a&~correct).sum());c=n-e
    return {'answered':n,'errors':e,'correct_answers':c,'answered_identities':len(set(labels[a])),
            'answered_error':None if n==0 else e/n,'wrong_return_rate':e/len(a),
            'correct_return_rate':c/len(a),'unsupported_answers':int((a&~supported).sum()),
            'unsupported_queries':int((~supported).sum()),
            'useful_old_rule':bool(n and e*10<=n and len(set(labels[a]))>=10)}

def main():
    git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)]
    for p in REQUIRED:
        subprocess.run([*git,'ls-files','--error-unmatch',p],check=True,capture_output=True)
        subprocess.run([*git,'diff','HEAD','--exit-code','--',p],check=True,capture_output=True)
    c=read(REQUIRED[0]); layout=read(c['layout']);old=read(c['old_thresholds'])['thresholds']
    fit={(x['case_id'],x['policy']):x['threshold'] for x in read(c['recalibrated_thresholds'])['fits']}
    for p,h in read('manifests/IDENTITY_INFERENCE_MANIFEST.json')['outputs_sha256'].items(): assert sha(p)==h,p
    O.mkdir(parents=True,exist_ok=True)
    with (O/'STARTED.json').open('x',encoding='utf8') as f: json.dump({'utc':datetime.now(timezone.utc).isoformat()},f)
    start=time.perf_counter()
    with np.load(R/c['features'],allow_pickle=False) as f:
        qf=f['query_f'];gf=f['gallery_f'];gid=f['gallery_id'];qid=f['query_id'];identity=f['query_identity'];role=f['query_role'];height=f['query_height']
    with np.load(R/c['rankings'],allow_pickle=False) as f: scores=f['scores']
    enc={b:encode(gf,b) for b in (8,4)}
    approx={b:qf@enc[b][2].T for b in (8,4)}
    qnorm=np.linalg.norm(qf.astype(np.float64),axis=1)
    bounds={b:approx[b].astype(float)+qnorm[:,None]*enc[b][3][None,:]+1e-6 for b in (8,4)}
    assert all(np.all(bounds[b]>=scores) for b in (8,4))
    metrics=[];queryrows=[];byte_rows=[]
    for case in layout['cases']:
        dead=np.flatnonzero(np.isin(gid,case['removed'])); active=np.flatnonzero(~np.isin(gid,case['removed']))
        order=np.argsort(-scores[:,active],axis=1,kind='stable');winner=active[order[:,0]]
        current_top=scores[np.arange(len(qf)),winner];second=scores[np.arange(len(qf)),active[order[:,1]]]
        exact_dead=np.max(scores[:,dead],axis=1) if len(dead) else np.full(len(qf),-np.inf)
        # Exact ties are ordered by the original ascending gallery ID.
        oldwinner=np.argmax(scores,axis=1); valid=~np.isin(oldwinner,dead)
        confs={'current_frozen':(current_top,current_top-second,np.ones(len(qf),bool)),
               'current_recalibrated':(current_top,current_top-second,np.ones(len(qf),bool)),
               'history_fp32':(current_top,current_top-np.maximum(second,exact_dead),valid)}
        for method in c['methods'][3:]:
            b=8 if 'sq8' in method else 4
            s=bounds[b] if method.endswith('bound') else approx[b]
            strongest=np.max(s[:,dead],axis=1) if len(dead) else np.full(len(qf),-np.inf)
            confs[method]=(current_top,current_top-np.maximum(second,strongest),current_top>strongest)
        byte_by_method={m:storage(gf,gid.astype(int),active,dead,m,enc) for m in c['methods']}
        meta={k:case[k] for k in ('case_id','seed','fraction','arm')}
        byte_rows.extend({**meta,'method':m,**v} for m,v in byte_by_method.items())
        for h in c['heights']:
            sel=np.flatnonzero((role=='evaluation')&(height==h));assert len(sel)==48
            lab=identity[sel]; correct=gid[winner[sel]]==lab;supported=~np.isin(lab,case['removed'])
            for policy,pi in [('top1',0),('margin',1)]:
                for method in c['methods']:
                    tau=fit[case['case_id'],policy] if method=='current_recalibrated' else old[policy]
                    answer=np.zeros(48,bool) if tau is None else (confs[method][pi][sel]>=tau)&confs[method][2][sel]
                    metrics.append({**meta,'height':h,'policy':policy,'method':method,'threshold':tau,
                                    **metric(lab,correct,supported,answer),**byte_by_method[method]})
                    for k,j in enumerate(sel):
                        queryrows.append({**meta,'height':h,'policy':policy,'method':method,'query_id':str(qid[j]),
                                          'identity':str(lab[k]),'pred':str(gid[winner[j]]),'supported':int(supported[k]),
                                          'correct':int(correct[k]),'accepted':int(answer[k])})
    save('metrics.json',metrics);save('index_bytes.json',byte_rows)
    with (O/'per_query.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=queryrows[0].keys());w.writeheader();w.writerows(queryrows)
    summary=[]
    for h in c['heights']:
        for fraction in (.25,.5):
            for arm in ('target','control'):
                for policy in c['policies']:
                    for method in c['methods']:
                        rows=[m for m in metrics if (m['height'],m['fraction'],m['arm'],m['policy'],m['method'])==(h,fraction,arm,policy,method)]
                        assert len(rows)==20
                        out={'height':h,'fraction':fraction,'arm':arm,'policy':policy,'method':method,'masks':20,
                             'nonempty_masks':sum(x['answered']>0 for x in rows),'useful_old_rule_masks':sum(x['useful_old_rule'] for x in rows)}
                        for key in ('answered','errors','correct_answers','answered_identities','answered_error','wrong_return_rate','correct_return_rate','unsupported_answers','serialized_index_bytes'):
                            values=[x[key] for x in rows if x[key] is not None];out['mean_'+key]=float(np.mean(values)) if values else None
                        summary.append(out)
    save('summary.json',summary)
    p=c['primary'];match=lambda x:x['height']==p['height'] and x['fraction']==p['fraction'] and x['arm']==p['arm'] and x['policy']==p['policy']
    a=next(x for x in summary if match(x) and x['method']==p['reference']);b=next(x for x in summary if match(x) and x['method']==p['method'])
    reduction=1-b['mean_errors']/a['mean_errors'];retention=b['mean_correct_answers']/a['mean_correct_answers'];byte_reduction=1-1548/6148
    decision={'wrong_return_relative_reduction':reduction,'correct_answer_retention':retention,'retired_representation_byte_reduction':byte_reduction,
              'engineering_primary_pass':bool(reduction>=p['wrong_return_relative_reduction_min'] and retention>=p['correct_answer_retention_min'] and byte_reduction>=p['retired_representation_byte_reduction_vs_fp32_min']),
              'innovation_validated':False,'risk_guarantee':False,'independent_confirmation':False}
    save('decision.json',decision)
    inp=REQUIRED+[c[k] for k in ('layout','features','rankings','old_thresholds','recalibrated_thresholds')]
    execution={'exp_id':c['exp_id'],'execution_commit':subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip(),
               'input_sha256':{p:sha(p) for p in inp},'runtime_seconds':time.perf_counter()-start,'gpu_seconds':0,
               'metric_rows':len(metrics),'query_rows':len(queryrows),'thresholds_fitted':0,'new_images':0,
               'output_sha256':{f'results/A-UPDATE-001/{p}':sha(f'results/A-UPDATE-001/{p}') for p in ('metrics.json','index_bytes.json','per_query.csv','summary.json','decision.json')},
               'status':'EXECUTED_PENDING_INDEPENDENT_REPLAY'}
    save('execution.json',execution);print(json.dumps({'execution':execution['runtime_seconds'],**decision}))

if __name__=='__main__':main()
