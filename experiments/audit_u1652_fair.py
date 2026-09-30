"""Posthoc fixed-decision serialized replay and fair-information budget audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,struct,subprocess,time
from pathlib import Path
import numpy as np
from run_update_memory import encode
R=Path(__file__).resolve().parents[1];O=R/'results/A-U1652-FAIR-001'
METHODS=['current_frozen','history_fp32','history_sq8_bound','history_sq4_bound']
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def save(p,x):(O/p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def serialize(g,gid,active,dead,method,enc):
    dd=dead if method!='current_frozen' else np.array([],int)
    parts=[struct.pack('<6I',0x55504431,1,g.shape[1],len(active),len(dd),0),g[active].astype('<f4').tobytes(),gid[active].astype('<u4').tobytes()]
    if method=='history_fp32':parts += [g[dd].astype('<f4').tobytes(),gid[dd].astype('<u4').tobytes()]
    elif method.startswith('history_sq'):
        bits=8 if 'sq8' in method else 4;co,sc,_,er=enc[bits];parts += [co[dd].tobytes(),sc[dd].astype('<f4').tobytes(),gid[dd].astype('<u4').tobytes(),er[dd].astype('<f4').tobytes()]
    return b''.join(parts)
class Index:
    def __init__(self,buf,method):
        self.buf=buf;self.method=method;magic,ver,self.dim,na,self.nd,_=struct.unpack('<6I',buf[:24]);assert (magic,ver)==(0x55504431,1);offset=24
        def take(dtype,shape):
            nonlocal offset
            v=np.frombuffer(buf,dtype=dtype,count=int(np.prod(shape)),offset=offset).reshape(shape);offset+=v.nbytes;return v
        self.g=take('<f4',(na,self.dim));self.gid=take('<u4',(na,))
        if method=='history_fp32':self.ret=take('<f4',(self.nd,self.dim));self.rid=take('<u4',(self.nd,))
        elif method.startswith('history_sq'):
            self.bits=8 if 'sq8' in method else 4;self.code=take('i1' if self.bits==8 else 'u1',(self.nd,self.dim if self.bits==8 else self.dim//2));self.scale=take('<f4',(self.nd,));self.rid=take('<u4',(self.nd,));self.err=take('<f4',(self.nd,))
        assert offset==len(buf)
    def query(self,q,tau):
        s=self.g@q;order=np.argsort(-s,kind='stable');j=int(order[0]);v=float(s[j]);second=float(s[order[1]]);retmax=-np.inf;valid=True
        if self.method=='history_fp32' and self.nd:
            rs=self.ret@q;k=int(np.argmax(rs));retmax=float(rs[k]);valid=v>retmax or (v==retmax and self.gid[j]<self.rid[k])
        elif self.method.startswith('history_sq') and self.nd:
            norm=np.linalg.norm(q.astype(float))
            for st in range(0,self.nd,16):
                co=self.code[st:st+16]
                if self.bits==8:dec=co.astype(np.float32)
                else:
                    dec=np.empty((len(co),self.dim),np.float32);dec[:,::2]=(co&15).astype(np.int8)-7;dec[:,1::2]=(co>>4).astype(np.int8)-7
                dec*=self.scale[st:st+16,None];upper=(dec@q).astype(float)+norm*self.err[st:st+16].astype(float)+1e-6;retmax=max(retmax,float(upper.max()))
            valid=v>retmax
        conf=(v,v-max(second,retmax))
        return int(self.gid[j]),*[bool(valid and tau[p] is not None and conf[k]>=tau[p]) for k,p in enumerate(('top1','margin'))]
def main():
    t=time.perf_counter();O.mkdir(parents=True,exist_ok=True);git=['git','-c','safe.directory='+R.as_posix(),'-C',str(R)]
    for p in ('configs/A-U1652-FAIR-001.json','experiments/audit_u1652_fair.py'):
        subprocess.run([*git,'ls-files','--error-unmatch',p],check=True,capture_output=True);subprocess.run([*git,'diff','HEAD','--exit-code','--',p],check=True,capture_output=True)
    commit=subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip()
    with (O/'STARTED.json').open('x') as f:json.dump({'epoch':time.time(),'commit':commit,'posthoc':True},f)
    assert read('results/A-U1652-001/output_audit.json')['status'].startswith('PASS')
    with np.load(R/'results/U1652-REF-001/features.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    ev=d['query_role']=='evaluation';q=d['query_f'][ev];g=d['gallery_f'];gid=d['gallery_id'].astype(np.uint32);enc={b:encode(g,b) for b in (8,4)};tau=read('results/A-U1652-001/thresholds.json')['thresholds'];cases=read('results/A-U1652-001/layout.json');metrics=read('results/A-U1652-001/metrics.json');mm={(r['case_id'],r['method']):r for r in metrics}
    with np.load(R/'results/A-U1652-001/decisions.npz',allow_pickle=False) as z:frozen=z['accepted'];pred=z['predictions']
    timings=[];costs=[];checked=0
    for ci,case in enumerate(cases):
        dead=np.flatnonzero(np.isin(d['gallery_id'],case['removed']));active=np.setdiff1d(np.arange(len(g)),dead);indexes={m:Index(serialize(g,gid,active,dead,m,enc),m) for m in METHODS}
        for mi,(method,index) in enumerate(indexes.items()):
            rr=np.array([index.query(v,tau) for v in q]);assert np.array_equal(rr[:,0],gid[pred[ci]]),(case['case_id'],method,'prediction');assert np.array_equal(rr[:,1:].astype(bool).T,frozen[ci,mi]),(case['case_id'],method,'acceptance');checked+=len(q)*2
            size=len(index.buf);assert size==mm[case['case_id'],method]['serialized_index_bytes'];costs.append({'case_id':case['case_id'],'method':method,'serialized_index_bytes':size,'decoded_float32_block_bytes':min(16,len(dead))*g.shape[1]*4 if method.startswith('history_sq') else 0})
        if case['case_id']=='full' or case['seed']==3407:
            samples={m:[] for m in METHODS}
            for rep in range(-1,5):
                rotation=(rep+1)%4
                for method in METHODS[rotation:]+METHODS[:rotation]:
                    for v in q:
                        ts=time.perf_counter_ns();indexes[method].query(v,tau);elapsed=(time.perf_counter_ns()-ts)/1e6
                        if rep>=0:samples[method].append(elapsed)
            for method,v in samples.items():timings.append({'case_id':case['case_id'],'method':method,'n':len(v),'median_ms':float(np.median(v)),'p95_ms':float(np.quantile(v,.95))})
        if ci%10==0:print(json.dumps({'cases_checked':ci+1,'total':len(cases),'seconds':round(time.perf_counter()-t,2)}),flush=True)
    # No threshold adjustment, significance claim, or duplicate-based relabeling.
    comparisons=[];cm=read('results/A-U1652-CAL-001/metrics.json')
    for arm,fraction in [('target',.25),('control',.25),('target',.5)]:
        for policy in ('top1','margin'):
            for method in METHODS+['current_recalibrated','history_fp32_recalibrated','history_sq8_bound_recalibrated']:
                rows=[r for r in (cm if 'recalibrated' in method else metrics) if r['arm']==arm and r['fraction']==fraction and r['policy']==policy and r['method']==method];assert len(rows)==20
                risks=[r['answered_error'] for r in rows if r['answered_error'] is not None]
                comparisons.append({'arm':arm,'fraction':fraction,'policy':policy,'method':method,'nonempty_masks':len(risks),'risk_exceedances':sum(v>.1 for v in risks),'useful_masks':sum(r['answered']>0 and r['answered_error']<=.1 and r['accepted_identities']>=50 and r['accepted_blocks']>=5 for r in rows),'mean_correct':float(np.mean([r['correct_answers'] for r in rows])),'mean_wrong':float(np.mean([r['errors'] for r in rows])),'mean_risk':float(np.mean(risks)) if risks else None,'worst_risk':max(risks) if risks else None})
    duplicate=[];scores=d['query_f']@g.T;order=np.argsort(-scores,axis=1,kind='stable')
    for group in read('results/U1652-INTAKE-001/output_audit.json')['duplicate_pixel_groups']:
        if not all(x['role']=='gallery' for x in group):continue
        labels=[x['identity'] for x in group];ix=np.isin(d['query_identity'],labels);tied=np.isin(d['gallery_id'][order[:,0]],labels)&(scores[np.arange(len(scores)),order[:,0]]==scores[np.arange(len(scores)),order[:,1]])
        duplicate.append({'gallery_ids':labels,'query_count':int(ix.sum()),'query_roles':{role:int((ix&(d['query_role']==role)).sum()) for role in ('calibration','evaluation')},'queries_with_duplicate_top1_tie':int(tied.sum()),'evaluation_tie_queries':int((tied&ev).sum())})
    shared={'checkpoint_bytes':(R/read('results/U1652-INTAKE-001/reference_checkpoint.json')['checkpoint']).stat().st_size,'calibration_descriptors_fp32_bytes':int(d['query_f'][~ev].nbytes),'calibration_identity_uint32_bytes':int((~ev).sum()*4),'calibration_block_uint32_bytes':int((~ev).sum()*4),'calibration_labeling_cost':'not measured; existing official identity labels; both recalibration competitors use same1251queries','fixed_label_snapshot_required_at_inference':False,'encoder_time_shared_and_excluded_from_index_latency':True}
    save('timing.json',timings);save('index_costs.json',costs);save('comparisons.json',comparisons);save('duplicate_diagnostic.json',duplicate);save('shared_costs.json',shared)
    inputs=['configs/A-U1652-FAIR-001.json','experiments/audit_u1652_fair.py','experiments/run_update_memory.py','results/U1652-REF-001/features.npz','results/A-U1652-001/decisions.npz','results/A-U1652-001/thresholds.json','results/A-U1652-001/metrics.json','results/A-U1652-CAL-001/metrics.json']
    ex={'status':'PASS_SERIALIZED_ALL_FIXED_DECISIONS','posthoc':True,'execution_commit':commit,'runtime_seconds':time.perf_counter()-t,'checked_decisions':checked,'timing_cells':len(timings),'gpu_seconds':0,'input_sha256':{p:sha(p) for p in inputs},'output_sha256':{f'results/A-U1652-FAIR-001/{p}':sha(f'results/A-U1652-FAIR-001/{p}') for p in ('timing.json','index_costs.json','comparisons.json','duplicate_diagnostic.json','shared_costs.json')},'limits':['small linear CPU index, no large ANN claim','on-demand16row FP32 decode size excludes other arrays and process memory','cached features for experiment remain resident; not counted as deployment index','repeated masks and query frames dependent','no inference-time calibration labels required for already fitted thresholds','descriptive comparison only; no novelty certification']};save('execution.json',ex);print(json.dumps(ex));print(json.dumps(comparisons))
if __name__=='__main__':main()
