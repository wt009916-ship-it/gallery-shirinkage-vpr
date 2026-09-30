"""Narrow inference adapter for the pinned official CAMP source.

No source edits. Only disable initializer download before strict checkpoint load.
The official evaluation uses normalized gap_feature=model(img)[-2], not the
position-added spatial tensor, and does not use flip augmentation.
"""
import hashlib,importlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np,torch
R=Path(__file__).resolve().parents[1]
def build():
 source=json.loads((R/'manifests/CAMP_SOURCE.json').read_text())
 for x in source['files']:assert hashlib.sha256((R/x['path']).read_bytes()).hexdigest()==x['sha256']
 sys.path.insert(0,str(R/'sources/CAMP'))
 m=importlib.import_module('sample4geo.hand_convnext.ConvNext.make_model');factory=m.create_model
 def local_factory(name,**kwargs):
  assert name=='convnext_base';kwargs['pretrained']=False;return factory(name,**kwargs)
 m.create_model=local_factory
 from sample4geo.hand_convnext.model import make_model
 opt=SimpleNamespace(views=2,nclasses=701,block=2,triplet_loss=.3,resnet=False,pos_scale=.6,if_learn_ECE_weights=True,learn_weight_D_D=0.,learn_weight_S_S=0.,learn_weight_D_fine_D_fine=.5,learn_weight_D_fine_S_fine=1.,learn_weight_S_fine_S_fine=0.)
 model=make_model(opt);record=json.loads((R/'results/CAMP-INTAKE-001/checkpoint.json').read_text());p=R/record['checkpoint'];assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'];state=torch.load(p,map_location='cpu',weights_only=True)
 missing=set(model.state_dict())-set(state);unexpected=set(state)-set(model.state_dict())
 assert missing=={'model_1.pos_scale'} and not unexpected
 # Current source registers this spatial-output-only scalar; the released
 # checkpoint does not contain it. Keep its declared default as a non-state
 # constant. It cannot affect the gap output consumed by official predict().
 del model.model_1._parameters['pos_scale'];model.model_1.pos_scale=.6
 model.load_state_dict(state,strict=True);assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items());model.eval().requires_grad_(False)
 return model,{'strict_keys':len(state),'all_values_equal':True,'checkpoint_sha256':record['sha256'],'source_commit':source['commit'],'initializer_download_disabled':True,'timm':importlib.import_module('timm').__version__,'source_checkpoint_mismatch':{'missing_original':['model_1.pos_scale'],'unexpected':[],'treatment':'Source-default .6 kept as non-state constant only on unused spatial output; inference equivalence tested separately. Not complete original-training reproduction.'}}
def tensor(row):
 b=(R/row['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256'];a=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR);assert a is not None;a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB);a=cv2.resize(a,(384,384),interpolation=cv2.INTER_LINEAR_EXACT).astype(np.float32);a-=np.array([.485,.456,.406],np.float32)*255;a*=1/(np.array([.229,.224,.225],np.float32)*255)
 return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))
def descriptor(model,x):return torch.nn.functional.normalize(model(x)[-2],dim=-1)
if __name__=='__main__':
 torch.set_num_threads(4);model,record=build();print(json.dumps(record));(R/'results/CAMP-INTAKE-001/strict_load.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
