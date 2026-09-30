"""Strict full-state ViT-S + 512-D BN-neck adapter for Bancie's checkpoint.

Use released baseline test.py semantics, not the model-card illustrative
backbone-only/0.5-normalization snippet. No key is silently omitted.
"""
import hashlib,json,io
from pathlib import Path
import torch,timm
from torch import nn
from PIL import Image
from torchvision import transforms
R=Path(__file__).resolve().parents[1]
class Container(nn.Module):
 def __init__(self,**children):
  super().__init__()
  for k,v in children.items():self.add_module(k,v)
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.backbone=Container(backbone=timm.create_model('vit_small_patch16_224',pretrained=False,num_classes=1000))
  self.head=Container(head=Container(classifier=Container(add_block=nn.Sequential(nn.Linear(384,512),nn.BatchNorm1d(512),nn.Dropout(.5)),classifier=nn.Sequential(nn.Linear(512,2256)))))
 def raw(self,x):
  # timm 0.5.4 forward_features returns the normalized CLS vector. The
  # original modified timm returns all tokens, then SingleBranch takes [:,0].
  cls=self.backbone.backbone.forward_features(x);assert cls.ndim==2 and cls.shape[1]==384
  return self.head.head.classifier.add_block(cls)
 def manual_raw(self,x):
  m=self.backbone.backbone;x=m.patch_embed(x);x=torch.cat((m.cls_token.expand(x.shape[0],-1,-1),x),dim=1);x=m.pos_drop(x+m.pos_embed);x=m.blocks(x);x=m.norm(x);return self.head.head.classifier.add_block(x[:,0])
def build():
 p=R/'checkpoints/denseuav/bancie_vits.pth';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='47b585e045d08a7cdae02b949ff051ccd81e48f532d2ecad95b18c449173908e'
 s=torch.load(p,weights_only=True,map_location='cpu');m=Model();assert set(s)==set(m.state_dict());m.load_state_dict(s,strict=True);assert all(torch.equal(v,s[k]) for k,v in m.state_dict().items());assert all(torch.isfinite(v).all() for v in s.values());m.eval().requires_grad_(False)
 return m,dict(strict_keys=len(s),all_values_equal=True,checkpoint_sha256=h,timm=timm.__version__,provenance='Third-party DenseUAV-trained model; not authenticated author checkpoint; published training split claim not independently certified',features='512-D trained BN neck; original+horizontal flip raw-feature sum L2 normalized; FP32',preprocessing='Pinned baseline test.py: PIL bicubic 224, ImageNet normalization; model card backbone-only/0.5 example not used')
transform=transforms.Compose([transforms.Resize((224,224),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def tensor(row):
 b=(R/row['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']
 with Image.open(io.BytesIO(b)) as im:return transform(im.convert('RGB'))
def descriptor(model,x):return torch.nn.functional.normalize(model.raw(x)+model.raw(x.flip(3)),dim=1)
if __name__=='__main__':
 torch.set_num_threads(4);m,d=build();x=torch.linspace(-2,2,2*3*224*224).reshape(2,3,224,224)
 with torch.inference_mode():
  a=m.raw(x);b=m.manual_raw(x);assert torch.equal(a,b);d['synthetic_structural_check_only']=dict(manual_all_tokens_cls_equals_timm_cls=True,shape=list(a.shape))
 p=R/'results/DENSEUAV-BASELINE-INTAKE-001/strict_load.json';p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
