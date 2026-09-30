"""Public metadata-only intake, bounded ZIP ranges; no image scoring."""
import datetime,hashlib,io,json,re,time,urllib.request,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'results/DENSEUAV-INTAKE-001';O.mkdir(exist_ok=True)
def get(url,limit=4000000,headers=None):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-Data-Audit',**(headers or {})}),timeout=35) as f:
  b=f.read(limit+1);assert len(b)<=limit;return b,f.status,dict(f.headers)
class Remote(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.bytes=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else self.size+n;assert 0<=self.pos<=self.size;return self.pos
 def read(self,n=-1):
  n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
  if not n:return b''
  assert n<=8000000
  a=self.pos;b=a+n-1
  raw,status,h=get(self.url+('&' if '?' in self.url else '?')+f'range_id={a}-{b}',n,{'Range':f'bytes={a}-{b}'})
  assert status==206 and h.get('Content-Range',h.get('content-range'))==f'bytes {a}-{b}/{self.size}',(status,h.get('Content-Range'));self.bytes+=len(raw);self.pos+=len(raw);assert self.bytes<20000000;return raw
def main():
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');log={'started':stamp,'images_downloaded':False,'model_scored':False}
 try:
  api='https://huggingface.co/api/datasets/Dmmm997/DenseUAV/tree/main';raw,_,_=get(api);(O/'hf_tree.json').write_bytes(raw);tree=json.loads(raw);entry=next(x for x in tree if x['path']=='DenseUAV.zip');log['archive_bytes']=entry['size']
  url='https://huggingface.co/datasets/Dmmm997/DenseUAV/resolve/main/DenseUAV.zip';remote=Remote(url,entry['size'])
  with zipfile.ZipFile(remote) as z:
   inv=[dict(name=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,crc32=f'{x.CRC:08x}',compression=x.compress_type,offset=x.header_offset) for x in z.infolist()];(O/'archive_index.json').write_text(json.dumps(inv,indent=2),encoding='utf8');meta=[]
   for x in z.infolist():
    if not x.is_dir() and x.file_size<2000000 and x.filename.lower().endswith(('.txt','.csv','.json','.md','.yaml')):
     b=z.read(x);p=O/'metadata'/x.filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);meta.append(dict(member=x.filename,path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
   log.update(status='PASS_METADATA_INTAKE',entries=len(inv),network_bytes=remote.bytes,metadata=meta)
 except Exception as e:log.update(status='FAIL',error=repr(e))
 (O/(stamp+'.json')).write_text(json.dumps(log,indent=2),encoding='utf8');(O/'intake.json').write_text(json.dumps(log,indent=2),encoding='utf8');print(json.dumps(log))
 if log['status']=='FAIL':raise SystemExit(1)
if __name__=='__main__':main()
