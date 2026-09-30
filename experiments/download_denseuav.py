"""Resume bounded, pinned ZIP-member downloads; CRC, SHA and decode audit."""
import concurrent.futures,datetime,hashlib,json,struct,time,urllib.request,urllib.parse,zlib,threading,http.client,argparse
from pathlib import Path,PurePosixPath
import cv2,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'results/A-DENSEUAV-EXTERNAL-001';C=R/'configs/A-DENSEUAV-EXTERNAL-001.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();assert 1<=args.workers<=12
 c=json.loads(C.read_text());O.mkdir(exist_ok=True);src=c['source'];url=f"https://huggingface.co/datasets/{src['repo']}/resolve/{src['revision']}/{src['archive']}";assert sha(R/'results/DENSEUAV-INTAKE-001/archive_index.json')==src['index_sha256'];manifest=O/'download_rows.jsonl';lock=threading.Lock();net=[0];began=time.monotonic();old={};local=threading.local()
 # Keep the short-lived public CDN URL in memory only. Its signature is never logged.
 with urllib.request.urlopen(urllib.request.Request(url+'?range_id=0-31',headers={'Range':'bytes=0-31'}),timeout=45) as f:
  assert f.status==206;f.read(33);cdn=urllib.parse.urlparse(f.url)
 assert cdn.scheme=='https' and cdn.hostname.endswith('.hf.co')
 target=cdn.path+'?'+cdn.query
 if manifest.exists():
  for line in manifest.read_text().splitlines():
   x=json.loads(line);old[x['member']]=x
 def one(x):
  name=x['name'];rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
  p=R/'data/denseuav_h90'/Path(*rel.parts)
  if name in old and p.is_file() and sha(p)==old[name]['sha256']:return old[name]
  a=x['offset'];b=min(src['archive_bytes']-1,a+x['compressed_bytes']+1023);n=b-a+1
  for attempt in range(4):
   try:
    if not hasattr(local,'conn'):local.conn=http.client.HTTPSConnection(cdn.hostname,timeout=45)
    local.conn.request('GET',target,headers={'Range':f'bytes={a}-{b}','User-Agent':'Research-Data-Audit'})
    f=local.conn.getresponse();assert f.status==206 and f.getheader('Content-Range')==f'bytes {a}-{b}/{src["archive_bytes"]}'
    blob=f.read(n+1);assert len(blob)==n
    with lock:
     net[0]+=len(blob);assert net[0]<=c['budget']['download_ceiling_bytes']
    h=struct.unpack('<4s5H3I2H',blob[:30]);assert h[0]==b'PK\x03\x04' and h[3]==x['compression'] and not h[2]&1
    fn,ex=h[-2:];assert blob[30:30+fn].decode('utf8')==name
    start=30+fn+ex;assert start+x['compressed_bytes']<=len(blob)
    compressed=blob[start:start+x['compressed_bytes']];raw=zlib.decompress(compressed,-15) if x['compression']==8 else compressed
    assert len(raw)==x['bytes'] and f'{zlib.crc32(raw)&0xffffffff:08x}'==x['crc32'];im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR);assert im is not None and im.ndim==3
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part');tmp.write_bytes(raw);tmp.replace(p)
    row=dict(member=name,path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),shape=list(im.shape),bytes=len(raw),identity=rel.parts[-2],view='drone' if 'query_drone' in name else 'satellite',variant='old' if '_old' in name else 'plain')
    with lock:
     with manifest.open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
    return row
   except Exception:
    if hasattr(local,'conn'):local.conn.close();del local.conn
    if attempt==3:raise
    time.sleep(2*(attempt+1))
 rows=[];failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  jobs={pool.submit(one,x):x['name'] for x in c['members']}
  for job in concurrent.futures.as_completed(jobs):
   try:rows.append(job.result())
   except Exception as e:failures.append(dict(member=jobs[job],error=repr(e)))
   n=len(rows)+len(failures)
   if n%100==0:print(json.dumps(dict(completed=n,total=len(jobs),failures=len(failures),network_mb=round(net[0]/1e6,1),seconds=round(time.monotonic()-began))),flush=True)
 result=dict(status='PASS_DOWNLOAD' if not failures else 'INCOMPLETE',files=sorted(rows,key=lambda r:r['member']),failures=failures,config_sha256=sha(C),source=src,network_bytes_this_run=net[0],elapsed_seconds=time.monotonic()-began,transfer_workers=args.workers,transport_note='Connection reuse and worker count are operational only; no scientific protocol changes.',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (O/'download.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='files'}))
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
