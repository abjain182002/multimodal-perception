"""CPU-only Colab transfer of two preselected night cases. No inference.
Run with Drive mounted. Verifies preserved source hashes before reading selected content.
"""
from pathlib import Path
import gc,hashlib,io,json,math,shutil,tarfile,time,zipfile
from datetime import datetime,timezone
BASE=Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation')
META=BASE/'case_followup_metadata_20261028/case_metadata_and_night_candidates.json'
EXPECTED_META='40ebf900154cf8656b0b87c55e9ca3b875908bf972453b0f48e3ddbea7d040c1'
ORIGINAL=BASE/'nuscenes_validation_parts_01_02_verified.zip'
KEYFRAMES=BASE/'queue_v1.0-trainval10_keyframes_verified.zip'
HASHES={str(ORIGINAL):'c89bc342a240d0bde22ef8b9200ebaf6de5b5f91c8a399073c4057d866f39c3a',str(KEYFRAMES):'c1645861b3da66d638387731830393f0e06185c8396719b3f58ec38118a6381f'}
PRED_HASH={'lidar':'15667405cb1af580e6f8082dcc50366560e45335c36f7122ec490f377cd86c63','lidar_camera':'dcd86893985366881b27d3841b64b93a85988158b518f0c9edfa274c291596dd'}
def sha(path,progress=False):
 h=hashlib.sha256();total=0;last=0
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):
   h.update(b);total+=len(b)
   if progress and total-last>=2_000_000_000:print('Hash-checked GB:',round(total/1e9,2),flush=True);last=total
 return h.hexdigest()

def array_rows(stream):
 """Stream a top-level JSON array with bounded buffering; standard library only."""
 text=io.TextIOWrapper(stream,encoding='utf-8');decoder=json.JSONDecoder();buf='';pos=0;started=False;done=False
 while not done:
  while True:
   while pos<len(buf) and buf[pos].isspace():pos+=1
   if not started:
    if pos==len(buf):break
    assert buf[pos]=='[';pos+=1;started=True;continue
   while pos<len(buf) and (buf[pos].isspace() or buf[pos]==','):pos+=1
   if pos==len(buf):break
   if buf[pos]==']':done=True;break
   try:row,end=decoder.raw_decode(buf,pos)
   except json.JSONDecodeError:break
   yield row;pos=end
  if done:break
  chunk=text.read(1024*1024)
  if not chunk:raise ValueError('Incomplete JSON array')
  buf=buf[pos:]+chunk;pos=0
 # The tables are single top-level arrays. Additional non-whitespace is invalid.
 assert not (buf[pos+1:]+text.read()).strip()

def verify_parser():
 fixture=[{'token':str(i),'text':'é\\quote"'} for i in range(100)]
 assert list(array_rows(io.BytesIO(json.dumps(fixture,ensure_ascii=False).encode())))==fixture
 assert list(array_rows(io.BytesIO(b'[]')))==[]
verify_parser()

def select_tables(names,selectors):
 out={}
 with zipfile.ZipFile(ORIGINAL) as z:
  with z.open('metadata/v1.0-trainval_meta.tgz') as stream:
   with tarfile.open(fileobj=stream,mode='r|gz') as tar:
    for member in tar:
     name=Path(member.name).name
     if member.isfile() and name in names:
      print('STREAMING:',name,flush=True);chosen=[]
      for i,row in enumerate(array_rows(tar.extractfile(member)),1):
       if selectors[name](row):chosen.append(row)
       if i%400000==0:print(name,'records scanned:',i,flush=True)
      out[name]=chosen;print('SELECTED:',name,len(chosen),flush=True)
      if set(out)==set(names):break
 assert set(out)==set(names),set(names)-set(out)
 return out

started=time.monotonic();assert sha(META)==EXPECTED_META;selection=json.loads(META.read_text())
tokens=[r['sample_token'] for r in selection['night_candidates']];assert len(tokens)==len(set(tokens))==2
required=set()
for c in selection['night_candidates']:
 inv=c['sensor_inventory'];required.update(inv['camera_files']);required.update(x for x in inv['lidar_files'] if x.startswith('samples/'))
assert len(required)==14
for source in [ORIGINAL,KEYFRAMES]:
 print('VERIFYING:',source.name,flush=True);assert sha(source,True)==HASHES[str(source)],source;print('SOURCE HASH PASS',flush=True)
with zipfile.ZipFile(ORIGINAL) as z:val=json.loads(z.read('preparation/validation_input_manifest.json'))
validation_tokens={r['sample_token'] for r in val};assert len(validation_tokens)==6019 and set(tokens)<=validation_tokens;del val
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');work=Path('/content/night_case_transfer')/stamp;work.mkdir(parents=True)
def put(name,value):
 p=work/name;p.parent.mkdir(parents=True,exist_ok=True)
 if isinstance(value,bytes):p.write_bytes(value)
 else:p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
 return p
put('case_metadata_and_night_candidates.json',META.read_bytes())
with zipfile.ZipFile(KEYFRAMES) as z:
 for rel in sorted(required):put('dataset/'+rel,z.read('dataset/'+rel))
print('EXTRACTED 14 KEYFRAMES FROM HASH-VERIFIED ARCHIVE',flush=True)
first=select_tables({'sample_data.json','calibrated_sensor.json'},{'sample_data.json':lambda r:r['filename'] in required,'calibrated_sensor.json':lambda r:True})
sd=first['sample_data.json'];assert len(sd)==14 and {r['filename'] for r in sd}==required
pose_tokens={r['ego_pose_token'] for r in sd};cal_tokens={r['calibrated_sensor_token'] for r in sd}
cs=[r for r in first['calibrated_sensor.json'] if r['token'] in cal_tokens];assert {r['token'] for r in cs}==cal_tokens
second=select_tables({'ego_pose.json'},{'ego_pose.json':lambda r:r['token'] in pose_tokens});ep=second['ego_pose.json'];assert {r['token'] for r in ep}==pose_tokens
put('sample_data.json',sd);put('calibrated_sensor.json',cs);put('ego_pose.json',ep)
source_audit={'archive_sha256':HASHES,'metadata_sha256':EXPECTED_META,'models':{},'scope':'Verified original containers and full predictions; extracted subset only. No inference or accuracy evaluation.'}
for model,expected in PRED_HASH.items():
 path=BASE/'full_validation_merged'/model/'results_nusc.json';print('VERIFYING FULL PREDICTIONS:',model,flush=True);assert sha(path)==expected
 with path.open() as f:data=json.load(f)
 assert set(data['results'])==validation_tokens
 chosen={t:data['results'][t] for t in tokens}
 for t,boxes in chosen.items():
  assert len(boxes)<=500
  for b in boxes:
   assert b['sample_token']==t
   assert all(math.isfinite(float(x)) for key in ['translation','size','rotation','velocity'] for x in b[key])
   assert math.isfinite(float(b['detection_score']))
 put('predictions/'+model+'.json',{'results':chosen,'meta':data.get('meta',{})});source_audit['models'][model]={'full_results_sha256':expected,'full_tokens':6019,'selected_tokens':tokens}
 del chosen,data;gc.collect();print('PREDICTION SUBSET PASS:',model,flush=True)
source_audit['preparation_script_url']=globals().get('PREPARATION_SCRIPT_URL','not supplied')
put('source_verification.json',source_audit)
manifest={str(p.relative_to(work)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(work.rglob('*')) if p.is_file()}
put('manifest.json',{'members':manifest,'scope':'Two outcome-blind night candidates; image review and diagnostic matching pending.'})
archive=work.parent/('night_case_inputs_'+stamp+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(work.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(work)))
with zipfile.ZipFile(archive) as z:
 for rel,r in manifest.items():
  b=z.read(rel);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
dest=BASE/'case_followup_metadata_20261028';dest.mkdir(exist_ok=True);target=dest/archive.name;shutil.copy2(archive,target);digest=sha(archive);assert sha(target)==digest
receipt={'status':'NIGHT_CASE_INPUT_TRANSFER_AND_DRIVE_COPY_PASS','archive':target.name,'bytes':target.stat().st_size,'sha256':digest,'sensor_files':14,'samples':2,'verified_payload_members':len(manifest),'elapsed_seconds':round(time.monotonic()-started,1),'scope':'Inputs and existing predictions preserved. No inference; visual analysis pending.'}
rpath=target.with_suffix('.receipt.json');rpath.write_text(json.dumps(receipt,indent=2)+'\n');assert json.loads(rpath.read_text())==receipt
print(json.dumps(receipt,indent=2),flush=True);print('DOWNLOAD:',target,flush=True)
from google.colab import files
files.download(str(target))
