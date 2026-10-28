"""Verify packaged export, restore payloads, and check saved input invariants."""
import hashlib,json,zipfile,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('input'));a=p.parse_args()
def sha(b):return hashlib.sha256(b).hexdigest()
assert sha(a.archive.read_bytes())=='6bbf19622627a1fe5e4307628ff12dc5f070828b3f3361b235e3df2fde7250f6'
def unpack(zpath,out,manifest):
 with zipfile.ZipFile(zpath) as z:
  d=json.loads(z.read(manifest));members=d['members']
  for name,r in members.items():
   target=out/name;assert target.resolve().is_relative_to(out.resolve())
   b=z.read(name);assert len(b)==r['bytes'] and sha(b)==r['sha256'],name
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 return len(members)
n=unpack(a.archive,a.output,'manifest.json')
n2=unpack(a.output/'inputs/four_case_inputs.zip',a.output/'prepared','transfer_manifest.json')
base=a.output/'inference';tokens=set()
for f in (base/'lidar_camera/clean/frames').glob('*.json'):
 t=f.stem;tokens.add(t);clean=json.loads(f.read_text());cc=clean['input_checks']
 for condition,gain in [('clean',1),('darkening_050',.5),('darkening_025',.25)]:
  d=json.loads((base/'lidar_camera'/condition/'frames'/f.name).read_text());c=d['input_checks']
  assert d['sample_token']==t and c['image_multiplier']==gain and c['pre_normalization_check']
  for k in ['points_sha256','metadata_sha256','raw_image_sha256']:assert c[k]==cc[k],(t,k)
  assert d['checkpoint_sha256']==clean['checkpoint_sha256']
 lidar=json.loads((base/'lidar/clean/frames'/f.name).read_text());assert lidar['input_checks']['points_sha256']==cc['points_sha256']
assert len(tokens)==4
print(json.dumps(dict(status='EXPORT_AND_RECORDED_INPUT_INVARIANTS_PASS',export_members=n,input_members=n2,frames=4,scope='Verifies preserved records; does not re-execute transforms or inference.'),indent=2))
