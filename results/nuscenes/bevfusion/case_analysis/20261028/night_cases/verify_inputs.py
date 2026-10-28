"""Verify and extract the preserved night input ZIP; standard library only."""
from pathlib import Path
import argparse,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('input'));a=p.parse_args()
assert hashlib.sha256(a.archive.read_bytes()).hexdigest()=='e09b902bc7be114021d435c9720c5223947fd9d4b9c74c016c320f5fe3d20f3d'
with zipfile.ZipFile(a.archive) as z:
 m=json.loads(z.read('manifest.json'))['members']
 for name,r in m.items():
  dest=a.output/name;assert dest.resolve().is_relative_to(a.output.resolve());b=z.read(name)
  assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],name
  dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
print('INPUT ARCHIVE AND',len(m),'PAYLOAD HASHES PASS')
