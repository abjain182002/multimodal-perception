"""Read-only Colab inventory. Run after Drive mount; does not extract sensors or run inference."""
from pathlib import Path
import json,zipfile
base=Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation')
p=base/'case_followup_metadata_20261028/case_metadata_and_night_candidates.json'
d=json.loads(p.read_text());needed=set()
for c in d['night_candidates']:
 inv=c['sensor_inventory'];needed.update(inv['camera_files']);needed.update(x for x in inv['lidar_files'] if x.startswith('samples/'))
coverage={x:[] for x in needed};errors=[]
for archive in sorted(base.rglob('*.zip')):
 try:
  with zipfile.ZipFile(archive) as z:
   for member in z.infolist():
    rel=member.filename.removeprefix('dataset/')
    if rel in coverage:coverage[rel].append({'archive':str(archive),'member':member.filename,'bytes':member.file_size})
 except Exception as e:errors.append({'archive':str(archive),'error':str(e)})
print('NIGHT KEYFRAME COVERAGE:',sum(bool(v) for v in coverage.values()),'/',len(needed))
print(json.dumps(coverage,indent=2))
print('POSSIBLE SAVED FULL PREDICTIONS:')
for f in sorted(base.rglob('*.json')):
 if f.stat().st_size>30_000_000 and ('pred' in f.name.lower() or 'results' in f.name.lower()):print(str(f),'| MB:',round(f.stat().st_size/1e6,2))
print('INSPECTION ERRORS:',json.dumps(errors))
print('READ-ONLY INVENTORY COMPLETE; ZIP DIRECTORY COVERAGE IS NOT CONTENT HASH VERIFICATION.')
