"""Ten-class case diagnostics with recovered original annotations; no model inference.
Filtering follows public nuScenes SDK rules; this is a diagnostic replay, not an official AP/NDS evaluation.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--metadata',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
b=a.metadata.read_bytes();assert hashlib.sha256(b).hexdigest()=='40ebf900154cf8656b0b87c55e9ca3b875908bf972453b0f48e3ddbea7d040c1';M=json.loads(b)
def rd(n):return json.loads((a.reference/n).read_text())
C=rd('candidate_contexts.json');SD={x['filename']:x for x in rd('sample_data.json')};EP={x['token']:x for x in rd('ego_pose.json')}
def R(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
mapping={'movable_object.barrier':'barrier','vehicle.bicycle':'bicycle','vehicle.bus.bendy':'bus','vehicle.bus.rigid':'bus','vehicle.car':'car','vehicle.construction':'construction_vehicle','vehicle.motorcycle':'motorcycle','movable_object.trafficcone':'traffic_cone','vehicle.trailer':'trailer','vehicle.truck':'truck',**{'human.pedestrian.'+x:'pedestrian' for x in ['adult','child','construction_worker','police_officer']}}
ranges={k:50 for k in ['car','truck','bus','trailer','construction_vehicle']};ranges.update(pedestrian=40,bicycle=40,motorcycle=40,traffic_cone=30,barrier=30)
def inside(box,point):
 local=R(box['rotation']).T@(np.array(point)-box['translation']);w,l,h=box['size'];return bool(np.all(np.abs(local)<=np.array([l,w,h])/2))
def filt(boxes,ego,racks,gt=False):
 out=[];counts={'original':len(boxes),'range_removed':0,'zero_points_removed':0,'rack_removed':0}
 for x in boxes:
  if np.linalg.norm(np.array(x['translation'][:2])-ego[:2])>=ranges[x['class']]:counts['range_removed']+=1;continue
  if gt and x['num_lidar_pts']+x['num_radar_pts']==0:counts['zero_points_removed']+=1;continue
  if x['class'] in ['bicycle','motorcycle'] and any(inside(r,x['translation']) for r in racks):counts['rack_removed']+=1;continue
  out.append(x)
 return out,counts
def key(x):return (x['class'],*np.round(x['translation'],6),*np.round(x['size'],6),*np.round(x['rotation'],6))
def match(gt,pr,th):
 used=set();hits=[];fp=[]
 for i in sorted(range(len(pr)),key=lambda i:(-pr[i]['score'],i)):
  if pr[i]['score']<th:continue
  dd,j=min(((float(np.linalg.norm(np.array(pr[i]['translation'][:2])-g['translation'][:2])),j) for j,g in enumerate(gt) if j not in used and pr[i]['class']==g['class']),default=(float('inf'),-1))
  if dd<2:used.add(j);hits.append((i,j))
  else:fp.append(i)
 return hits,fp,[j for j in range(len(gt)) if j not in used]
vis={r['token']:r['description'] for r in M['visibility_definitions']};rows=[];details=[];visibility=[];audit=[];Bsets={}
for letter,gi in [('A',22),('B',26),('C',23),('D',7)]:
 t=M['existing_cases'][letter];c=C[t];ego=EP[SD[c['sensor_files']['LIDAR_TOP']]['ego_pose_token']]['translation'];ann=[r for r in M['annotations'] if r['sample_token']==t];racks=[r for r in ann if r['category_name']=='static_object.bicycle_rack'];gt0=[dict(r,**{'class':mapping[r['category_name']]}) for r in ann if r['category_name'] in mapping];gg,filters=filt(gt0,ego,racks,True)
 assert len(gg)==len(c['ground_truth']) and sorted(map(key,gg))==sorted(map(key,c['ground_truth'])),(letter,'GT filter mismatch')
 originalgt=c['ground_truth'];focal=originalgt[gi];ar=[r for r in gg if key(r)==key(focal)];assert len(ar)==1;ar=ar[0]
 visibility.append(dict(case=letter,class_name=focal['class'],sample_token=t,annotation_token=ar['token'],visibility_token=ar['visibility_token'],visibility_description=vis[ar['visibility_token']],annotation_lidar_points=ar['num_lidar_pts'],annotation_radar_points=ar['num_radar_pts']))
 audit.append(dict(case=letter,ground_truth_filter_counts=filters,filtered_ground_truth_matches_preserved=True,bicycle_racks=len(racks)))
 # Preserve the original filtered GT order and therefore the original diagnostic tie-breaking rule.
 for model,condition in [('lidar','clean'),('lidar_camera','clean'),('lidar_camera','darkening_050'),('lidar_camera','darkening_025')]:
  raw=json.loads((a.input/'inference'/model/condition/'results_nusc.json').read_text())['results'][t];pr0=[dict(x,score=x['detection_score'],**{'class':x['detection_name']}) for x in raw];pr,pf=filt(pr0,ego,racks)
  for th in [.1,.3,.5]:
   hits,fp,fn=match(originalgt,pr,th);rows.append(dict(case=letter,model=model,condition=condition,threshold=th,classes=10,gt=len(originalgt),tp=len(hits),fp=len(fp),fn=len(fn),precision=len(hits)/(len(hits)+len(fp)) if hits or fp else '',recall=len(hits)/len(originalgt),predictions_removed_by_racks=pf['rack_removed']))
   details.append(dict(case=letter,model=model,condition=condition,threshold=th,matched_gt_indices=[j for i,j in hits],missed_gt_indices=fn,false_positive_predictions=[pr[i] for i in fp]))
   if letter=='B' and model=='lidar_camera' and th==.3:Bsets[condition]={j for i,j in hits}
clean=Bsets['clean'];half=Bsets['darkening_050'];change={'gained_gt_indices':sorted(half-clean),'lost_gt_indices':sorted(clean-half),'unchanged_matched_gt_indices':sorted(clean&half)}
for name,data in [('ten_class_threshold_tradeoffs.csv',rows),('focal_visibility.csv',visibility)]:
 with (a.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(a.output/'matching_details.json').write_text(json.dumps(details,indent=2))
(a.output/'completion_audit.json').write_text(json.dumps(dict(status='TEN_CLASS_DIAGNOSTICS_AND_VISIBILITY_PASS',metadata_sha256=hashlib.sha256(b).hexdigest(),ground_truth_checks=audit,threshold_records=len(rows),B_half_vs_clean=change,scope='Local fixed-threshold matching diagnostics; not official nuScenes AP/NDS or an independent evaluator validation.'),indent=2))
print(json.dumps(visibility,indent=2));print(json.dumps(change));print(json.dumps([r for r in rows if r['case']=='B' and r['threshold']==.3],indent=2))
