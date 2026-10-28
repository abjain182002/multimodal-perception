"""CPU diagnostic analysis of two preselected night frames. No inference.
Usage: python build_night_analysis.py --input input --output results
Requires numpy, scipy, matplotlib, Pillow.
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def R(q):return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()

mapping={'movable_object.barrier':'barrier','vehicle.bicycle':'bicycle','vehicle.bus.bendy':'bus','vehicle.bus.rigid':'bus','vehicle.car':'car','vehicle.construction':'construction_vehicle','vehicle.motorcycle':'motorcycle','movable_object.trafficcone':'traffic_cone','vehicle.trailer':'trailer','vehicle.truck':'truck',**{'human.pedestrian.'+x:'pedestrian' for x in ['adult','child','construction_worker','police_officer']}}

ranges={k:50 for k in ['car','truck','bus','trailer','construction_vehicle']}

ranges.update(pedestrian=40,bicycle=40,motorcycle=40,traffic_cone=30,barrier=30)

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

def match(gt,pr,th):
 used=set();hits=[];fp=[]
 for i in sorted(range(len(pr)),key=lambda i:(-pr[i]['score'],i)):
  if pr[i]['score']<th:continue
  dd,j=min(((float(np.linalg.norm(np.array(pr[i]['translation'][:2])-g['translation'][:2])),j) for j,g in enumerate(gt) if j not in used and pr[i]['class']==g['class']),default=(float('inf'),-1))
  if dd<2:used.add(j);hits.append((i,j))
  else:fp.append(i)
 return hits,fp,[j for j in range(len(gt)) if j not in used]
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((a.input/n).read_text())
M=read('case_metadata_and_night_candidates.json');SD={x['filename']:x for x in read('sample_data.json')};EP={x['token']:x for x in read('ego_pose.json')};CS={x['token']:x for x in read('calibrated_sensor.json')};P={m:read('predictions/'+m+'.json')['results'] for m in ['lidar','lidar_camera']}
def project(b,sd):
 w,l,h=b['size'];v=np.array([[x*l/2,y*w/2,z*h/2] for z in [-1,1] for x,y in [(1,1),(1,-1),(-1,-1),(-1,1)]]).T;v=R(b['rotation'])@v+np.array(b['translation'])[:,None]
 for pose in [EP[sd['ego_pose_token']],CS[sd['calibrated_sensor_token']]]:v=R(pose['rotation']).T@(v-np.array(pose['translation'])[:,None])
 if v[2].min()<=.2:return None
 uv=np.array(CS[sd['calibrated_sensor_token']]['camera_intrinsic'])@v;return uv[:2]/uv[2]
edges=[(i,(i//4)*4+(i+1)%4) for i in range(8)]+[(i,i+4) for i in range(4)]
def boxplot(ax,uv,color,ls='-'):
 if uv is None:return
 for j,k in edges:ax.plot(uv[0,[j,k]],uv[1,[j,k]],color=color,ls=ls,lw=1.3)
def writecsv(n,rr):
 with (a.output/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
rows=[];objectrows=[];focalrows=[];selection=[];details=[];fig,axs=plt.subplots(2,2,figsize=(11,9));fig.subplots_adjust(top=.86,bottom=.11,hspace=.55,wspace=.10)
for row,c in enumerate(M['night_candidates']):
 t=c['sample_token'];inv=c['sensor_inventory'];lfile=next(x for x in inv['lidar_files'] if x.startswith('samples/'));ego=EP[SD[lfile]['ego_pose_token']]['translation'];ann=[x for x in M['annotations'] if x['sample_token']==t];racks=[x for x in ann if x['category_name']=='static_object.bicycle_rack'];gt0=[dict(x,**{'class':mapping[x['category_name']]}) for x in ann if x['category_name'] in mapping];gt,gtfilter=filt(gt0,ego,racks,True)
 # Focal selection uses only GT distance and projection, never model outcomes.
 eligible=[]
 for gi,g in enumerate(gt):
  priority=0 if g['class']=='pedestrian' else 1 if g['class']=='car' else 2
  views=[]
  for fn in inv['camera_files']:
   uv=project(g,SD[fn])
   if uv is None:continue
   size=Image.open(a.input/'dataset'/fn).size;lo,hi=uv.min(1),uv.max(1);visible=np.maximum(0,np.minimum(hi,size)-np.maximum(lo,0));area=float(visible.prod())
   if area>0:views.append((area,fn,uv))
  if views:
   area,fn,uv=max(views,key=lambda x:(x[0],x[1]));distance=float(np.linalg.norm(np.array(g['translation'][:2])-ego[:2]));eligible.append((priority,distance,g['token'],gi,fn,uv))
 if eligible:
  _,distance,_,gi,filename,uv=min(eligible,key=lambda x:x[:3]);g=gt[gi];im=Image.open(a.input/'dataset'/filename);lo,hi=uv.min(1),uv.max(1);center=(lo+hi)/2;span=max(float(max(hi-lo))*1.7,180);width=min(span*1.65,im.width);height=min(span,im.height);cx=np.clip(center[0],width/2,im.width-width/2);cy=np.clip(center[1],height/2,im.height-height/2);lim=(cx-width/2,cx+width/2,cy+height/2,cy-height/2)
  selection.append(dict(scene=c['scene_name'],sample_token=t,annotation_token=g['token'],class_name=g['class'],ego_xy_distance_m=distance,visibility_token=g['visibility_token'],num_lidar_pts=g['num_lidar_pts'],num_radar_pts=g['num_radar_pts'],camera=filename.split('/')[1],rule='Nearest eligible pedestrian, else car, else other class; largest clipped GT projection selects camera. No predictions used.'))
 else:
  assert not gt,'GT present but no eligible projection; requires review'
  filename=next(x for x in inv['camera_files'] if '/CAM_FRONT/' in x);gi=None
  refpred=[dict(x,score=x['detection_score'],**{'class':x['detection_name']}) for x in P['lidar'][t]];refpred,_=filt(refpred,ego,racks);anchors=sorted([x for x in refpred if x['score']>=.3],key=lambda x:-x['score'])
  view_options=[]
  if anchors:
   for fn in inv['camera_files']:
    vv=project(anchors[0],SD[fn])
    if vv is None:continue
    size=Image.open(a.input/'dataset'/fn).size;area=float(np.maximum(0,np.minimum(vv.max(1),size)-np.maximum(vv.min(1),0)).prod())
    if area>0:view_options.append((area,fn,vv))
  if view_options:
   _,filename,vv=max(view_options,key=lambda x:(x[0],x[1]));im=Image.open(a.input/'dataset'/filename);lo,hi=vv.min(1),vv.max(1);center=(lo+hi)/2;span=max(float(max(hi-lo))*2.0,240);width=min(span*1.65,im.width);height=min(span,im.height);cx=np.clip(center[0],width/2,im.width-width/2);cy=np.clip(center[1],height/2,im.height-height/2);lim=(cx-width/2,cx+width/2,cy+height/2,cy-height/2)
  else:im=Image.open(a.input/'dataset'/filename);lim=(0,im.width,im.height,0)
  selection.append(dict(scene=c['scene_name'],sample_token=t,annotation_token=None,class_name=None,camera=filename.split('/')[1],view_rule='Camera and crop selected to show highest-score LiDAR unmatched candidate, then shared by both models; outcome-informed visualisation only.',scope='No evaluable ten-class GT: negative-frame check only. Scene text does not prove a pedestrian is in this frame.'))
 for col,model in enumerate(['lidar','lidar_camera']):
  raw=[dict(x,score=x['detection_score'],**{'class':x['detection_name']}) for x in P[model][t]];pred,pf=filt(raw,ego,racks);prmatches={}
  for th in [.1,.3,.5]:
   hit,fp,fn=match(gt,pred,th);prmatches[th]=(hit,fp,fn);rows.append(dict(scene=c['scene_name'],sample_token=t,model=model,threshold=th,gt=len(gt),tp=len(hit),fp=len(fp),fn=len(fn),precision=len(hit)/(len(hit)+len(fp)) if hit or fp else '',recall=len(hit)/len(gt) if gt else '',scope='ten-class diagnostic at XY <2 m; not AP/NDS'))
   details.append(dict(scene=c['scene_name'],model=model,threshold=th,matches=[{'prediction_index':i,'annotation_token':gt[j]['token']} for i,j in hit],false_positive_boxes=[pred[i] for i in fp],missed_annotation_tokens=[gt[j]['token'] for j in fn],gt_filter=gtfilter,pred_filter=pf))
  for oi,og in enumerate(gt):
   candidates=[x for x in pred if x['class']==og['class']]
   near=min(candidates,key=lambda x:float(np.linalg.norm(np.array(x['translation'][:2])-og['translation'][:2]))) if candidates else None
   objectrows.append(dict(scene=c['scene_name'],model=model,annotation_token=og['token'],class_name=og['class'],visibility_token=og['visibility_token'],num_lidar_pts=og['num_lidar_pts'],ego_xy_distance_m=float(np.linalg.norm(np.array(og['translation'][:2])-ego[:2])),nearest_same_class_score=near['score'] if near else '',matched_010=any(j==oi for i,j in prmatches[.1][0]),matched_030=any(j==oi for i,j in prmatches[.3][0]),matched_050=any(j==oi for i,j in prmatches[.5][0])))
  ax=axs[row,col];ax.imshow(im);hit,fp,fn=prmatches[.3];ax.set_title(f"{c['scene_name']} | {'LiDAR' if col==0 else 'Fusion'}",fontsize=12)
  if gi is not None:
   assigned=next((i for i,j in hit if j==gi),None);same=[i for i,x in enumerate(pred) if x['class']==g['class']];pi=min(same,key=lambda i:(float(np.linalg.norm(np.array(pred[i]['translation'][:2])-g['translation'][:2])),i)) if same else None
   boxplot(ax,uv,'#ff35a0');err=None;score=None
   if pi is not None:
    b=pred[pi];score=b['score'];err=float(np.linalg.norm(np.array(b['translation'][:2])-g['translation'][:2]));boxplot(ax,project(b,SD[filename]),'#00d9ff','-' if assigned==pi else '--')
   focalrows.append(dict(scene=c['scene_name'],sample_token=t,annotation_token=g['token'],class_name=g['class'],model=model,nearest_score=score,nearest_xy_error_m=err,matched_030=assigned is not None,nearest_is_assigned=pi==assigned if assigned is not None else False))
   label=f"{g['class']} | score {score:.3f} | XY error {err:.3f} m" if pi is not None else f"{g['class']} | no same-class candidate"
   ax.set_xlabel(label+f"\n{'MATCH' if assigned is not None else 'MISS'} at 0.30 | frame TP/FP/FN: {len(hit)}/{len(fp)}/{len(fn)}",fontsize=10)
  else:
   displayed=0
   for i in fp:
    vv=project(pred[i],SD[filename])
    if vv is not None:boxplot(ax,vv,'#f6a500');displayed+=1
   ax.set_xlabel(f"No evaluable GT | frame FP at 0.30: {len(fp)}\nUnmatched-candidate crop; counts cover full 360° frame",fontsize=10)
  ax.set_xlim(lim[:2]);ax.set_ylim(lim[2:]);ax.set_xticks([]);ax.set_yticks([])
fig.suptitle('Preselected night cases: object detection and a negative-frame check',fontsize=15,y=.97)
fig.text(.5,.915,'Pink GT | cyan nearest candidate (solid if assigned at 0.30) | orange: unmatched predictions under annotation protocol',ha='center',fontsize=10)
fig.text(.5,.02,'Different scenes do not isolate illumination or rain effects. Negative-frame recall is undefined, not 100%.\nOriginal images shown without brightness enhancement. Single-view projections omit object-motion compensation and occlusion tests.',ha='center',fontsize=9)
for ext in ['png','pdf']:fig.savefig(a.output/f'night_case_comparison.{ext}',dpi=170)
plt.close(fig)
writecsv('all_positive_object_diagnostics.csv',objectrows);writecsv('frame_threshold_diagnostics.csv',rows);writecsv('focal_object_diagnostics.csv',focalrows)
(a.output/'focal_selection.json').write_text(json.dumps(selection,indent=2));(a.output/'matching_details.json').write_text(json.dumps(details,indent=2));(a.output/'analysis_audit.json').write_text(json.dumps(dict(status='TWO_PRESELECTED_NIGHT_FRAME_DIAGNOSTICS_PASS',frames=2,positive_frames=sum(x['annotation_token'] is not None for x in selection),negative_frames=sum(x['annotation_token'] is None for x in selection),model_threshold_rows=len(rows),scope='Original paired pretrained predictions; illustrative diagnostic study, no causal weather inference or AP/NDS.'),indent=2))
print(json.dumps(selection,indent=2));print(json.dumps(focalrows,indent=2));print(json.dumps([r for r in rows if r['threshold']==.3],indent=2))
