"""Draw conceptual study architecture and the pinned BEVFusion inference flow."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out=Path(__file__).resolve().parents[1]/'docs/architecture'
out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
gray='#E8EDF2';teal='#D9F0ED';ink='#233649'
def box(ax,x,y,w,h,text,own=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',facecolor=teal if own else gray,edgecolor='#087E8B' if own else '#7A8794',lw=1.3))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,color=ink)
def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color=ink,lw=1.4))
def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(13,8.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.965,'Software-only perception study: evaluated branches',weight='bold',fontsize=17,color=ink)
ax.text(.02,.925,'Teal: project integration / analysis     Grey: reused datasets, models and metric definitions',fontsize=11,color=ink)
box(ax,.02,.68,.18,.16,'KITTI\nImages + LiDAR\nCalibration + labels')
box(ax,.27,.80,.36,.085,'Camera detection + projected LiDAR depth',True)
box(ax,.27,.68,.36,.085,'PointPillars / PV-RCNN\nLiDAR 3D detection')
box(ax,.27,.56,.36,.085,'CLOCs on PointPillars\nCamera / 3D candidate score fusion')
for y in [.842,.722,.602]:arrow(ax,(.20,.76),(.265,y))
box(ax,.71,.79,.27,.10,'2D detection / matched depth\nTask-specific evaluation',True)
box(ax,.71,.60,.27,.145,'KITTI Car 3D AP40\nCommon split, controlled settings\nSeed and failure diagnostics',True)
arrow(ax,(.635,.842),(.705,.84));arrow(ax,(.635,.722),(.705,.70));arrow(ax,(.635,.602),(.705,.64))
ax.plot([.02,.98],[.515,.515],color='#CCD4DC',lw=1)
box(ax,.02,.30,.18,.15,'nuScenes\nLiDAR + six cameras\nCalibration + labels')
box(ax,.27,.40,.36,.085,'Official LiDAR-only checkpoint')
box(ax,.27,.27,.36,.085,'Official camera–LiDAR BEVFusion checkpoint')
arrow(ax,(.20,.39),(.265,.442));arrow(ax,(.20,.35),(.265,.312))
box(ax,.71,.29,.27,.19,'Full validation: mAP / NDS\nSelected case studies\nCamera masks on 600 frames\nRuntime on nested 150 frames',True)
arrow(ax,(.635,.442),(.705,.44));arrow(ax,(.635,.312),(.705,.34))
box(ax,.02,.075,.96,.11,'Shared project orchestration: frozen inputs → model checks → saved predictions → evaluation → hashes, plots and evidence',True)
ax.text(.02,.025,'Branches are separate experiments. KITTI and nuScenes metrics are not pooled. No robot navigation or hardware deployment demonstrated.',fontsize=9,color=ink)
save(fig,'study_overview')

fig,ax=plt.subplots(figsize=(13,8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.965,'Pinned BEVFusion implementation: inference data flow',weight='bold',fontsize=17,color=ink)
ax.text(.02,.915,'Reused model components • MMDetection3D fe25f7a • Simplified from the executed configuration and model source',fontsize=10,color=ink)
box(ax,.02,.71,.17,.13,'Six camera images\n256 × 704 per view')
box(ax,.25,.71,.22,.13,'Swin-Tiny backbone\nGeneralizedLSSFPN')
box(ax,.53,.71,.22,.13,'DepthLSSTransform\nImage features → BEV')
arrow(ax,(.195,.775),(.24,.775));arrow(ax,(.48,.775),(.52,.775))
box(ax,.02,.39,.17,.13,'LiDAR keyframe\n+ up to 9 past sweeps')
box(ax,.25,.39,.22,.13,'Voxelization\nMean voxel features')
box(ax,.53,.39,.22,.13,'BEVFusionSparseEncoder\nLiDAR BEV features')
arrow(ax,(.195,.455),(.24,.455));arrow(ax,(.48,.455),(.52,.455))
ax.annotate('',xy=(.64,.70),xytext=(.105,.53),arrowprops={'arrowstyle':'->','color':'#087E8B','lw':1.4,'connectionstyle':'angle,angleA=90,angleB=180,rad=12'})
ax.text(.35,.60,'LiDAR depth cues + recorded calibration',ha='center',fontsize=9,color='#087E8B',bbox={'facecolor':'white','edgecolor':'none','pad':3})
box(ax,.81,.56,.17,.14,'ConvFuser\nCombine BEV features')
arrow(ax,(.755,.775),(.805,.66));arrow(ax,(.755,.455),(.805,.60))
box(ax,.49,.12,.28,.13,'SECOND + SECONDFPN\nShared BEV decoding stages')
ax.plot([.895,.895,.63],[.55,.32,.32],color=ink,lw=1.4)
arrow(ax,(.63,.32),(.63,.255))
box(ax,.81,.12,.17,.13,'TransFusionHead\n3D boxes + scores')
arrow(ax,(.775,.185),(.805,.185))
box(ax,.02,.115,.36,.14,'LiDAR-only checkpoint:\nimage branch and ConvFuser absent;\nLiDAR BEV feeds decoding directly')
ax.text(.02,.035,'Separate pretrained checkpoints are compared. Camera-masking tests keep the fusion weights and all six view slots.',fontsize=10,color=ink)
save(fig,'bevfusion_inference')

manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.iterdir()) if p.suffix in ['.png','.pdf','.svg']}
(out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Two architecture figures generated in PNG/PDF/SVG.')
