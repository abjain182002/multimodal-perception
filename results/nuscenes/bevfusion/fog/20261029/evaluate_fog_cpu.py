"""Score eight saved fog runs with the already validated nuScenes CPU evaluator.

No inference. The published full-val GT/metadata filters are retained, then both
prediction and GT containers are restricted to the exact frozen 600 tokens.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''): h.update(data)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # SDK metrics legitimately contain undefined class-specific NaNs.
    with path.open('x') as f: json.dump(data, f, indent=2); f.write('\n')


def gt_fingerprint(boxes):
    """Hash SDK serialization without changing its NumPy-backed GT objects."""
    import numpy as np

    def numpy_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Unsupported GT audit type: {type(value).__name__}')

    # SDK velocities can legitimately be NaN; retain those audit values. This
    # fingerprint never changes the SDK boxes or the metric calculation.
    raw = json.dumps(boxes.serialize(), sort_keys=True, default=numpy_value)
    return hashlib.sha256(raw.encode()).hexdigest()


def pad_prediction(data, full_tokens, chosen):
    assert len(chosen) == len(set(chosen)) and set(data['results']) == set(chosen)
    assert set(chosen) <= set(full_tokens) and len(full_tokens) == len(set(full_tokens))
    # Retain original prediction iteration order, then append empty outside tokens.
    # Padding exists only to satisfy the stock full-val constructor token assertion.
    result = dict(data['results'])
    result.update({token: [] for token in full_tokens if token not in result})
    return {'meta': data['meta'], 'results': result}


def preserve(folder, destination):
    files = {str(p.relative_to(folder)): {'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(folder.rglob('*')) if p.is_file()}
    archive = destination / (folder.name + '_metrics.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in files: z.write(folder / name, name)
        z.writestr('manifest.json', json.dumps({'files':files}, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,r in files.items():
            raw=z.read(name); assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
    return {'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}


def evaluate(args):
    import nuscenes
    from nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.utils.splits import create_splits_scenes
    work=args.work; out=work/'evidence'; reference=work/'weather_reference'
    helper=reference/'evaluate_weather_groups.py'
    assert sha(helper)=='e1787b690cd7639b85489b438487849d23c110e4f0daf9ed187bca615b5fea08'
    sys.path.insert(0,str(reference)); import evaluate_weather_groups as previous
    expected=json.loads((reference/'reference/environment_audit.json').read_text())
    for package,version in expected['packages'].items(): assert metadata.version(package)==version, package
    sdk=Path(nuscenes.__file__).parent
    for relative,digest in expected['evaluator_source_sha256'].items(): assert sha(sdk/relative)==digest, relative
    cfg=config_factory('detection_cvpr_2019')
    assert cfg.serialize()==json.loads((reference/'reference/detection_config.json').read_text())
    write(out/'evaluator_environment.json',{'packages':expected['packages'],'source_hashes_verified':True,'evaluator_source_sha256':expected['evaluator_source_sha256'],'worker_sha256':sha(__file__)})
    inputs=json.loads((work/'inputs/verified_runs.json').read_text())
    chosen=(work/'inputs/sample_tokens.txt').read_text().splitlines(); assert len(chosen)==600
    nusc=NuScenes(version='v1.0-trainval',dataroot=str(args.root),verbose=True)
    official=set(create_splits_scenes()['val'])
    full_tokens=[s['token'] for s in nusc.sample if nusc.get('scene',s['scene_token'])['name'] in official]
    assert len(full_tokens)==6019 and set(chosen)<=set(full_tokens)
    scenes=Counter(nusc.get('sample',t)['scene_token'] for t in chosen)
    assert len(scenes)==150 and set(scenes.values())=={4}
    evaluator=None; frozen_gt=None; gt_hash=None; completed=[]
    for run in inputs:
        model,level=run['model'],run['level']; source=Path(run['prediction_path'])
        assert sha(source)==run['prediction_sha256']
        data=json.loads(source.read_text()); padded=pad_prediction(data,full_tokens,chosen)
        augmented=work/'inputs'/f'{model}_{level}_padded.json'
        write(augmented,padded)
        if evaluator is None:
            evaluator=NuScenesEval(nusc=nusc,config=cfg,result_path=str(augmented),eval_set='val',
                output_dir=str(work/'sdk_initialization'),verbose=True)
            assert set(evaluator.gt_boxes.sample_tokens)==set(full_tokens)
            frozen_gt=previous.subset_boxes(evaluator.gt_boxes,set(chosen))
            gt_hash=gt_fingerprint(frozen_gt)
        else:
            boxes,meta=load_prediction(str(augmented),cfg.max_boxes_per_sample,DetectionBox,verbose=True)
            evaluator.pred_boxes=filter_eval_boxes(nusc,add_center_dist(nusc,boxes),cfg.class_range,verbose=True)
            evaluator.meta=meta
        evaluator.gt_boxes=frozen_gt
        evaluator.pred_boxes=previous.subset_boxes(evaluator.pred_boxes,set(chosen))
        evaluator.sample_tokens=evaluator.gt_boxes.sample_tokens
        assert set(evaluator.pred_boxes.sample_tokens)==set(evaluator.gt_boxes.sample_tokens)==set(chosen)
        directory=out/f'level_{level}_{model}'
        print('EVALUATING SAVED FOG PREDICTIONS:',level,model,'600 frames',flush=True)
        summary,coverage=previous.save_evaluation(evaluator,directory)
        assert gt_fingerprint(frozen_gt)==gt_hash
        audit={'status':'FROZEN_FOG_CONDITION_CPU_EVALUATION_PASS','model':model,'level':level,'samples':600,
            'filtered_gt_sha256':gt_hash,'prediction_sha256':sha(source),'source_archive_sha256':run['source_archive_sha256'],
            'padding':len(full_tokens)-len(chosen),'padding_removed_before_evaluate':True,
            'input_token_order_preserved':True,'metrics_summary_sha256':sha(directory/'metrics_summary.json'),
            'metrics_details_sha256':sha(directory/'metrics_details.json'),
            'scope':'Stock SDK full-val loading/filtering, then exact 600-token subset before evaluation. No outside-frame scores, changed GT filters or inference.'}
        write(directory/'evaluation_audit.json',audit)
        receipt=preserve(directory,args.destination); completed.append({**audit,'backup':receipt})
        print(json.dumps({'level':level,'model':model,'mAP_x100':summary['mean_ap']*100,'NDS_x100':summary['nd_score']*100}),flush=True)
    write(out/'completed_cpu_evaluations.json',completed)
    report(out)


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries={};coverage={};rows=[];classes=[]
    for level in range(4):
        for model in ('lidar','lidar_camera'):
            folder=out/f'level_{level}_{model}'
            summaries[model,level]=json.loads((folder/'metrics_summary.json').read_text())
            coverage[model,level]=json.loads((folder/'coverage.json').read_text())
    first=coverage['lidar',0]['filtered_gt_counts']
    assert all(c['filtered_gt_counts']==first for c in coverage.values())
    class_rows=[]
    for model in ('lidar','lidar_camera'):
        baseline=summaries[model,0]
        for level in range(4):
            s=summaries[model,level];c=coverage[model,level]
            rows.append({'model':model,'level':level,'samples':600,'mAP_x100':100*s['mean_ap'],'NDS_x100':100*s['nd_score'],
                'mAP_change_from_own_baseline_points':100*(s['mean_ap']-baseline['mean_ap']),
                'NDS_change_from_own_baseline_points':100*(s['nd_score']-baseline['nd_score']),
                **{'mean_'+key:val for key,val in s['tp_errors'].items()}})
            for name,ap in s['mean_dist_aps'].items():
                class_rows.append({'model':model,'level':level,'class':name,'filtered_GT':c['filtered_gt_counts'][name],
                    'filtered_predictions':c['filtered_prediction_counts'][name],'AP_x100':100*ap,
                    'change_from_own_baseline_points':100*(ap-baseline['mean_dist_aps'][name])})
    for filename,data in [('fog_metrics.csv',rows),('fog_class_metrics.csv',class_rows)]:
        with (out/filename).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(out/'paired_fog_summary.json',{'status':'ALL_EIGHT_FOG_CPU_EVALUATIONS_PASS','samples':600,'scenes':150,
        'same_filtered_GT_all_conditions':True,'filtered_GT_counts':first,'absent_gt_classes':[name for name,count in first.items() if count==0],
        'fusion_minus_lidar_points':{str(level):{key:100*(summaries['lidar_camera',level][key]-summaries['lidar',level][key]) for key in ('mean_ap','nd_score')} for level in range(4)},
        'uncertainty':'Descriptive point estimates, one seeded corruption realisation, correlated frames/scenes/logs. No confidence intervals, significance or deployment-safety claims.',
        'interpretation':'Synthetic paired input challenge for separately pretrained pipelines. Use fresh strict baselines. Not real fog, shared physical visibility distance, or controlled camera-component ablation.'})
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('mean_ap','nd_score','pedestrian'),('Ten-class mAP x100','NDS x100','Pedestrian AP x100')):
        for model,label,color in [('lidar','LiDAR','#356899'),('lidar_camera','LiDAR + camera','#d67a25')]:
            values=[100*(summaries[model,l]['mean_dist_aps']['pedestrian'] if key=='pedestrian' else summaries[model,l][key]) for l in range(4)]
            ax.plot(range(4),values,'o-',label=label,color=color);ax.set_ylim(0,100);ax.set_xticks(range(4),['Baseline','Fog 1','Fog 2','Fog 3']);ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Frozen 600-frame / 150-scene synthetic fog challenge')
    fig.text(.02,.02,'Same GT and frames; separately pretrained pipelines; one corruption realisation. Severity indices are not calibrated visibility distances.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.94))
    for ext in ('png','pdf'):fig.savefig(out/f'fog_severity_comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--work',type=Path,required=True);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    evaluate(parser.parse_args())
