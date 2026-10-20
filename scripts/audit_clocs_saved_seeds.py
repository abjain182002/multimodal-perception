"""Audit preserved CLOCs records without training, inference or AP40 recomputation."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=Path('.'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
sources = {}

def read(path):
    data=(args.repo/path).read_bytes()
    sources[path]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                   'git_blob_sha':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}
    return json.loads(data)

base='results/kitti/clocs/'
study=read(base+'aligned_seed_study/study_protocol.json')
selection=read(base+'aligned_seed_study/validation_selection.json')
seed_rows=[];epoch_rows=[];records={};canonical=None;baseline=None
for seed in study['training_seeds']:
    folder=base+f'aligned_seed_study/seed_{seed}/'
    r={n:read(folder+n+'.json') for n in ['protocol','report','history','split','input_audit','before_dev']}
    assert r['report']['status']=='PASS'
    assert r['report']['protocol']==r['protocol']
    assert r['report']['split']==r['split']
    assert r['report']['before_dev']==r['before_dev']
    assert r['protocol']['settings']['seed']==seed
    assert r['protocol']['settings']['split_seed']==study['split_seed']
    assert r['split']['train_frame_ids']==study['optimization_frame_ids']
    assert r['split']['dev_frame_ids']==study['development_frame_ids']
    assert set(r['split']['train_frame_ids']).isdisjoint(r['split']['dev_frame_ids'])
    assert len(r['split']['train_frame_ids'])==96 and len(r['split']['dev_frame_ids'])==32
    protocol=copy.deepcopy(r['protocol'])
    for key in ['seed','output_dir']:protocol['settings'].pop(key)
    invariant={'protocol':protocol,'split':r['split'],'input_audit':r['input_audit']}
    if canonical is None:canonical=invariant
    else:assert invariant==canonical,seed
    assert [h['epoch'] for h in r['history']]==list(range(1,13))
    best=max(r['history'],key=lambda h:h['dev']['fusion_candidate_ap'])
    assert best['epoch']==r['report']['best_epoch']==12
    assert best['dev']==r['report']['best_dev']
    assert r['report']['reload_ap_absolute_error']==0
    for h in r['history']:
        assert h['selected_positive']==5798 and h['selected_negative']==10272648
        assert h['dev']['evaluated_candidates']==3425680 and h['dev']['positive_candidates']==2378
        epoch_rows.append(dict(seed=seed,epoch=h['epoch'],candidate_ap=h['dev']['fusion_candidate_ap'],
                               training_loss=h['mean_training_selected_loss'],dev_loss=h['dev']['mean_frame_focal_loss']))
    ap=read(base+f'seed_diagnostics/seed_{seed}_ap40.json')
    assert ap['status']=='PASS' and ap['frames']==3769 and ap['point_cloud_mode']=='reduced_fov'
    assert ap['baseline_max_absolute_ap40_error']==0
    assert all(x==0 for group in ap['baseline_ap40_differences'].values() for x in group.values())
    if baseline is None:baseline=ap['ap40_strict_iou_070']['pointpillars']
    else:assert ap['ap40_strict_iou_070']['pointpillars']==baseline
    row=dict(seed=seed,best_epoch=12,initial_candidate_ap=r['before_dev']['fusion_candidate_ap'],
             epoch1_candidate_ap=r['history'][0]['dev']['fusion_candidate_ap'],
             best_candidate_ap=best['dev']['fusion_candidate_ap'],
             first_epoch_above_candidate_baseline=next((h['epoch'] for h in r['history'] if h['dev']['fusion_candidate_ap']>h['dev']['pointpillars_candidate_ap']),None),
             baseline_detections=ap['detections']['pointpillars'],fusion_detections=ap['detections']['fusion'],
             recorded_checkpoint_sha256=r['report']['checkpoint_sha256'],
             validation_input_manifest_sha256=ap['source_sha256']['input_manifest'])
    for metric in ['2D','BEV','3D']:
        for difficulty in ['easy','moderate','hard']:
            v=ap['ap40_strict_iou_070']['fusion'][metric][difficulty]
            row[f'{metric}_{difficulty}']=v
            row[f'{metric}_{difficulty}_delta']=v-baseline[metric][difficulty]
    records[seed]=r;seed_rows.append(row)

assert len({r['validation_input_manifest_sha256'] for r in seed_rows})==1
assert selection['selected_seed']==max(seed_rows,key=lambda r:r['best_candidate_ap'])['seed']
assert selection['checkpoint_sha256']==records[selection['selected_seed']]['report']['checkpoint_sha256']
assert selection['validation_results_used_for_selection'] is False
cal=read(base+'pilot_calibration/clocs_yolo26_pilot_score_calibration.json')
split=read(base+'pilot_calibration/clocs_yolo26_pilot_calibration_split.json')
pilot=read(base+'pilot_calibration/clocs_yolo26_pilot_report.json')
assert cal['holdout_frame_ids']==split['holdout_frame_ids']
assert split['holdout_candidates']==427994 and split['holdout_positive_candidates']==508
assert split['selected_on_train']['method']=='logit_prior_blend'
assert split['selected_on_train']['parameter']==1.875
assert set(split['train_frame_ids']).isdisjoint(split['holdout_frame_ids'])
assert abs(split['holdout_selected_calibrated_ap']-split['holdout_baseline_pointpillars_ap']-split['holdout_improvement_over_pointpillars'])<1e-12
assert set(pilot['all_frame_ids'])<=set(study['optimization_frame_ids'])
current_trainer_path='scripts/train_clocs_128.py'
current_trainer=(args.repo/current_trainer_path).read_bytes()
current_hash=hashlib.sha256(current_trainer).hexdigest()
sources[current_trainer_path]={'bytes':len(current_trainer),'sha256':current_hash,
    'git_blob_sha':hashlib.sha1(b'blob '+str(len(current_trainer)).encode()+b'\0'+current_trainer).hexdigest()}
for name,rows in [('seed_audit_table.csv',seed_rows),('epoch_audit_table.csv',epoch_rows)]:
    with (args.output/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
audit={'status':'BOUNDED_CLOCS_SAVED_RECORD_AUDIT_COMPLETE_CAUSE_INCONCLUSIVE',
       'recorded_protocol_input_split_equality':True,'all_nine_baseline_metrics_exact_each_seed':True,
       'best_epoch_and_reload_checks':True,'matched_validation_manifest_record':True,
       'seed29_easy_delta_ap_points':seed_rows[1]['3D_easy_delta'],
       'seed29_vs_seed30_development_candidate_ap_difference':seed_rows[1]['best_candidate_ap']-seed_rows[2]['best_candidate_ap'],
       'seed29_vs_seed30_easy_ap40_difference':seed_rows[1]['3D_easy']-seed_rows[2]['3D_easy'],
       'blend_train_selected_parameter':split['selected_on_train']['parameter'],
       'blend_four_frame_candidate_ap':split['holdout_selected_calibrated_ap'],
       'blend_four_frame_baseline_candidate_ap':split['holdout_baseline_pointpillars_ap'],
       'blend_full_ap40_status':'Unperformed; retain as unfinished exploratory pilot',
       'recorded_executed_trainer_sha256':study['trainer_sha256'],
       'current_repository_trainer_sha256':current_hash,
       'current_trainer_matches_recorded_execution_source':current_hash==study['trainer_sha256'],
       'cause':'Not identified from saved aggregate records',
       'scope':'Audits recorded metadata, learning histories and stored AP40. No raw-candidate/checkpoint rehash, per-frame attribution, training, inference, or fresh AP40 evaluation.'}
(args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(args.output/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(audit,indent=2))
