"""Self-contained CPU-only fog evaluation from verified Drive artifacts.

Works after a runtime reset. No detector environment, raw sensor restoration,
model checkpoints or GPU required. Same source-checked SDK as prior CPU replay.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile

BASE=Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation')
FOLDER='results/nuscenes/bevfusion/fog/20261029/'
SUMMARY_SHA='197843a404229dd12443af13b271e941a6f89f75c97a47bfb8a4a7a1e5dc065f'
WEATHER_SHA='75126afda2354f95f9402fccfa37100fb45d3f956cb1d5600cfb6fdf2c289227'
WORKER_SHA='ff7b635acaf2bb9c31bca8da14db34f9f898f38294451ff23afeaf0966af1410'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')


def verified_archive(path,expected,manifest_name='manifest.json',extract=None):
    assert sha(path)==expected,'Archive SHA mismatch: '+str(path)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        files=json.loads(z.read(manifest_name))['files']
        assert set(z.namelist())==set(files)|{manifest_name}
        for name,record in files.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data=z.read(name)
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
            if extract is not None:
                target=extract/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return files


def validate_frame(frame,token,index,ledger):
    assert frame['sample_token']==token and frame['index']==index
    assert frame['packed_points_sha256']==ledger['packed_points_sha256']
    boxes,scores,labels=frame['boxes_lidar'],frame['scores'],frame['labels']
    assert len(boxes)==len(scores)==len(labels)==ledger['prediction_count']
    assert all(len(box)==9 and all(math.isfinite(x) for x in box) for box in boxes)
    assert all(math.isfinite(s) and 0<=s<=1 for s in scores)
    assert all(type(label) is int and 0<=label<10 for label in labels)


def run(command,log):
    with log.open('x') as f:
        p=subprocess.Popen(list(map(str,command)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in p.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            code=p.wait()
        except BaseException:p.terminate();p.wait();raise
    assert code==0,'CPU worker stopped; inspect saved log'


def backup(out,dest,stamp,success):
    files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'manifest.json',{'files':files})
    archive=out.parent/('fog_cpu_evaluation_'+stamp+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*files,'manifest.json']:z.write(out/name,name)
    verified_archive(archive,sha(archive))
    target=dest/archive.name
    with archive.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
    assert sha(target)==sha(archive)
    receipt={'status':'FOG_CPU_EVALUATION_AND_DRIVE_COPY_PASS' if success else 'FOG_CPU_EVALUATION_FAILED_EVIDENCE_PRESERVED',
        'archive':str(target),'sha256':sha(target),'bytes':target.stat().st_size,'payload_members':len(files),
        'scope':'Saved prediction scoring only; no detector inference. Review metrics and source audits before dissertation claims.'}
    write(dest/'backup_receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True);print('UPLOAD THIS ZIP:',target,flush=True)


def verify_reusable_metadata(root, prior):
    checked={}
    for name,record in prior['source_files'].items():
        path=root/name
        digest=prior['derived_sha256'].get(name,record['sha256'])
        assert path.is_file() and sha(path)==digest,'Prepared metadata mismatch: '+name
        if name not in prior['derived_sha256']:assert path.stat().st_size==record['bytes']
        checked[name]={'sha256':digest,'bytes':path.stat().st_size}
    return checked


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code-commit',required=True);p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--reuse-metadata',type=Path,help='Reuse an existing evaluation-only root after checking every file against the pinned reference.')
    args=p.parse_args();base=args.base
    assert base.is_dir(),'Mount the project Drive first'
    assert len(args.code_commit)==40 and all(c in '0123456789abcdef' for c in args.code_commit)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work=Path('/content/fog_cpu_evaluation_20261029')/stamp;out=work/'evidence';out.mkdir(parents=True,exist_ok=False)
    dest=base/'fog_20261029/cpu_evaluation'/stamp;dest.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/Path(__file__).name)
    error=None
    try:
        summary_zip=base/'fog_20261029/full_study/20261029T094333309055Z/fog_full_inference_summary_20261029T125638808108Z.zip'
        print('VERIFYING COMPLETED EIGHT-RUN SUMMARY',flush=True)
        verified_archive(summary_zip,SUMMARY_SHA,extract=work/'summary')
        summary=json.loads((work/'summary/full_inference_audit.json').read_text())
        assert summary['status']=='FULL_FOG_EIGHT_MODEL_CONDITION_OUTPUTS_PRESERVED' and len(summary['runs'])==8
        assert {(r['level'],r['model']) for r in summary['runs']}=={(l,m) for l in range(4) for m in ('lidar','lidar_camera')}
        study_protocol=json.loads((work/'summary/protocol.json').read_text())
        tokens=[r['sample_token'] for r in json.loads((work/'summary/level_0_lidar_input_checks.json').read_text())]
        token_bytes=('\n'.join(tokens)+'\n').encode()
        assert len(tokens)==len(set(tokens))==600
        assert hashlib.sha256(token_bytes).hexdigest()=='ffda09ad54f1381f8bd56d0d204ef72ce66d58a08563dcb4bbe8ca589c89164f'
        inputs=work/'inputs';inputs.mkdir();(inputs/'sample_tokens.txt').write_bytes(token_bytes);(out/'sample_tokens.txt').write_bytes(token_bytes)
        protocol={'status':'FOG_CPU_EVALUATION_PROTOCOL_PRESERVED_BEFORE_METRICS','code_commit':args.code_commit,
            'worker_sha256':WORKER_SHA,'inference_summary_sha256':SUMMARY_SHA,'weather_reference_sha256':WEATHER_SHA,
            'samples':600,'scenes':150,'runs':8,'sample_tokens_sha256':hashlib.sha256(token_bytes).hexdigest(),
            'evaluation':'Unchanged detection_cvpr_2019 configuration, SDK 1.2.0 and source hashes from the successful weather CPU full-split replay. All original GT/annotation velocity links preserved. Load/filter through stock full-val SDK, then exact 600-token GT and prediction subsets before evaluation.',
            'constructor_padding':'Append empty outside-subset predictions solely for stock full-val initialization; remove all outside tokens before scoring. Preserve original 600 prediction iteration order.',
            'reporting':'mAP, NDS, per-class AP, TP errors, own-fresh-baseline changes and fusion-minus-LiDAR differences. Same GT coverage asserted. Descriptive point estimates; no new confidence intervals or significance tests.',
            'limits':'Synthetic fog, one realisation, correlated scenes/logs and separately pretrained pipelines. Not a calibrated shared visibility distance or controlled camera-component ablation.',
            'no_inference':True,'no_gpu_required':True}
        write(out/'evaluation_protocol.json',protocol);shutil.copyfile(out/'evaluation_protocol.json',dest/'evaluation_protocol.json')
        assert sha(dest/'evaluation_protocol.json')==sha(out/'evaluation_protocol.json')
        verified_runs=[]
        for run_info in sorted(summary['runs'],key=lambda r:(r['level'],r['model'])):
            model,level=run_info['model'],run_info['level']
            archive=base/'fog_20261029/full_study/20261029T094333309055Z'/f'level_{level}'/(model+'.zip')
            assert archive.stat().st_size==run_info['bytes']
            print('VERIFYING FULL PREDICTION ARCHIVE:',level,model,flush=True)
            files=verified_archive(archive,run_info['sha256']);assert len(files)==608
            with zipfile.ZipFile(archive) as z:
                assert json.loads(z.read('run_protocol.json'))==study_protocol
                audit=json.loads(z.read('inference_audit.json'))
                assert audit==json.loads((work/f'summary/level_{level}_{model}_inference_audit.json').read_text())
                assert audit['strict_determinism'] and audit['accepted_pilot_prediction_match']
                assert json.loads(z.read('condition_cache_audit.json'))==json.loads((work/f'summary/level_{level}_cache_audit.json').read_text())
                ledger=json.loads(z.read('input_checks.json'));assert [r['sample_token'] for r in ledger]==tokens
                assert hashlib.sha256(z.read('input_checks.json')).hexdigest()==audit['input_checks_sha256']
                for index,token in enumerate(tokens):
                    name=f'frames/{index:04d}_{token}.json'
                    assert files[name]['sha256']==ledger[index]['prediction_sha256']
                    frame=json.loads(z.read(name))
                    for key in ('model','level','worker_sha256','cache_audit_sha256','checkpoint_sha256','annotation_sha256','source_commit'):assert frame['identity'][key]==audit[key]
                    validate_frame(frame,token,index,ledger[index])
                raw=z.read('results_nusc.json');assert hashlib.sha256(raw).hexdigest()==audit['results_sha256']
                prediction=json.loads(raw);assert list(prediction['results'])==tokens
                assert all(all(b['sample_token']==t for b in boxes) for t,boxes in prediction['results'].items())
                target=inputs/f'level_{level}_{model}.json';target.write_bytes(raw)
                verified_runs.append({'level':level,'model':model,'prediction_path':str(target),'prediction_sha256':audit['results_sha256'],
                    'source_archive_sha256':run_info['sha256'],'source_archive_bytes':run_info['bytes'],'verified_payload_members':len(files)})
        write(inputs/'verified_runs.json',verified_runs);write(out/'verified_prediction_inputs.json',verified_runs)
        weather_zip=base/'weather_group_evaluation_20261028/20261028T214858551297Z/weather_group_evaluation_20261028T214858551297Z.zip'
        verified_archive(weather_zip,WEATHER_SHA,'evidence_manifest.json',work/'weather_reference')
        ref=work/'weather_reference'
        helper=ref/'run_weather_groups_colab.py';assert sha(helper)=='9bcc402c745b41e808ebb4ba22e815312d44abe5c8c0f5c87926c68b8e892571'
        assert sha(ref/'requirements-cpu310.txt')=='e7689e47d16cf5b5f5ddd8c4d743d9c527ed9d3f0c0b1806c41a6174b69961e6'
        spec=importlib.util.spec_from_file_location('previous_weather_launcher',helper);previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
        shutil.copyfile(ref/'requirements-cpu310.txt',out/'requirements-cpu310.txt')
        python=previous.setup_environment(out)
        prior=json.loads((ref/'evaluation_metadata_derivation.json').read_text())
        if args.reuse_metadata is not None and args.reuse_metadata.is_dir():
            derived=args.reuse_metadata
            print('RECHECKING PREPARED METADATA; SKIPPING ARCHIVE RESTORATION',flush=True)
            checked=verify_reusable_metadata(derived,prior)
            derivation=prior
            write(out/'metadata_reuse_audit.json',{'status':'ALL_PREPARED_METADATA_FILES_HASH_PASS','root':str(derived),'files':checked,
                'scope':'Fresh hashes of existing evaluation-only files against the verified prior weather reference; no archive restoration or metadata derivation repeated.'})
        else:
            source=work/'source_metadata';source.mkdir()
            expected=json.loads((ref/'reference/metadata_restoration_audit.json').read_text())['metadata_files']
            previous.restore_metadata(source,base/'nuscenes_validation_parts_01_02_verified.zip',expected)
            with (ref/'preparation/sample_membership.csv').open() as f:full_tokens={r['sample_token'] for r in csv.DictReader(f)}
            assert len(full_tokens)==6019
            derived=work/'metadata_evaluation_only'
            derivation=previous.derive_evaluation_metadata(source,derived,full_tokens,expected)
            assert derivation['derived_sha256']==prior['derived_sha256']
            assert derivation['source_files']==prior['source_files']
        write(out/'evaluation_metadata_derivation.json',derivation)
        write(out/'metadata_replay_provenance.json',{'status':'METADATA_IDENTICAL_TO_PREVIOUS_SUCCESSFUL_FULL_REPLAY_ROOT',
            'source_files_hash_checked':True,'derived_keyframe_pose_hashes_match':True,'prior_weather_archive_sha256':WEATHER_SHA,
            'fresh_full_6019_frame_replay_repeated':False,'scope':'Same validated SDK source/configuration and byte-identical evaluation metadata. No changed GT/filter construction; previous full-split replay evidence retained.'})
        for name in ('environment_audit.json','detection_config.json'):shutil.copyfile(ref/'reference'/name,out/('reference_'+name))
        worker=out/'evaluate_fog_cpu.py'
        raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/abjain182002/multimodal-perception/{args.code_commit}/{FOLDER}evaluate_fog_cpu.py',timeout=180).read()
        assert hashlib.sha256(raw).hexdigest()==WORKER_SHA;worker.write_bytes(raw)
        print('STARTING CPU ACCURACY EVALUATION; NO DETECTOR INFERENCE',flush=True)
        run([python,'-u',worker,'--work',work,'--root',derived,'--destination',dest],out/'cpu_evaluation.log')
        assert json.loads((out/'paired_fog_summary.json').read_text())['status']=='ALL_EIGHT_FOG_CPU_EVALUATIONS_PASS'
        success=True
    except BaseException:
        error=traceback.format_exc();write(out/'failure.json',{'traceback':error});success=False
    backup(out,dest,stamp,success)
    if error:raise RuntimeError('CPU stage stopped; failure evidence preserved. No inference needs repeating.')


if __name__=='__main__':main()
