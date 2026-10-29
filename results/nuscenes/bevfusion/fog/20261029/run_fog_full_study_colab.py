"""Run four fog conditions x two pinned models in the retained Colab session.

One shared condition cache at a time; completed model outputs are hash-verified
on Drive before that cache is removed. Streams progress, no reinstall/training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request
import zipfile

BASE = Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation')
PRIOR = Path('/content/fog_gpu_smoke_20261029/20261029T080852328937Z')
WORK = Path('/content/fog_full_study_20261029')
FOLDER = 'results/nuscenes/bevfusion/fog/20261029/'
WORKER_SHA = '912a7ae4d50787f500f6e18734b8354876ca8b791eb3ff14b0167de5bb76b8df'
RESTORATION_SHA = '725a06181884d0af1eb43646e43bdb472562ca2f9737402de83a6661c54e5402'
SMOKE_SHA = 'a3127d6776fd175d5466a3d7272234efcdf3cd3df0aa60cff12d5ec739b3ba6d'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def check_zip(path, expected):
    assert sha(path) == expected
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        records = json.loads(z.read('manifest.json'))['files']
        assert set(z.namelist()) == set(records) | {'manifest.json'}
        for name, record in records.items():
            data = z.read(name)
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
    return records


def unpack_known(path, expected, root):
    records = check_zip(path, expected)
    with zipfile.ZipFile(path) as z:
        for name in records:
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(name)
            if target.exists(): assert target.read_bytes() == data
            else: target.write_bytes(data)
    return records


def archive_folder(folder, target):
    """Immutable artifact snapshot, without cache files or raw sensor inputs."""
    records = {str(p.relative_to(folder)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
        for p in sorted(folder.rglob('*')) if p.is_file() and not p.name.endswith('.partial') and p.name != 'manifest.json'}
    manifest = json.dumps({'files': records}, indent=2).encode() + b'\n'
    temporary = target.with_name(target.name + '.partial'); target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name in records: z.write(folder / name, name)
        z.writestr('manifest.json', manifest)
    digest = sha(temporary); check_zip(temporary, digest)
    assert not target.exists(), 'Refuse to overwrite an existing immutable artifact'
    temporary.replace(target)
    return {'archive': str(target), 'bytes': target.stat().st_size, 'sha256': digest, 'payload_members': len(records)}


def preserve_model(folder, destination, complete=True):
    local_archive = folder.parent / (folder.name + '.zip')
    if local_archive.exists(): local_archive.unlink()  # Own reproducible local archive only.
    receipt = archive_folder(folder, local_archive)
    target = destination / local_archive.name; assert not target.exists()
    pending = target.with_name(target.name + '.partial')
    with local_archive.open('rb') as src, pending.open('wb') as dst: shutil.copyfileobj(src, dst)
    check_zip(pending, receipt['sha256']); pending.replace(target)
    receipt['archive'] = str(target); receipt['status'] = 'FULL_FOG_MODEL_OUTPUT_AND_DRIVE_COPY_PASS' if complete else 'PARTIAL_FOG_MODEL_OUTPUTS_PRESERVED'
    atomic(destination / (folder.name + '_receipt.json'), receipt)
    return receipt



def collect_pair(level_dest, protocol, summary_dir, level):
    ledgers = {}
    for model in ('lidar', 'lidar_camera'):
        receipt = json.loads((level_dest / (model+'_receipt.json')).read_text())
        with zipfile.ZipFile(receipt['archive']) as z:
            assert json.loads(z.read('run_protocol.json')) == protocol
            audit = json.loads(z.read('inference_audit.json'))
            assert audit['status'] == 'FOG_FULL_MODEL_CONDITION_INFERENCE_PASS'
            assert audit['model'] == model and audit['level'] == level and audit['samples'] == 600
            assert audit['worker_sha256'] == WORKER_SHA and audit['strict_determinism'] and audit['accepted_pilot_prediction_match']
            assert hashlib.sha256(z.read('results_nusc.json')).hexdigest() == audit['results_sha256']
            assert hashlib.sha256(z.read('input_checks.json')).hexdigest() == audit['input_checks_sha256']
            ledgers[model] = json.loads(z.read('input_checks.json'))
            atomic(summary_dir / f'level_{level}_{model}_inference_audit.json', audit)
            atomic(summary_dir / f'level_{level}_{model}_input_checks.json', ledgers[model])
    assert len(ledgers['lidar']) == len(ledgers['lidar_camera']) == 600
    tokens = (WORK / 'restoration/sample_tokens.txt').read_text().splitlines()
    for model in ledgers: assert [r['sample_token'] for r in ledgers[model]] == tokens
    for left, right in zip(ledgers['lidar'], ledgers['lidar_camera']):
        assert left['sample_token'] == right['sample_token'] and left['packed_points_sha256'] == right['packed_points_sha256']
    atomic(summary_dir / f'level_{level}_paired_check.json', {'status':'FULL_FOG_CONDITION_PAIRED_INPUTS_PASS','level':level,'samples':600,'same_packed_points_all_frames':True})

def run(command, log, cwd, env):
    print('RUNNING:', ' '.join(map(str, command)), flush=True)
    with log.open('a') as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout: print(line, end='', flush=True); stream.write(line); stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
    assert code == 0, f'Worker stopped ({code}); see {log}'


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--code-commit', required=True)
    args = parser.parse_args()
    assert len(args.code_commit) == 40 and all(c in '0123456789abcdef' for c in args.code_commit)
    assert BASE.is_dir() and PRIOR.is_dir(), 'Retain the existing Colab runtime and mounted project Drive'
    runtime = Path('/content/fog_runtime'); python = runtime / 'bevfusion310/bin/python'; repo = runtime / 'mmdetection3d'
    assert python.is_file() and shutil.which('nvidia-smi'), 'Existing installed GPU runtime required'
    WORK.mkdir(exist_ok=True)
    identity = {'code_commit': args.code_commit, 'coordinator_sha256': sha(__file__), 'worker_sha256': WORKER_SHA,
        'restoration_zip_sha256': RESTORATION_SHA, 'accepted_smoke_zip_sha256': SMOKE_SHA}
    state_path = WORK / 'run_identity.json'
    if state_path.exists():
        state = json.loads(state_path.read_text()); assert state['identity'] == identity, 'Existing study uses different code/protocol'
    else:
        state = {'identity': identity, 'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}; atomic(state_path, state)
    dest = BASE / 'fog_20261029/full_study' / state['run_id']; dest.mkdir(parents=True, exist_ok=True)
    summary_dir = WORK / 'summary'; summary_dir.mkdir(exist_ok=True)
    error = None; current_folder = None
    try:
        print('VERIFYING ACCEPTED SMOKE AND FULL-INPUT RESTORATION', flush=True)
        restoration = BASE / 'fog_20261029/full_inputs/20261029T090602713295Z/fog_full_input_restoration_20261029T090602713295Z.zip'
        unpack_known(restoration, RESTORATION_SHA, WORK / 'restoration')
        smoke = BASE / 'fog_20261029/gpu_smoke_deterministic/20261029T085752261528Z/fog_gpu_smoke_deterministic_20261029T085752261528Z.zip'
        unpack_known(smoke, SMOKE_SHA, WORK / 'smoke')
        restored = json.loads((WORK / 'restoration/restoration_audit.json').read_text())
        assert restored['verified_sensor_files'] == 9600 and restored['samples'] == 600 and restored['scenes'] == 150
        assert restored['dataset_root'] == '/content/fog_full_inputs_20261029/dataset'
        # All original files are freshly checked again here before new corruption/inference.
        for index, (relative, record) in enumerate(sorted(restored['members'].items()), 1):
            path = Path(restored['dataset_root']) / relative
            assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
            if index % 1000 == 0 or index == 9600: print(f'ORIGINAL CONTENT RECHECK: {index}/9600', flush=True)
        worker = WORK / 'fog_full_worker.py'
        data = urllib.request.urlopen(f'https://raw.githubusercontent.com/abjain182002/multimodal-perception/{args.code_commit}/{FOLDER}fog_full_worker.py', timeout=180).read()
        assert hashlib.sha256(data).hexdigest() == WORKER_SHA
        if worker.exists(): assert worker.read_bytes() == data
        else: worker.write_bytes(data)
        shutil.copyfile(__file__, summary_dir / 'run_fog_full_study_colab.py'); shutil.copyfile(worker, summary_dir / 'fog_full_worker.py')
        operations = json.loads((runtime / 'setup/cuda_operations_audit.json').read_text())
        for extension in operations['extensions'].values(): assert sha(extension['path']) == extension['sha256']
        for name in ('cuda_operations_audit.json','environment.txt'): shutil.copyfile(runtime / 'setup' / name, summary_dir / name)
        protocol = {'status': 'FULL_FOG_STUDY_IMPLEMENTATION_FIXED_BEFORE_FULL_INFERENCE', **identity,
            'sample_tokens_sha256': sha(WORK / 'restoration/sample_tokens.txt'), 'samples': 600, 'scenes': 150,
            'conditions': [0,1,2,3], 'models': ['lidar','lidar_camera'], 'model_condition_runs': 8,
            'source_commit': 'fe25f7a51d36e3702f961e198894580d83c4387b',
            'controls': 'Same pinned checkpoints and original evaluator/GT filters; all six cameras and current + nine raw LiDAR scans. No population or threshold selection, no training. Strict PyTorch determinism, warn_only False, cuDNN deterministic True, TF32/benchmark off, no autocast, seed 20261029, batch 1.',
            'corruption': 'Previously CPU-validated adapter/RNG mapping and lossless PNG for baseline and all fog levels. One seeded realisation per raw file/severity shared between models. Corrupt each raw scan before aggregation/projection/voxelisation. Pilot caches and predictions must match accepted smoke at original tolerances.',
            'storage': 'Prepare one shared condition cache; save/hash-verify each completed model artifact on Drive. Remove only that regenerable condition cache after both outputs are preserved. Original restored sensors and pilot retained.',
            'resume': 'Within retained session, verify immutable completed Drive artifacts and skip them. Existing local frame outputs/caches are reused only under matching identity and input hashes. Partial local work requires this session to remain alive.',
            'evaluation_scope': 'This launch performs inference and pinned upstream NuScenes JSON formatting only. Unchanged CPU AP/NDS evaluation is a subsequent stage.',
            'limits': 'Synthetic model, not measured natural fog or a calibrated shared visibility distance. Two separately pretrained models; no controlled camera-component ablation. One realisation; correlated scenes/logs. Historical default-execution predictions remain separate from these fresh strict baselines.'}
        if (summary_dir / 'protocol.json').exists(): assert json.loads((summary_dir / 'protocol.json').read_text()) == protocol
        else: atomic(summary_dir / 'protocol.json', protocol)
        if (dest / 'protocol.json').exists(): assert (dest / 'protocol.json').read_bytes() == (summary_dir / 'protocol.json').read_bytes()
        else: shutil.copyfile(summary_dir / 'protocol.json', dest / 'protocol.json')
        assert sha(dest / 'protocol.json') == sha(summary_dir / 'protocol.json')
        print('FULL STUDY PROTOCOL PRESERVED ON DRIVE BEFORE INFERENCE', flush=True)
        env = os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'0', 'CUBLAS_WORKSPACE_CONFIG':':4096:8',
            'CUDA_HOME':str(runtime / 'cuda-12.1'), 'MPLBACKEND':'Agg', 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
            'PYTHONPATH':str(repo)+':'+str(PRIOR / 'evidence/code')+':'+env.get('PYTHONPATH',''),
            'LD_LIBRARY_PATH':str(runtime / 'cuda-12.1/lib64')+':'+env.get('LD_LIBRARY_PATH',''),
            'PATH':str(runtime / 'cuda-12.1/bin')+':'+env.get('PATH','')})
        receipts = []
        for level in (0,1,2,3):
            level_root = WORK / f'level_{level}'; level_root.mkdir(exist_ok=True)
            level_dest = dest / f'level_{level}'; level_dest.mkdir(exist_ok=True)
            complete = []
            for model in ('lidar','lidar_camera'):
                saved_receipt = level_dest / (model + '_receipt.json')
                orphan = level_dest / (model + '.zip')
                if not saved_receipt.exists() and orphan.is_file():
                    digest = sha(orphan); members = check_zip(orphan, digest)
                    with zipfile.ZipFile(orphan) as z:
                        assert json.loads(z.read('run_protocol.json')) == protocol
                        audit = json.loads(z.read('inference_audit.json'))
                        assert audit['status'] == 'FOG_FULL_MODEL_CONDITION_INFERENCE_PASS' and audit['model'] == model and audit['level'] == level
                    atomic(saved_receipt, {'archive':str(orphan),'sha256':digest,'bytes':orphan.stat().st_size,'payload_members':len(members),'status':'FULL_FOG_MODEL_OUTPUT_AND_DRIVE_COPY_PASS'})
                if saved_receipt.exists():
                    receipt = json.loads(saved_receipt.read_text()); check_zip(Path(receipt['archive']), receipt['sha256'])
                    assert receipt['status'] == 'FULL_FOG_MODEL_OUTPUT_AND_DRIVE_COPY_PASS'
                    complete.append(model); receipts.append({'level':level,'model':model,**receipt})
            if len(complete) == 2:
                collect_pair(level_dest, protocol, summary_dir, level)
                previous_cache = WORK / 'cache' / f'level_{level}'
                if previous_cache.is_dir(): shutil.rmtree(previous_cache)
                print('REUSING VERIFIED COMPLETED CONDITION:',level,flush=True); continue
            cache = WORK / 'cache' / f'level_{level}'; cache.parent.mkdir(exist_ok=True)
            print('PREPARING SHARED CONDITION:',level,flush=True)
            run([python,'-u',worker,'prepare','--level',level,'--cache',cache,'--restoration',WORK / 'restoration/restoration_audit.json','--output',level_root], level_root / 'preparation.log', repo, env)
            shutil.copyfile(level_root / 'cache_audit.json', summary_dir / f'level_{level}_cache_audit.json')
            ledgers = {}
            for model in ('lidar','lidar_camera'):
                model_out = level_root / model; model_out.mkdir(exist_ok=True)
                if model not in complete:
                    current_folder = model_out
                    shutil.copyfile(summary_dir / 'protocol.json', model_out / 'run_protocol.json')
                    shutil.copyfile(level_root / 'cache_audit.json', model_out / 'condition_cache_audit.json')
                    run([python,'-u',worker,'predict','--level',level,'--output',model_out,'--model',model,'--smoke',WORK / 'smoke'], model_out / 'inference.log', repo, env)
                    receipt = preserve_model(model_out, level_dest); receipts.append({'level':level,'model':model,**receipt})
                    print('MODEL OUTPUT PRESERVED ON DRIVE:',level,model,flush=True)
                else:
                    receipt = json.loads((level_dest / (model+'_receipt.json')).read_text())
            collect_pair(level_dest, protocol, summary_dir, level)
            atomic(summary_dir / 'completed_models.json',receipts)
            # Both immutable Drive artifacts have been verified above; only own regenerable cache is removed.
            assert cache.parent == WORK / 'cache'; shutil.rmtree(cache)
            current_folder = None
            print('CONDITION COMPLETE; TEMPORARY CACHE RELEASED:',level,flush=True)
        assert len(receipts) == 8
        atomic(summary_dir / 'full_inference_audit.json',{'status':'FULL_FOG_EIGHT_MODEL_CONDITION_OUTPUTS_PRESERVED',
            'samples_per_run':600,'runs':receipts,'accuracy_evaluated':False,'scope':'Full inference and export/preservation complete. CPU AP/NDS evaluation and interpretation remain pending.'})
    except BaseException:
        error = traceback.format_exc(); atomic(summary_dir / ('failure_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json'),{'traceback':error})
        if current_folder is not None and current_folder.is_dir():
            failure_dest = dest / ('partial_failure_'+datetime.now(timezone.utc).strftime('%H%M%S')); failure_dest.mkdir()
            try: preserve_model(current_folder,failure_dest,complete=False)
            except Exception as backup_error: print('PARTIAL OUTPUT BACKUP ERROR:',repr(backup_error),flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = WORK / ('fog_full_inference_summary_'+stamp+'.zip')
    receipt = archive_folder(summary_dir,archive)
    target = dest / archive.name
    with archive.open('rb') as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
    check_zip(target,receipt['sha256']); receipt['archive']=str(target)
    receipt['status']='FULL_FOG_INFERENCE_SUMMARY_AND_DRIVE_COPY_PASS' if error is None else 'FULL_FOG_INFERENCE_FAILED_EVIDENCE_PRESERVED'
    atomic(dest / ('summary_receipt_'+stamp+'.json'),receipt)
    print(json.dumps(receipt,indent=2),flush=True);print('UPLOAD THIS ZIP:',target,flush=True)
    if error: raise RuntimeError('Full study stopped; evidence preserved. Keep this runtime for recovery.')


if __name__ == '__main__': main()
