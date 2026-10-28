"""CPU stage 2: restore verified metadata, evaluate saved predictions, back up.

Run with Drive mounted; no detector packages or GPU inference. Isolated Python
3.10 environment. Original archives, predictions and metadata remain unchanged.
"""
import argparse
import codecs
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

REFERENCE_COMMIT = 'e4bbc648a8b61f1688a55aca1e24fa6835464428'
PREPARATION_SHA = 'd9aa5377247df83162cbfd8126638e4ded39e6459f036e378cf4f35cc47ad1a0'
WORKER_SHA = 'e1787b690cd7639b85489b438487849d23c110e4f0daf9ed187bca615b5fea08'
REQUIREMENTS_SHA = 'e7689e47d16cf5b5f5ddd8c4d743d9c527ed9d3f0c0b1806c41a6174b69961e6'
ROOT_URL = 'https://raw.githubusercontent.com/abjain182002/multimodal-perception/'
FOLDER = 'results/nuscenes/bevfusion/weather_groups/20261028/'
REVIEW = {
    'status': 'ALL_150_SCENE_DESCRIPTIONS_REVIEWED_LITERAL_GROUPS_RETAINED',
    'preparation_zip_sha256': PREPARATION_SHA,
    'review_scope': 'All 150 supplied scene descriptions inspected before new subgroup metrics. No negated rain/night mentions or alternative condition wording requiring amendment found. No image-level weather verification.',
    'groups': {
        'neither_mentioned': {'scenes': 111, 'samples': 4449, 'logs': 15},
        'rain_only_mentioned': {'scenes': 24, 'samples': 968, 'logs': 2},
        'night_only_mentioned': {'scenes': 12, 'samples': 482, 'logs': 1},
        'night_and_rain_mentioned': {'scenes': 3, 'samples': 120, 'logs': 1}},
    'unique_logs': 18,
    'night_log': 'e55205b1f2894b49957905d7ddfdb96d',
    'night_log_shares_groups': ['night_only_mentioned', 'night_and_rain_mentioned'],
    'interpretation': 'No mention is not verified dry/daylight. Scene descriptions do not establish rain throughout every frame. All 15 night scenes are from one log; natural groups are observational and confounded.',
    'uncertainty_decision_before_metrics': 'Report descriptive point estimates and coverage, no inferential confidence intervals or significance tests. Scene bootstrap would not capture between-log generalisation with one night log or two rain-only logs.',
    'reporting_amendment_before_metrics': 'Retain ten-class SDK mAP/NDS, explicitly show absent/sparse classes, and provide separately labelled represented-class and common-GT-class macro-AP sensitivity summaries. The common class set depends only on GT coverage, not model outcomes.'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n')


def download(url, path):
    data = urllib.request.urlopen(url, timeout=120).read()
    with Path(path).open('xb') as f:
        f.write(data)
    return data


def run_logged(command, log):
    print('RUNNING:', ' '.join(map(str, command)), flush=True)
    with Path(log).open('x') as f:
        proc = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True); f.write(line); f.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    assert code == 0, f'Command failed ({code}); log retained: {log}'


def array_rows(stream):
    """Bounded-buffer JSON-array reader; retains all source numeric values."""
    reader = codecs.getreader('utf-8')(stream)
    decoder = json.JSONDecoder(); buffer = ''; pos = 0; started = False
    while True:
        while True:
            while pos < len(buffer) and buffer[pos].isspace(): pos += 1
            if not started:
                if pos == len(buffer): break
                assert buffer[pos] == '['
                pos += 1; started = True; continue
            while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] == ','): pos += 1
            if pos == len(buffer): break
            if buffer[pos] == ']':
                assert not (buffer[pos + 1:] + reader.read()).strip()
                return
            try: row, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError: break
            yield row; pos = end
        chunk = reader.read(1024 * 1024)
        if not chunk: raise ValueError('Incomplete JSON array')
        buffer = buffer[pos:] + chunk; pos = 0


def restore_metadata(root, archive, expected):
    missing = set()
    for name, record in expected.items():
        f = root / name
        if f.exists():
            assert f.stat().st_size == record['bytes'] and file_sha(f) == record['sha256'], name
        else:
            missing.add(name)
    if not missing:
        return
    assert archive.is_file(), f'Missing preserved metadata source: {archive}'
    with zipfile.ZipFile(archive) as z:
        assert z.namelist().count('metadata/v1.0-trainval_meta.tgz') == 1
        with z.open('metadata/v1.0-trainval_meta.tgz') as compressed:
            with tarfile.open(fileobj=compressed, mode='r|gz') as tar:
                for member in tar:
                    name = member.name.removeprefix('./')
                    if not member.isfile() or name not in missing:
                        continue
                    record = expected[name]
                    assert member.size == record['bytes'], name
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(target.name + '.weather_partial')
                    print('RESTORING VERIFIED METADATA:', name, flush=True)
                    h = hashlib.sha256()
                    with tar.extractfile(member) as src, temp.open('xb') as dst:
                        for b in iter(lambda: src.read(8 * 1024 * 1024), b''):
                            h.update(b); dst.write(b)
                    assert temp.stat().st_size == record['bytes'] and h.hexdigest() == record['sha256'], name
                    temp.replace(target)
                    missing.remove(name)
                    if not missing: break
    assert not missing, f'Missing metadata members: {missing}'


def derive_evaluation_metadata(source, target, tokens, expected):
    """Keep annotations/trajectories unchanged; prune only unused sensor/pose rows.

    Retain all scene/sample/annotation records so annotation velocity links remain
    intact. Official evaluation needs current keyframes/ego poses for val only.
    This root is strictly for evaluation, never inference or training.
    """
    target.mkdir(parents=True, exist_ok=False)
    for name in expected:
        if name in ('v1.0-trainval/sample_data.json', 'v1.0-trainval/ego_pose.json'):
            continue
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, dest)
        assert file_sha(dest) == expected[name]['sha256'], name
    poses, seen = set(), set()
    counts = {}
    sd_name = 'v1.0-trainval/sample_data.json'
    with (source / sd_name).open('rb') as src, (target / sd_name).open('x') as dst:
        dst.write('['); selected = 0; scanned = 0
        for row in array_rows(src):
            scanned += 1
            if row['sample_token'] in tokens and row['is_key_frame']:
                if selected: dst.write(',')
                json.dump(row, dst, separators=(',', ':'))
                selected += 1; poses.add(row['ego_pose_token']); seen.add(row['sample_token'])
            if scanned % 400000 == 0:
                print('SENSOR ROWS SCANNED:', scanned, 'KEYFRAMES RETAINED:', selected, flush=True)
        dst.write(']\n')
    assert seen == tokens, 'Some validation frames have no current sensor keyframe'
    counts[sd_name] = {'source_rows': scanned, 'retained_rows': selected}
    ep_name = 'v1.0-trainval/ego_pose.json'
    retained = set()
    with (source / ep_name).open('rb') as src, (target / ep_name).open('x') as dst:
        dst.write('['); selected = 0; scanned = 0
        for row in array_rows(src):
            scanned += 1
            if row['token'] in poses:
                assert row['token'] not in retained
                if selected: dst.write(',')
                json.dump(row, dst, separators=(',', ':'))
                selected += 1; retained.add(row['token'])
        dst.write(']\n')
    assert retained == poses
    counts[ep_name] = {'source_rows': scanned, 'retained_rows': selected}
    return {'scope': 'Evaluation-only root. All annotations, temporal annotation links, samples and scenes unchanged. Retain all validation keyframes and referenced poses; no raw sensors required.',
            'row_counts': counts, 'source_files': expected,
            'derived_sha256': {n: file_sha(target / n) for n in counts},
            'acceptance_check': 'Full-split original summary and detailed metrics must match within 1e-12 before subgroup evaluation.'}


def verify_preparation(archive, out):
    assert file_sha(archive) == PREPARATION_SHA
    out.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('manifest.json'))['files']
        assert set(z.namelist()) == set(manifest) | {'manifest.json'}
        for name, record in manifest.items():
            assert Path(name).name == name and name not in ('.', '..')
            data = z.read(name)
            assert len(data) == record['bytes'] and sha(data) == record['sha256']
            (out / name).write_bytes(data)
        (out / 'manifest.json').write_bytes(z.read('manifest.json'))
    audit = json.loads((out / 'preparation_audit.json').read_text())
    assert audit['groups'] == REVIEW['groups']
    assert audit['negative_mention_flags'] == []
    with (out / 'sample_membership.csv').open() as f:
        rows = list(csv.DictReader(f))
    tokens = {r['sample_token'] for r in rows}
    assert len(rows) == len(tokens) == 6019
    return tokens


def setup_environment(work):
    env = Path('/content/weather_groups_cpu310')
    python = env / 'bin/python'
    if not python.is_file():
        run_logged([sys.executable, '-m', 'pip', 'install', 'uv==0.13.4'], work / 'install_uv.log')
        uv = shutil.which('uv')
        assert uv, 'uv executable missing after installation'
        run_logged([uv, 'venv', '--python', '3.10.22', '--managed-python', str(env)], work / 'create_cpu_env.log')
    else:
        uv = shutil.which('uv')
        if not uv:
            run_logged([sys.executable, '-m', 'pip', 'install', 'uv==0.13.4'], work / 'install_uv.log')
            uv = shutil.which('uv')
    run_logged([uv, 'pip', 'install', '--python', str(python), '-r',
                work / 'requirements-cpu310.txt'], work / 'install_cpu_packages.log')
    run_logged([uv, 'pip', 'freeze', '--python', str(python)], work / 'cpu_package_freeze.txt')
    return python


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-commit', required=True)
    p.add_argument('--base', type=Path, default=Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation'))
    p.add_argument('--preparation', type=Path)
    args = p.parse_args()
    assert len(args.code_commit) == 40 and all(c in '0123456789abcdef' for c in args.code_commit)
    assert args.base.is_dir(), 'Mount Drive first'
    preparation = args.preparation or (args.base / 'weather_groups_20261028/20261028T213135530171Z/weather_group_preparation_20261028T213135530171Z.zip')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent = Path('/content/weather_groups_evaluation_20261028') / stamp
    output = parent / 'evidence'
    output.mkdir(parents=True, exist_ok=False)
    destination = args.base / 'weather_group_evaluation_20261028' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    tokens = verify_preparation(preparation, output / 'preparation')
    write_json(output / 'description_review.json', REVIEW)
    shutil.copyfile(output / 'description_review.json', destination / 'description_review.json')
    assert file_sha(output / 'description_review.json') == file_sha(destination / 'description_review.json')
    print('DESCRIPTION REVIEW AND ANALYSIS DECISIONS PRESERVED BEFORE METRIC ACCESS', flush=True)
    shutil.copyfile(__file__, output / 'run_weather_groups_colab.py')
    worker = output / 'evaluate_weather_groups.py'
    code = download(ROOT_URL + args.code_commit + '/' + FOLDER + worker.name, worker)
    assert sha(code) == WORKER_SHA
    requirements = download(ROOT_URL + args.code_commit + '/' + FOLDER + 'requirements-cpu310.txt',
                            output / 'requirements-cpu310.txt')
    assert sha(requirements) == REQUIREMENTS_SHA
    reference = output / 'reference'
    reference.mkdir()
    requests = {
        'environment_audit.json': 'results/nuscenes/bevfusion/full_validation/replay_20261020/original_evidence/setup/environment_audit.json',
        'metadata_restoration_audit.json': 'results/nuscenes/bevfusion/full_validation/replay_20261020/original_evidence/setup/metadata_restoration_audit.json',
        'detection_config.json': 'results/nuscenes/bevfusion/full_validation/original_evidence/evaluation/detection_config.json'}
    for model in ('lidar', 'lidar_camera'):
        (reference / model).mkdir()
        for name in ('evaluation_audit.json', 'metrics_summary.json', 'metrics_details.json'):
            requests[model + '/' + name] = 'results/nuscenes/bevfusion/full_validation/original_evidence/evaluation/' + model + '/' + name
    downloads = {}
    for name, relative in requests.items():
        data = download(ROOT_URL + REFERENCE_COMMIT + '/' + relative, reference / name)
        downloads[name] = {'reference_commit': REFERENCE_COMMIT, 'path': relative, 'sha256': sha(data)}
    for model in ('lidar', 'lidar_camera'):
        audit = json.loads((reference / model / 'evaluation_audit.json').read_text())
        for name, key in [('metrics_summary.json', 'metrics_summary_sha256'), ('metrics_details.json', 'metrics_details_sha256')]:
            assert file_sha(reference / model / name) == audit[key]
    write_json(output / 'reference_downloads.json', downloads)
    python = setup_environment(output)
    restored = parent / 'source_metadata'
    restored.mkdir()
    record = json.loads((reference / 'metadata_restoration_audit.json').read_text())
    restore_metadata(restored, args.base / 'nuscenes_validation_parts_01_02_verified.zip', record['metadata_files'])
    derived = parent / 'metadata_evaluation_only'
    derivation = derive_evaluation_metadata(restored, derived, tokens, record['metadata_files'])
    write_json(output / 'evaluation_metadata_derivation.json', derivation)
    for model in ('lidar', 'lidar_camera'):
        run_logged([python, '-u', worker, '--work', output, '--root', derived,
                    '--base', args.base, '--model', model], output / (model + '_evaluation.log'))
        # Preserve each completed model before starting the next one.
        shutil.copytree(output / model, destination / model)
        for f in (output / model).rglob('*'):
            if f.is_file():
                assert file_sha(f) == file_sha(destination / model / f.relative_to(output / model))
    run_logged([python, '-u', worker, '--work', output, '--report'], output / 'report_generation.log')
    files = {str(f.relative_to(output)): {'bytes': f.stat().st_size, 'sha256': file_sha(f)}
             for f in sorted(output.rglob('*')) if f.is_file()}
    write_json(output / 'evidence_manifest.json', {'files': files})
    archive = parent / ('weather_group_evaluation_' + stamp + '.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in [*files, 'evidence_manifest.json']:
            z.write(output / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, record in files.items():
            data = z.read(name)
            assert len(data) == record['bytes'] and sha(data) == record['sha256']
    target = destination / archive.name
    with archive.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    assert file_sha(archive) == file_sha(target)
    receipt = {'status': 'WEATHER_GROUP_CPU_EVALUATION_AND_DRIVE_COPY_PASS',
               'archive': str(target), 'bytes': target.stat().st_size, 'sha256': file_sha(target),
               'payload_members': len(files), 'code_commit': args.code_commit,
               'scope': 'Saved predictions; no inference. Both full-validation replays and four description-group evaluations per model passed. Descriptive subgroup results; no causal weather claim.'}
    write_json(destination / 'backup_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print('UPLOAD THIS ZIP:', target, flush=True)


if __name__ == '__main__':
    main()
