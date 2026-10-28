"""CPU stage 1: freeze description-defined validation groups, without predictions.

Standard library only. Run with Drive mounted. No inference, image extraction,
package installation or full source-archive rehash. Metadata bytes are checked
against the previously preserved full-validation restoration audit.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile

REFERENCE_COMMIT = '3c606fa1cea1b7112442f85f318993705ddfdacf'
BASE = Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation')
TOKEN_BLOBS = {
    'batch_01': '9bf053c015860363b2f6fcdea78dedbc9b7341a1',
    'batch_02': 'd0fba883b8ff3bd7d6aab85fe72c77395691004f',
    'batch_03': '45fecf7c91ecaaf018b55c18195ba1fff12b5203',
}
METADATA_HASHES = {
    'scene.json': 'bdc3211bfbfda652d5e9dab73c45c7132925ca84f869bf45e8e14edeafb1d20b',
    'sample.json': '6035ac58b6e971622be2bb1be15b917e7cb4e05ae984d6b339c27c1699c4ad9d',
}
GROUPS = ('neither_mentioned', 'rain_only_mentioned',
          'night_only_mentioned', 'night_and_rain_mentioned')
PROTOCOL = {
    'stage': 'Description-defined full-validation group membership; no model outcomes read.',
    'population': 'All 6019 previously evaluated samples in 150 official validation scenes; exact preserved batch token lists.',
    'rain_pattern': r'\brain(?:y|ing)?\b',
    'night_pattern': r'\bnight(?:time)?\b',
    'matching': 'Case insensitive; positive word mentions in scene descriptions, not frame-level verified weather.',
    'groups': list(GROUPS),
    'negative_mentions': 'Descriptions containing no/not/without/never followed by a condition word within three words are flagged. Freeze literal membership but do not evaluate until all scene descriptions and flags are reviewed. Amendments must precede group metric access.',
    'review': 'Review the complete 150-scene description inventory before metrics, including synonyms and transitional conditions. Preserve original literal groups if adding a separately documented sensitivity analysis.',
    'primary_analysis': 'Same frozen tokens per pipeline; existing ten-class detection config, GT filters and evaluator. Report group mAP/NDS and per-class AP with GT counts; disclose absent/sparse classes. Full-split replay check before subgroup interpretation.',
    'secondary_analysis': 'Rain-mention and night-mention marginal groups may overlap; do not add counts or treat them as independent. Compare paired pipeline differences within groups before between-group interpretation.',
    'uncertainty': 'Frames within scenes are correlated. Use scene resampling for any confidence intervals; record algorithm, seed and repetitions before computing them. Do not substitute frame bootstrap or average scene AP for pooled group AP.',
    'boundaries': 'Observational metadata strata, not causal effects of weather; no rain-intensity labels; no automatic daylight/dry labels; two separately pretrained pipelines, not a controlled camera ablation. Prior results were already inspected; this is a post-hoc extension, with membership fixed before new subgroup metrics.',
    'next': 'Stage 2 CPU evaluator and reporting after inventory review. Fog and optional snow require separate protocols and inference; not run by this script.',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    data = (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as f:
        f.write(data)
    assert Path(path).read_bytes() == data


def classify(description):
    rain = bool(re.search(PROTOCOL['rain_pattern'], description, re.I))
    night = bool(re.search(PROTOCOL['night_pattern'], description, re.I))
    group = GROUPS[int(rain) + 2 * int(night)]
    negative = bool(re.search(
        r'\b(?:no|not|without|never)\b(?:[\s,]+\w+){0,3}[\s,]+(?:rain(?:y|ing)?|night(?:time)?)\b',
        description, re.I))
    return group, rain, night, negative


def build_inventory(scenes, samples, tokens, expected_samples=6019, expected_scenes=150):
    assert len(tokens) == expected_samples
    scene_map = {s['token']: s for s in scenes}
    assert len(scene_map) == len(scenes), 'Duplicate scene tokens'
    all_samples = {s['token']: s for s in samples}
    assert len(all_samples) == len(samples), 'Duplicate sample tokens'
    assert tokens <= set(all_samples), 'Missing preserved validation samples'
    selected_scene_tokens = {all_samples[t]['scene_token'] for t in tokens}
    assert len(selected_scene_tokens) == expected_scenes
    # Require complete scenes: no availability- or outcome-based frame selection.
    assert {s['token'] for s in samples if s['scene_token'] in selected_scene_tokens} == tokens
    rows, memberships = [], []
    for st in sorted(selected_scene_tokens, key=lambda t: (scene_map[t]['name'], t)):
        scene = scene_map[st]
        frames = sorted((all_samples[t] for t in tokens if all_samples[t]['scene_token'] == st),
                        key=lambda s: (s['timestamp'], s['token']))
        assert len(frames) == scene['nbr_samples'], scene['name']
        group, rain, night, negative = classify(scene['description'])
        rows.append({'scene_name': scene['name'], 'scene_token': st,
                     'log_token': scene['log_token'], 'description': scene['description'],
                     'group': group, 'rain_mentioned': rain, 'night_mentioned': night,
                     'negative_mention_review': negative, 'samples': len(frames)})
        for s in frames:
            memberships.append({'sample_token': s['token'], 'scene_token': st,
                                'scene_name': scene['name'], 'log_token': scene['log_token'],
                                'timestamp': s['timestamp'], 'group': group})
    assert len(memberships) == len({r['sample_token'] for r in memberships}) == expected_samples
    assert len({r['scene_name'] for r in rows}) == expected_scenes
    return rows, memberships


def metadata_bytes(root, archive):
    names = set(METADATA_HASHES)
    if all((root / 'v1.0-trainval' / name).is_file() for name in names):
        return {name: (root / 'v1.0-trainval' / name).read_bytes() for name in names}, 'existing restored metadata'
    assert archive.is_file(), 'Neither restored metadata nor preserved source ZIP exists'
    found = {}
    with zipfile.ZipFile(archive) as z:
        assert z.namelist().count('metadata/v1.0-trainval_meta.tgz') == 1
        with z.open('metadata/v1.0-trainval_meta.tgz') as stream:
            with tarfile.open(fileobj=stream, mode='r|gz') as tar:
                for member in tar:
                    # No filesystem extraction. Accept only the intended table paths.
                    name = member.name.removeprefix('./')
                    if member.isfile() and name in {'v1.0-trainval/' + n for n in names}:
                        short = Path(name).name
                        assert short not in found, 'Duplicate metadata table'
                        print('READING VERIFIED TABLE:', short, flush=True)
                        found[short] = tar.extractfile(member).read()
                        if set(found) == names:
                            break
    assert set(found) == names, 'Metadata tables missing'
    return found, 'tables read from preserved ZIP/TGZ; table hashes verified, whole ZIP not rehashed'


def write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--root', type=Path, default=Path('/content/datasets/nuscenes_full'))
    p.add_argument('--work', type=Path, default=Path('/content/weather_groups_20261028'))
    args = p.parse_args()
    assert args.base.is_dir(), 'Mount Drive first; the preserved project folder must exist'
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.work / stamp
    dest = args.base / 'weather_groups_20261028' / stamp
    out.mkdir(parents=True, exist_ok=False)
    dest.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protocol.json', PROTOCOL)
    shutil.copyfile(out / 'protocol.json', dest / 'protocol.json')
    assert file_sha(out / 'protocol.json') == file_sha(dest / 'protocol.json')
    print('PROTOCOL PRESERVED BEFORE METADATA GROUPING', flush=True)
    tokens, references = set(), {}
    for batch, blob in TOKEN_BLOBS.items():
        path = f'results/nuscenes/bevfusion/full_validation/original_evidence/inference/{batch}/sample_tokens.txt'
        url = f'https://raw.githubusercontent.com/abjain182002/multimodal-perception/{REFERENCE_COMMIT}/{path}'
        data = urllib.request.urlopen(url, timeout=90).read()
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blob
        lines = data.decode().splitlines()
        assert all(re.fullmatch('[0-9a-f]{32}', x) for x in lines)
        assert len(set(lines)) == len(lines) and not tokens.intersection(lines)
        tokens.update(lines)
        references[batch] = {'url': url, 'git_blob_sha1': blob, 'sha256': sha(data), 'samples': len(lines)}
    assert len(tokens) == 6019
    raw, source = metadata_bytes(args.root, args.base / 'nuscenes_validation_parts_01_02_verified.zip')
    for name, expected in METADATA_HASHES.items():
        assert sha(raw[name]) == expected, f'Metadata hash mismatch: {name}'
    rows, members = build_inventory(json.loads(raw['scene.json']), json.loads(raw['sample.json']), tokens)
    write_csv(out / 'scene_inventory.csv', rows)
    write_csv(out / 'sample_membership.csv', members)
    counts = {g: {'scenes': sum(r['group'] == g for r in rows),
                  'samples': sum(r['group'] == g for r in members),
                  'logs': len({r['log_token'] for r in rows if r['group'] == g})} for g in GROUPS}
    assert sum(v['samples'] for v in counts.values()) == 6019
    assert sum(v['scenes'] for v in counts.values()) == 150
    predictions = {}
    for model in ('lidar', 'lidar_camera'):
        f = args.base / 'full_validation_merged' / model / 'results_nusc.json'
        predictions[model] = {'path': str(f), 'exists': f.is_file(),
                              'bytes': f.stat().st_size if f.is_file() else None,
                              'opened': False, 'hash_verified_this_stage': False}
    report = {
        'status': 'WEATHER_GROUP_MEMBERSHIP_FROZEN_REVIEW_PENDING',
        'samples': 6019, 'scenes': 150, 'groups': counts,
        'negative_mention_flags': [r['scene_name'] for r in rows if r['negative_mention_review']],
        'review_all_scene_descriptions_before_evaluation': True,
        'metadata_source': source, 'metadata_sha256': METADATA_HASHES,
        'preserved_population_sources': references, 'predictions': predictions,
        'known_cpu_python_candidates': [str(f) for f in (
            Path('/content/camera_challenges_cpu310/bin/python'),
            Path('/content/full_validation_cpu310/bin/python')) if f.is_file()],
        'restored_metadata_directory_exists': (args.root / 'v1.0-trainval').is_dir(),
        'python': platform.python_version(), 'script_sha256': file_sha(__file__),
        'protocol_sha256': file_sha(out / 'protocol.json'),
        'elapsed_seconds': round(time.monotonic() - started, 2),
        'scope': 'Membership and file availability only. No predictions opened, no AP/NDS evaluation, no GPU inference. Full source ZIP not rehashed; selected metadata table bytes checked against preserved restoration hashes.',
    }
    write_json(out / 'preparation_audit.json', report)
    shutil.copyfile(__file__, out / 'prepare_weather_groups.py')
    payload = {f.name: {'bytes': f.stat().st_size, 'sha256': file_sha(f)} for f in out.iterdir() if f.is_file()}
    write_json(out / 'manifest.json', {'files': payload})
    archive = out / f'weather_group_preparation_{stamp}.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted([*payload, 'manifest.json']):
            z.write(out / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, record in payload.items():
            data = z.read(name)
            assert len(data) == record['bytes'] and sha(data) == record['sha256']
    target = dest / archive.name
    with archive.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    assert file_sha(archive) == file_sha(target)
    receipt = {'status': 'WEATHER_GROUP_PREPARATION_AND_DRIVE_COPY_PASS',
               'archive': str(target), 'bytes': target.stat().st_size,
               'sha256': file_sha(target), 'groups': counts,
               'next': 'Upload this ZIP for description review and stage-2 CPU evaluation setup.'}
    write_json(dest / 'backup_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    print('UPLOAD THIS ZIP:', target, flush=True)


if __name__ == '__main__':
    main()
