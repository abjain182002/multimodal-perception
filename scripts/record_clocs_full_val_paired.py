"""Verify and record the 3769-frame CLOCs paired validation diagnostic.

The historical PointPillars baseline did not reproduce in this run. This
script deliberately records that failure alongside the paired comparison.
Large per-frame predictions remain in the downloaded ZIP outside Git.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


EXPECTED_ARCHIVE_SHA256 = (
    '1b034d06f49c343dfc4a21c80339481e37db455f237a19f3db3764d3a1306413'
)
EXPECTED_MANIFEST_SHA256 = (
    '2f68c2591d41aa320f1d4abe65a3e73e4af05212435a81bf83327203704327a7'
)
EXPECTED_ORDERED_DIGESTS = (
    '16c7792b188a4e4e90185482d6926da9012218b95ba111a152e1acb295375fde'
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_archive(path):
    if digest(path) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError('Downloaded evidence archive SHA-256 differs')
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP member integrity check failed')
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) != 3775:
            raise ValueError('Unexpected or duplicate archive members')
        manifest_bytes = archive.read('inputs/manifest.json')
        if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
            raise ValueError('Validation manifest changed')
        manifest = json.loads(manifest_bytes)
        ids = manifest['frame_ids']
        if (manifest.get('schema_version') != 'clocs_kitti_full_validation_inputs_v1'
                or manifest.get('split') != 'validation' or len(ids) != 3769
                or len(set(ids)) != 3769):
            raise ValueError('Expected the complete ordered validation split')
        expected_members = {
            'inputs/manifest.json',
            'audits/baseline_parity_failed.json',
            'audits/paired_diagnostic.json',
            'code/run_clocs_full_validation.py',
            'code/evaluate_clocs_full_validation.py',
            'code/evaluate_clocs_val_subset.py',
        } | {f'predictions/{fid}.json' for fid in ids}
        if set(names) != expected_members:
            raise ValueError('Archive member names differ from the 3769-frame receipt')

        parity_bytes = archive.read('audits/baseline_parity_failed.json')
        paired = json.loads(archive.read('audits/paired_diagnostic.json'))
        parity = json.loads(parity_bytes)
        if (parity.get('status') != 'BASELINE_PARITY_PENDING'
                or paired.get('status') !=
                'PAIRED_RESULT_WITH_HISTORICAL_BASELINE_MISMATCH'
                or paired.get('failed_parity_audit_sha256') !=
                hashlib.sha256(parity_bytes).hexdigest()
                or paired.get('input_manifest_sha256') != EXPECTED_MANIFEST_SHA256
                or paired.get('historical_baseline_max_absolute_ap40_error', 0) < 8
                or paired.get('detections') != {'pointpillars': 24180,
                                                 'fusion': 19036}):
            raise ValueError('Paired and parity audit provenance differs')
        ordered = hashlib.sha256()
        baseline_count = fusion_count = 0
        for fid in ids:
            raw = archive.read(f'predictions/{fid}.json')
            ordered.update(hashlib.sha256(raw).digest())
            frame = json.loads(raw)
            if (frame.get('frame_id') != fid
                    or frame.get('provenance', {}).get('input_manifest_sha256')
                    != EXPECTED_MANIFEST_SHA256):
                raise ValueError(f'Prediction identity/provenance differs: {fid}')
            baseline_count += len(frame['detections']['pointpillars']['score'])
            fusion_count += len(frame['detections']['fusion']['score'])
        if (ordered.hexdigest() != EXPECTED_ORDERED_DIGESTS
                or paired.get('ordered_prediction_digests') != EXPECTED_ORDERED_DIGESTS
                or [baseline_count, fusion_count] != [24180, 19036]):
            raise ValueError('Prediction content differs from Kaggle audit')
        for metric in ('2D', 'BEV', '3D'):
            for difficulty in ('easy', 'moderate', 'hard'):
                before = parity['ap40_strict_iou_070']['pointpillars'][metric][difficulty]
                current = paired['ap40_strict_iou_070']['pointpillars_this_run'][metric][difficulty]
                after = paired['ap40_strict_iou_070']['fusion_this_run'][metric][difficulty]
                delta = paired['fusion_minus_pointpillars_this_run_ap_points'][metric][difficulty]
                if abs(before - current) > 1e-9 or abs(after - current - delta) > 1e-9:
                    raise ValueError(f'Inconsistent paired AP: {metric}/{difficulty}')
        return parity, paired, ordered.hexdigest()


def figure(paired, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    difficulties = ('easy', 'moderate', 'hard')
    metrics = ('2D', 'BEV', '3D')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, metric in zip(axes, metrics):
        baseline = [paired['ap40_strict_iou_070']['pointpillars_this_run']
                    [metric][level] for level in difficulties]
        fusion = [paired['ap40_strict_iou_070']['fusion_this_run']
                  [metric][level] for level in difficulties]
        x = np.arange(3)
        ax.bar(x - .18, baseline, .36, label='PointPillars (this run)')
        ax.bar(x + .18, fusion, .36, label='CLOCs (this run)')
        ax.set_xticks(x, ['Easy', 'Moderate', 'Hard'])
        ax.set_title(metric)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Car AP40, strict IoU 0.70 (%)')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('KITTI validation: 3,769 paired frames')
    fig.text(.5, .015,
             'Historical PointPillars baseline did not reproduce; compare only within this run.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(output / 'paired_ap40.png', dpi=180)
    fig.savefig(output / 'paired_ap40.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/clocs_full_val_paired'))
    args = parser.parse_args()
    parity, paired, ordered = verify_archive(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (('baseline_parity_failed.json', parity),
                       ('paired_diagnostic.json', paired)):
        (args.output_dir / name).write_text(json.dumps(data, indent=2) + '\n')
    figure(paired, args.output_dir)
    receipt = {
        'schema_version': 'clocs_full_val_evidence_archive_receipt_v1',
        'status': 'VERIFIED_PAIRED_DIAGNOSTIC_HISTORICAL_BASELINE_MISMATCH',
        'frames_verified': 3769,
        'archive_sha256': digest(args.archive),
        'ordered_prediction_digests': ordered,
        'interpretation': 'Paired full-validation measurement; historical PointPillars parity failed.',
    }
    (args.output_dir / 'archive_receipt.json').write_text(
        json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'frames': 3769,
                      '3D_moderate': {
                          'pointpillars_this_run': paired['ap40_strict_iou_070']
                          ['pointpillars_this_run']['3D']['moderate'],
                          'fusion_this_run': paired['ap40_strict_iou_070']
                          ['fusion_this_run']['3D']['moderate']},
                      'output_dir': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
