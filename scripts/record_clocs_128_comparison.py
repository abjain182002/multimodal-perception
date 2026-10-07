"""Verify two CLOCs training runs and publish their internal development comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED = {
    0: ('a2203b9adee46b782e896b7f3b9344b5916433b5404d4bd871aa7e05d6b0c473', 12),
    8: ('f8b8b5e32614553f38e6b7bff6090ac99de7ff29073b4448d3d358f42287881a', 2),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_run(root, ratio):
    folder = root / f'clocs_128_ratio{ratio}_seed20260928'
    report = json.loads((folder / 'report.json').read_text())
    split = json.loads((folder / 'split.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    input_audit = json.loads((folder / 'input_audit.json').read_text())
    before = json.loads((folder / 'before_dev.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    with (folder / 'learning_curve.csv').open(newline='') as handle:
        curve = list(csv.DictReader(handle))
    checkpoint_hash = sha256(folder / 'best_dev_checkpoint.pt')
    expected_hash, expected_epoch = EXPECTED[ratio]
    if (report['status'] != 'PASS' or report['best_epoch'] != expected_epoch
            or checkpoint_hash != expected_hash or report['checkpoint_sha256'] != expected_hash
            or report['reload_ap_absolute_error'] > 1e-7):
        raise ValueError(f'Checkpoint or report mismatch for ratio {ratio}')
    if (report['split'] != split or report['protocol'] != protocol or report['before_dev'] != before
            or input_audit['status'] != 'PASS'):
        raise ValueError(f'Run metadata mismatch for ratio {ratio}')
    if (len(split['train_frame_ids']) != 96 or len(split['dev_frame_ids']) != 32
            or set(split['train_frame_ids']) & set(split['dev_frame_ids'])
            or len(history) != 12 or len(curve) != 12):
        raise ValueError(f'Split or training history mismatch for ratio {ratio}')
    if (protocol['settings']['negative_ratio'] != ratio
            or protocol['settings']['seed'] != 20260928):
        raise ValueError(f'Protocol mismatch for ratio {ratio}')
    for index, (row, record) in enumerate(zip(curve, history), start=1):
        if (int(row['epoch']) != index or record['epoch'] != index
                or abs(float(row['fusion_candidate_ap']) - record['dev']['fusion_candidate_ap']) > 1e-8
                or abs(float(row['pointpillars_candidate_ap']) - record['dev']['pointpillars_candidate_ap']) > 1e-8):
            raise ValueError(f'History/curve mismatch for ratio {ratio}, epoch {index}')
    best = max(history, key=lambda item: item['dev']['fusion_candidate_ap'])
    if (best['epoch'] != report['best_epoch']
            or abs(best['dev']['fusion_candidate_ap'] - report['best_dev']['fusion_candidate_ap']) > 1e-8):
        raise ValueError(f'Best checkpoint selection mismatch for ratio {ratio}')
    return folder, report, split, protocol, input_audit, curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/kitti/clocs/training_comparison'))
    args = parser.parse_args()
    runs = {ratio: read_run(args.runs, ratio) for ratio in EXPECTED}
    base, sampled = runs[0], runs[8]
    for index in (2, 4):
        if base[index] != sampled[index]:
            raise ValueError('Runs have different splits or audited training inputs')
    for report in (base[1], sampled[1]):
        dev = report['best_dev']
        if (dev['frames'] != 32 or dev['evaluated_candidates'] != 3425683
                or dev['positive_candidates'] != 2433
                or abs(dev['pointpillars_candidate_ap'] - 0.5391782875407971) > 1e-8):
            raise ValueError('Development evaluation differs from recorded experiment')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    for ratio, label, color in ((0, 'All valid negatives', '#145a80'),
                                (8, '8:1 negative sampling', '#e07830')):
        curve = runs[ratio][5]
        ax.plot([int(row['epoch']) for row in curve],
                [100 * float(row['fusion_candidate_ap']) for row in curve],
                marker='o', markersize=3.5, linewidth=2, label=label, color=color)
    baseline = base[1]['best_dev']['pointpillars_candidate_ap']
    ax.axhline(100 * baseline, linestyle='--', linewidth=1.5, color='#555555',
               label='PointPillars candidate ranking')
    ax.set(xlabel='Training epoch', ylabel='Internal development candidate AP (%)',
           title='CLOCs training: negative sampling comparison', xticks=list(range(1, 13)))
    ax.grid(alpha=0.23)
    ax.legend(loc='lower right')
    fig.savefig(args.output / 'learning_curves.png', dpi=200)
    fig.savefig(args.output / 'learning_curves.pdf')
    plt.close(fig)

    rows = []
    for ratio in EXPECTED:
        folder, report, _, _, _, _ = runs[ratio]
        dest = args.output / f'ratio{ratio}'
        dest.mkdir(exist_ok=True)
        for name in ('report.json', 'split.json', 'protocol.json', 'input_audit.json',
                     'before_dev.json', 'history.json', 'learning_curve.csv'):
            shutil.copy2(folder / name, dest / name)
        rows.append(dict(method='All valid negatives' if ratio == 0 else '8:1 sampled negatives',
                         negative_ratio=ratio, best_epoch=report['best_epoch'],
                         candidate_ap=report['best_dev']['fusion_candidate_ap'],
                         checkpoint_sha256=report['checkpoint_sha256']))
    rows.insert(0, dict(method='PointPillars', negative_ratio=None, best_epoch=None,
                        candidate_ap=baseline, checkpoint_sha256=None))
    with (args.output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(schema_version='clocs_128_candidate_comparison_v1', status='PASS',
                   interpretation='Internal development candidate ranking; not KITTI 3D or BEV AP40.',
                   optimization_frames=96, development_frames=32,
                   development_candidates=3425683, development_positives=2433,
                   pointpillars_candidate_ap=baseline,
                   all_negatives_candidate_ap=base[1]['best_dev']['fusion_candidate_ap'],
                   ratio8_candidate_ap=sampled[1]['best_dev']['fusion_candidate_ap'],
                   all_minus_ratio8_percentage_points=100 * (
                       base[1]['best_dev']['fusion_candidate_ap']
                       - sampled[1]['best_dev']['fusion_candidate_ap']),
                   caveats=['One frame split and one seed per training protocol.',
                            'The initial untrained network was also evaluated on development frames.',
                            'Checkpoint selection used the same development set; full KITTI validation is pending.'],
                   checkpoint_sha256={str(r): runs[r][1]['checkpoint_sha256'] for r in EXPECTED})
    (args.output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()

