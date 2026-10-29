"""Verify the preserved repeat diagnostic and recompute its prediction checks."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.optimize import linear_sum_assignment

EXPECTED = '4ef30b9a808cabd729edc9ada7116a800ac52b7832a94def2d7ce6d4f51565b2'


def verify(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('manifest.json'))['files']
        assert set(z.namelist()) == set(manifest) | {'manifest.json'}
        for name, record in manifest.items():
            raw = z.read(name)
            assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
        read = lambda name: json.loads(z.read(name))
        assert read('subprocess_results.json')['return_codes'] == {'default': 0, 'deterministic': 0}
        modes = {}; packed_fingerprints = []
        for mode in ('default', 'deterministic'):
            inputs = read(mode + '/input_reload_fingerprints.json')
            packed_fingerprints.append(inputs['reloads'][0])
            assert inputs['reloads_identical'] and inputs['reloads'][0] == inputs['reloads'][1] == inputs['reloads'][2]
            arrays = [{k: np.asarray(v) for k, v in read(f'{mode}/prediction_{i}.json').items()} for i in range(3)]
            for p in arrays:
                assert p['boxes'].shape == (200, 9) and p['scores'].shape == p['labels'].shape == (200,)
                assert np.isfinite(p['boxes']).all() and np.isfinite(p['scores']).all()
                assert ((p['scores'] >= 0) & (p['scores'] <= 1)).all()
                assert ((p['labels'] >= 0) & (p['labels'] < 10)).all()
            comparisons = []
            for i in (1, 2):
                a, b = arrays[0], arrays[i]
                valid = ((a['labels'][:, None] == b['labels'][None, :]) &
                    np.isclose(a['scores'][:, None], b['scores'][None, :], rtol=1e-5, atol=1e-5) &
                    np.isclose(a['boxes'][:, None, :], b['boxes'][None, :, :], rtol=1e-5, atol=1e-4).all(axis=2))
                row, col = linear_sum_assignment(~valid)
                result = {'label_position_mismatches': int(np.count_nonzero(a['labels'] != b['labels'])),
                    'bitwise_values_equal': all(np.array_equal(a[k], b[k]) for k in a),
                    'original_positional_check_pass': bool(np.array_equal(a['labels'], b['labels']) and np.allclose(a['scores'], b['scores'], rtol=1e-5, atol=1e-5) and np.allclose(a['boxes'], b['boxes'], rtol=1e-5, atol=1e-4)),
                    'unmatched_under_original_tolerances': int(200 - valid[row, col].sum())}
                saved = read(mode + '/diagnostic.json')['comparisons_to_first'][i-1]
                assert result['original_positional_check_pass'] == saved['original_positional_check_pass']
                assert result['unmatched_under_original_tolerances'] == saved['unmatched_under_original_tolerances']
                assert result['label_position_mismatches'] == saved['positional_label_mismatches']
                comparisons.append(result)
            forwards = [read(f'{mode}/forward_{i}.json') for i in range(3)]
            assert all(f['same_packed_input_before_forward'] and f['original_cpu_input_unchanged'] for f in forwards)
            for name in forwards[0]['intermediates']:
                first = forwards[0]['intermediates'][name]['sha256']
                for f in forwards:
                    assert f['intermediates'][name]['exact_to_first'] == (f['intermediates'][name]['sha256'] == first)
            modes[mode] = {'recomputed_prediction_comparisons': comparisons,
                'dataset_reloads_identical': True, 'packed_inputs_identical': True,
                'recorded_intermediate_comparisons': [f['intermediates'] for f in forwards],
                'intermediate_arrays_exported': False}
        assert packed_fingerprints[0] == packed_fingerprints[1]
        assert all(c['bitwise_values_equal'] for c in modes['deterministic']['recomputed_prediction_comparisons'])
        assert all(item['exact_to_first'] for run in modes['deterministic']['recorded_intermediate_comparisons'] for item in run.values())
        assert all(not c['original_positional_check_pass'] for c in modes['default']['recomputed_prediction_comparisons'])
        return {'status': 'FUSION_REPEAT_DIAGNOSTIC_LOCAL_REVIEW_PASS', 'archive_sha256': EXPECTED,
            'archive_bytes': path.stat().st_size, 'verified_payload_members': len(manifest), 'modes': modes,
            'interpretation': 'Variation was observed at projected depth with identical inputs under default execution; strict PyTorch determinism removed all observed intermediate/prediction variation in three baseline forwards on this T4 pilot. Ordering alone does not explain the default outputs.',
            'limitations': 'Intermediate comparisons are recorded GPU summaries, not recomputed depth arrays. Exact CUDA-kernel attribution and cross-hardware reproducibility are not established. No fog-level or full-study accuracy conclusion.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('archive', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); result = verify(args.archive)
    with args.output.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(result['status'])
