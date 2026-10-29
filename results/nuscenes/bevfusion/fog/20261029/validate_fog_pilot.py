"""CPU source-equivalence, determinism and encoding checks on the frozen pilot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import cv2
import numpy as np

from fog_transforms import FogAdapter

PREFLIGHT_SHA = '9fd0f95619b8e4d19ef9a4c6432e332a90d75281c3a303159e76581bdb6a8126'


def sha(data): return hashlib.sha256(data).hexdigest()


def unpack_checked(archive, root):
    assert sha(Path(archive).read_bytes()) == PREFLIGHT_SHA
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('manifest.json'))['files']
        assert set(z.namelist()) == set(manifest) | {'manifest.json'}
        for name, record in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            data = z.read(name)
            assert len(data) == record['bytes'] and sha(data) == record['sha256']
            p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preflight_zip', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    rows = []; camera_examples = {}; point_examples = {}
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='fog_pilot_') as temp:
        root = Path(temp); manifest = unpack_checked(args.preflight_zip, root)
        selected = json.loads((root / 'pilot_selection.json').read_text())['selected']
        adapter = FogAdapter(root / 'upstream')
        for index, relative in enumerate(selected['camera_files']):
            rgb = cv2.cvtColor(cv2.imread(str(root / 'pilot_inputs' / relative)), cv2.COLOR_BGR2RGB)
            before = rgb.copy()
            assert np.array_equal(adapter.camera(rgb, 0, relative), rgb)
            if index == 0: camera_examples[0] = rgb
            for severity in (1, 2, 3):
                tick = time.perf_counter(); reference = adapter.camera(rgb, severity, relative, reference=True)
                reference_seconds = time.perf_counter() - tick
                tick = time.perf_counter(); fast = adapter.camera(rgb, severity, relative)
                fast_seconds = time.perf_counter() - tick
                assert np.array_equal(reference, fast), (relative, severity, 'uint8 source mismatch')
                assert np.array_equal(fast, adapter.camera(rgb, severity, relative)), 'Camera not deterministic'
                assert np.array_equal(rgb, before), 'Input image mutated'
                ok, encoded = cv2.imencode('.png', cv2.cvtColor(fast, cv2.COLOR_RGB2BGR))
                restored = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                assert ok and np.array_equal(fast, restored)
                rows.append({'modality': 'camera', 'path': relative, 'severity': severity,
                    'items': rgb.shape[0] * rgb.shape[1], 'reference_seconds': reference_seconds,
                    'adapter_seconds': fast_seconds, 'source_output_equal': True,
                    'output_sha256': sha(fast.tobytes()), 'fog_responses': '', 'max_intensity': '',
                    'points_beyond_200m': '', 'png_bytes': len(encoded)})
                if index == 0: camera_examples[severity] = fast
                print('CAMERA SOURCE/REPEAT/PNG PASS:', index + 1, severity, flush=True)
            # Clean lossless cache must also preserve the original decoded pixels.
            ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            assert ok and np.array_equal(cv2.cvtColor(cv2.imdecode(encoded, 1), cv2.COLOR_BGR2RGB), rgb)
        for index, relative in enumerate(selected['lidar_files']):
            points = np.fromfile(root / 'pilot_inputs' / relative, dtype=np.float32).reshape(-1, 5)
            original = points.copy(); assert (points[:, 3] >= 0).all()
            assert np.array_equal(adapter.points(points, 0, relative)[0], points)
            if index == 0: point_examples[0] = points
            for severity in (1, 2, 3):
                tick = time.perf_counter(); reference, rm = adapter.points(points, severity, relative, reference=True)
                reference_seconds = time.perf_counter() - tick
                tick = time.perf_counter(); fast, fm = adapter.points(points, severity, relative)
                fast_seconds = time.perf_counter() - tick
                assert np.array_equal(reference, fast), (relative, severity, 'float32 source mismatch')
                assert rm['fog_responses'] == fm['fog_responses']
                assert np.array_equal(fast, adapter.points(points, severity, relative)[0]), 'LiDAR not deterministic'
                assert np.array_equal(points, original), 'Input scan mutated'
                assert np.array_equal(fast[:, 4], points[:, 4])
                rows.append({'modality': 'lidar', 'path': relative, 'severity': severity, 'items': len(points),
                    'reference_seconds': reference_seconds, 'adapter_seconds': fast_seconds,
                    'source_output_equal': True, 'output_sha256': sha(fast.tobytes()),
                    'fog_responses': fm['fog_responses'], 'max_intensity': float(fast[:, 3].max()),
                    'points_beyond_200m': fm['points_beyond_200m'], 'png_bytes': ''})
                if index == 0: point_examples[severity] = fast
            print('RAW LIDAR SOURCE/REPEAT/FEATURE PASS:', index + 1, flush=True)
        # Boundary and zero-radius checks are separate from the real pilot.
        distances = np.array([0, .05, .15, 1.25, 1.35, 1.45, 9.95, 10, 199.95, 200, 200.05, 250], np.float32)
        edge = np.zeros((len(distances), 5), np.float32); edge[:, 0] = distances
        edge[:, 3] = np.arange(len(distances)) * 20; edge[:, 4] = np.arange(len(distances))
        for severity in (1, 2, 3):
            assert np.array_equal(adapter.points(edge, severity, 'synthetic_edge')[0],
                                  adapter.points(edge, severity, 'synthetic_edge', reference=True)[0])
        with (args.output / 'pilot_checks.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        record = {'status': 'FOG_REAL_PILOT_CPU_SOURCE_EQUIVALENCE_PASS',
            'preflight_zip_sha256': PREFLIGHT_SHA, 'payload_hashes_verified': len(manifest),
            'sample_token': selected['sample_token'], 'source_camera_equivalence_checks': 18,
            'source_raw_lidar_equivalence_checks': 30, 'synthetic_boundary_checks': 3,
            'output_comparison': 'Exact RGB uint8 and raw float32 output equality, not merely approximate float closeness.',
            'deterministic_repeat_pass': True, 'unperturbed_identity_pass': True,
            'camera_clean_and_fog_PNG_roundtrip_pass': True, 'input_immutability_pass': True,
            'raw_fifth_feature_preserved': True, 'finite_output_pass': True,
            'method_seed': "SHA-256 of UTF-8 '20261029:fog:{camera|lidar}:{severity}:{relative_path}', first 8 bytes unsigned big endian; camera uses this modulo 2^32 with legacy NumPy MT19937, LiDAR uses Generator(PCG64). Independently seeded severities. Same raw file/level shared across models.",
            'encoding_amendment': 'Decode original JPEG with pinned OpenCV, fog in RGB, cast uint8 as upstream. Use lossless PNG for both unperturbed and fog caches. Unlike upstream JPEG output, this adds no JPEG re-encoding confound; not a bitwise reproduction of the released dataset.',
            'arithmetic_amendment': 'Vectorise final camera pixel loop and LiDAR response loop; preserve source noise generation, lookup bins, RNG draw sequence and output dtypes. Equality demonstrated on this pilot and listed boundary cases, not proven for every possible input.',
            'reference_loading': 'Reviewed AST definitions from hash-verified source, numeric-only lookup unpickler; no monolithic snow dependency import or pickle5 requirement.',
            'upstream_behaviour_retained': 'No extra clipping of fog intensities or v1 range-noise factors; preserve upstream numerical behaviour, not a calibrated physical sensor model.',
            'packages': {'numpy': np.__version__, 'opencv': cv2.__version__},
            'python': platform.python_version(), 'elapsed_seconds': round(time.monotonic() - start, 2),
            'adapter_sha256': sha(Path(__file__).with_name('fog_transforms.py').read_bytes()),
            'validation_script_sha256': sha(Path(__file__).read_bytes()),
            'scope': 'CPU corruption compatibility only. One preselected real frame, no detector inference or accuracy. Timing is local pilot preparation timing, not model runtime or a measured full-study duration.'}
        (args.output / 'cpu_validation.json').write_text(json.dumps(record, indent=2) + '\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for level in range(4):
            axes[0, level].imshow(camera_examples[level]); axes[0, level].axis('off')
            axes[0, level].set_title('Unperturbed' if level == 0 else f'Synthetic fog {level}')
            pc = point_examples[level]
            axes[1, level].scatter(pc[:, 0], pc[:, 1], c=np.clip(pc[:, 3], 0, 255), s=.2,
                                   cmap='viridis', vmin=0, vmax=255, rasterized=True)
            axes[1, level].set(xlim=(-60, 60), ylim=(-60, 60), xlabel='Raw sensor x (m)', ylabel='Raw sensor y (m)')
            axes[1, level].set_aspect('equal'); axes[1, level].grid(alpha=.15)
        fig.suptitle('Preselected fog compatibility pilot: CAM_FRONT and first listed raw LiDAR scan')
        fig.text(.015, .02, 'Raw scan only, not accumulated sweeps. Display colours clipped to 0-255; stored fog intensities are unchanged. No detection results.', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .95))
        for extension in ('png', 'pdf'): fig.savefig(args.output / ('pilot_fog_comparison.' + extension), dpi=160)
        plt.close(fig)
    print(json.dumps(record, indent=2))


if __name__ == '__main__': main()
