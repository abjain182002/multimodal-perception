"""Standard-library checks for pilot selection and private CPU input handoff."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).with_name('preflight_fog_colab.py')
spec = importlib.util.spec_from_file_location('fog_preflight', SCRIPT)
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def population(scenes):
    rows, tokens = [], []
    for i in range(scenes * 4):
        token = f'{i:032x}'; tokens.append(token)
        rows.append({'sample_token': token, 'scene_token': f'scene{i // 4}',
            'camera_files': [f'samples/{c}/{token}.jpg' for c in CAMERAS],
            'lidar_files': [f'samples/LIDAR_TOP/{token}_current.pcd.bin',
                            f'samples/LIDAR_TOP/{token}_historical.pcd.bin'] +
                           [f'sweeps/LIDAR_TOP/{token}_{j}.pcd.bin' for j in range(8)]})
    return ('\n'.join(tokens) + '\n').encode(), json.dumps(rows).encode(), rows


class Checks(unittest.TestCase):
    def test_population_and_pilot_keep_historical_keyframes(self):
        tokens, inventory, rows = population(2)
        ordered, mapping, required = f.validate_population(tokens, inventory, 8, 2)
        self.assertEqual(mapping[ordered[0]], rows[0])
        self.assertEqual(len(required), 128)
        self.assertEqual(len([p for p in mapping[ordered[0]]['lidar_files'] if p.startswith('samples/')]), 2)
        rows[0]['lidar_files'].pop()
        with self.assertRaises(AssertionError): f.validate_population(tokens, json.dumps(rows).encode(), 8, 2)

    def test_archive_paths_and_ambiguity(self):
        required = {'samples/LIDAR_TOP/a.pcd.bin', 'sweeps/LIDAR_TOP/b.pcd.bin'}
        result = f.sensor_index(['dataset/samples/LIDAR_TOP/a.pcd.bin',
                                 'root/dataset/sweeps/LIDAR_TOP/b.pcd.bin',
                                 '../samples/LIDAR_TOP/unsafe.pcd.bin'], required)
        self.assertEqual(set(result), required)
        with self.assertRaises(AssertionError):
            f.sensor_index(['dataset/samples/LIDAR_TOP/a.pcd.bin', 'samples/LIDAR_TOP/a.pcd.bin'], required)

    def test_bounded_project_catalogue(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            for directory in ('camera_study', 'unrelated_folder'):
                (base / directory).mkdir()
                (base / directory / 'inputs.zip').write_bytes(b'not opened by catalogue')
            (base / 'root.zip').write_bytes(b'not opened by catalogue')
            files, folders = f.candidates(base)
            self.assertEqual(len(files), 2)
            self.assertIn('unrelated_folder', folders)
            self.assertFalse(any('unrelated_folder' in str(p) for p in files))

    def test_end_to_end_600_frame_transfer(self):
        tokens, inventory, rows = population(150)
        def blob(data): return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        inputs = {f.STUDY + 'sample_tokens.txt': tokens,
                  f.STUDY + 'selected_input_manifest.json': inventory}
        # Dummy pinned assets are transferred as opaque bytes and never executed.
        assets = {'LICENSE': b'Test license', 'converter/lidar.py': b'raise RuntimeError("DO NOT EXECUTE")',
                  'converter/fog_lookup_tables/test.pickle': b'NOT A PICKLE; DO NOT DESERIALISE'}
        def download(url, timeout):
            for relative, data in {**inputs, **assets}.items():
                if url.endswith('/' + relative): return io.BytesIO(data)
            raise AssertionError(url)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); base = root / 'drive'; base.mkdir()
            archive = base / 'camera_challenges600_inputs.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for name in rows[0]['camera_files'] + rows[0]['lidar_files']:
                    z.writestr('dataset/' + name, b'\0' * 40 if name.endswith('.bin') else b'opaque camera bytes')
                z.writestr('annotations/metadata.pkl', b'NOT EXECUTED')
            with patch.object(f, 'PROJECT_FILES', {p: blob(b) for p, b in inputs.items()}), \
                 patch.object(f, 'UPSTREAM_FILES', {p: blob(b) for p, b in assets.items()}), \
                 patch.object(f, 'PLAN', {**f.PLAN, 'tokens_sha256': f.sha(tokens)}), \
                 patch.object(f.urllib.request, 'urlopen', download), \
                 patch.object(f.sys, 'argv', ['preflight', '--base', str(base), '--work', str(root / 'work')]), \
                 contextlib.redirect_stdout(io.StringIO()):
                f.main()
            outputs = list(base.glob('fog_20261029/preflight/*/*.zip'))
            self.assertEqual(len(outputs), 1)
            with zipfile.ZipFile(outputs[0]) as z:
                audit = json.loads(z.read('preflight_audit.json'))
                self.assertEqual(audit['required_sensor_files'], 9600)
                self.assertEqual(audit['pilot_files_recovered'], 16)
                self.assertEqual(audit['available_required_files_by_name'], 16)
                self.assertEqual(audit['status'], 'FOG_PREFLIGHT_PILOT_INPUTS_READY')
                self.assertFalse(json.loads(z.read('pilot_selection.json'))['predictions_opened'])
                manifest = json.loads(z.read('manifest.json'))['files']
                for name, record in manifest.items():
                    self.assertEqual(f.sha(z.read(name)), record['sha256'])


if __name__ == '__main__':
    unittest.main()
