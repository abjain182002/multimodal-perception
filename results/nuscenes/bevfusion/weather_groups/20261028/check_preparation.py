"""Scientific membership and CPU handoff checks with synthetic metadata only."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

SCRIPT = Path(__file__).with_name('prepare_weather_groups.py')
spec = importlib.util.spec_from_file_location('weather_prepare', SCRIPT)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class Checks(unittest.TestCase):
    def test_description_boundaries_and_negation(self):
        for text, expected in [
            ('Night, rain, pedestrians', ('night_and_rain_mentioned', True, True, False)),
            ('Rainy intersection', ('rain_only_mentioned', True, False, False)),
            ('NIGHTTIME road', ('night_only_mentioned', False, True, False)),
            ('Train, nighttime-like lighting', ('night_only_mentioned', False, True, False)),
            ('Train road, overnight delivery', ('neither_mentioned', False, False, False)),
            ('No rain at night', ('night_and_rain_mentioned', True, True, True)),
            ('Without heavy rain', ('rain_only_mentioned', True, False, True)),
        ]:
            self.assertEqual(w.classify(text), expected)

    def test_complete_scene_population(self):
        scenes = [{'token': 'scene', 'name': 'scene-1', 'description': 'Rain',
                   'log_token': 'log', 'nbr_samples': 2}]
        samples = [{'token': 'a', 'scene_token': 'scene', 'timestamp': 2},
                   {'token': 'b', 'scene_token': 'scene', 'timestamp': 1}]
        rows, members = w.build_inventory(scenes, samples, {'a', 'b'}, 2, 1)
        self.assertEqual([r['sample_token'] for r in members], ['b', 'a'])
        self.assertEqual(rows[0]['samples'], 2)
        with self.assertRaises(AssertionError):
            w.build_inventory(scenes, samples, {'a'}, 1, 1)
        with self.assertRaises(AssertionError):
            w.build_inventory(scenes, samples + [samples[0]], {'a', 'b'}, 2, 1)

    def test_full_stage_handoff_without_prediction_access(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            base, root = d / 'drive', d / 'root'
            base.mkdir()
            scenes, samples, tokens = [], [], []
            for i in range(150):
                n = 41 if i < 19 else 40
                scenes.append({'token': f's{i}', 'name': f'scene-{i:04}',
                               'description': ['Road', 'Rain', 'Night', 'Night, rain'][i % 4],
                               'log_token': f'log{i // 2}', 'nbr_samples': n})
                for j in range(n):
                    token = f'{len(tokens):032x}'
                    tokens.append(token)
                    samples.append({'token': token, 'scene_token': f's{i}', 'timestamp': j})
            tables = {'scene.json': json.dumps(scenes).encode(),
                      'sample.json': json.dumps(samples).encode()}
            tgz = io.BytesIO()
            with tarfile.open(fileobj=tgz, mode='w:gz') as tar:
                for name, data in tables.items():
                    entry = tarfile.TarInfo('./v1.0-trainval/' + name)
                    entry.size = len(data)
                    tar.addfile(entry, io.BytesIO(data))
            with zipfile.ZipFile(base / 'nuscenes_validation_parts_01_02_verified.zip', 'w') as z:
                z.writestr('metadata/v1.0-trainval_meta.tgz', tgz.getvalue())
            # Deliberately invalid prediction bytes: the preparation must not parse them.
            for model in ('lidar', 'lidar_camera'):
                f = base / 'full_validation_merged' / model / 'results_nusc.json'
                f.parent.mkdir(parents=True)
                f.write_bytes(b'THIS IS NOT JSON; DO NOT READ')
            data = {f'batch_0{i+1}': ('\n'.join(ts) + '\n').encode()
                    for i, ts in enumerate((tokens[:2368], tokens[2368:4736], tokens[4736:]))}
            blobs = {k: hashlib.sha1(b'blob ' + str(len(v)).encode() + b'\0' + v).hexdigest()
                     for k, v in data.items()}
            def fetch(url, timeout):
                return io.BytesIO(next(v for k, v in data.items() if '/' + k + '/' in url))
            with patch.object(w, 'TOKEN_BLOBS', blobs), \
                 patch.object(w, 'METADATA_HASHES', {k: w.sha(v) for k, v in tables.items()}), \
                 patch.object(w.urllib.request, 'urlopen', fetch), \
                 patch.object(w.sys, 'argv', ['prepare', '--base', str(base), '--root', str(root),
                                             '--work', str(d / 'work')]):
                w.main()
            archives = list(base.glob('weather_groups_20261028/*/*.zip'))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as z:
                audit = json.loads(z.read('preparation_audit.json'))
                self.assertEqual(sum(r['samples'] for r in audit['groups'].values()), 6019)
                self.assertTrue(all(not r['opened'] for r in audit['predictions'].values()))
                manifest = json.loads(z.read('manifest.json'))['files']
                for name, record in manifest.items():
                    self.assertEqual(w.sha(z.read(name)), record['sha256'])
            # Restored metadata route returns the same bytes without requiring archive access.
            (root / 'v1.0-trainval').mkdir(parents=True)
            for name, payload in tables.items():
                (root / 'v1.0-trainval' / name).write_bytes(payload)
            actual, _ = w.metadata_bytes(root, d / 'nonexistent.zip')
            self.assertEqual(actual, tables)


if __name__ == '__main__':
    unittest.main()
