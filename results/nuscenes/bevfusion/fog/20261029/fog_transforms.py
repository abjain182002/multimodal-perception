"""Deterministic raw-input fog adapter, checked against pinned MultiCorrupt.

Uses reviewed function definitions from hash-verified upstream source. Does not
import the monolithic snow dependencies or execute unrestricted pickle globals.
Camera output is RGB uint8 before lossless PNG encoding; LiDAR is raw Nx5 f32.
"""
import ast
from contextlib import contextmanager
import copy
import hashlib
import io
import math
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import numpy as np

from preflight_fog_colab import UPSTREAM_FILES


class NumericTableUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
                   ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise ValueError('Unexpected lookup-table global: ' + module + '.' + name)
        return allowed[module, name]


def numeric_load(stream):
    value = NumericTableUnpickler(stream).load()
    assert isinstance(value, dict) and len(value) == 2001
    for i in range(2001):
        row = value[i / 10]
        assert isinstance(row, tuple) and len(row) == 2
        assert all(isinstance(x, (int, float, np.floating)) and math.isfinite(x) for x in row)
    return value


def seed(kind, relative_path, severity):
    text = f'20261029:fog:{kind}:{severity}:{relative_path}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


@contextmanager
def camera_rng(value):
    # Preserve caller state. Use process workers, not threads, for preparation.
    old = np.random.get_state()
    np.random.seed(value % (2 ** 32))
    try:
        yield
    finally:
        np.random.set_state(old)


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class FogAdapter:
    def __init__(self, upstream):
        assert np.__version__ == '1.26.4', 'Use the pinned CPU/GPU preparation environment'
        upstream = Path(upstream)
        for name, expected in UPSTREAM_FILES.items():
            data = (upstream / name).read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert actual == expected, 'Upstream file mismatch: ' + name
        from types import SimpleNamespace
        self.image = definitions(upstream / 'converter/img.py',
            ['plasma_fractal', 'next_power_of_2', 'noise', 'fog'], {'np': np, 'math': math})
        self.lidar = definitions(upstream / 'converter/lidar.py',
            ['ParameterSet', 'get_integral_dict', 'P_R_fog_hard', 'P_R_fog_soft', 'simulate_fog'],
            {'np': np, 'math': math, 'copy': copy, 'Path': Path, 'Dict': Dict, 'List': List,
             'Tuple': Tuple, 'c': 299792458., 'pickle': SimpleNamespace(load=numeric_load),
             'INTEGRAL_PATH': upstream / 'converter/fog_lookup_tables'})
        self.tables = {}
        for severity, (alpha, beta) in enumerate(((.02, .008), (.03, .008), (.06, .05)), 1):
            p = self.lidar['ParameterSet'](alpha=alpha, beta=beta)
            table = self.lidar['get_integral_dict'](p)
            self.tables[severity] = np.array([table[i / 10] for i in range(2001)], dtype=np.float64)

    def camera(self, rgb, severity, relative_path, reference=False):
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        assert severity in (0, 1, 2, 3)
        if severity == 0:
            return rgb.copy()
        with camera_rng(seed('camera', relative_path, severity)):
            if reference:
                result = self.image['fog'](rgb, severity)
            else:
                image = self.image['noise'](rgb, severity) / 255.
                height, width, _ = image.shape
                y1, x1 = int(height * .65), int(width * .1)
                y2, x2 = int(height * .99), int(width * .85)
                dy = np.maximum(0, np.maximum(np.arange(height) - y2, y1 - np.arange(height)))
                dx = np.maximum(0, np.maximum(np.arange(width) - x2, x1 - np.arange(width)))
                rect = np.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
                distance = 2. * rect * (1 + np.random.rand(height, width) / 4) + math.sqrt(max(height, width))
                transmission = np.exp(-(.0005, .001, .002)[severity - 1] * distance)
                result = (image * transmission[..., None] + .5 * (1 - transmission[..., None])) * 255
        assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= 255 + 1e-10
        return result.astype(np.uint8)

    def points(self, points, severity, relative_path, reference=False):
        assert points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 5 and len(points)
        assert np.isfinite(points).all() and severity in (0, 1, 2, 3)
        if severity == 0:
            return points.copy(), {'fog_responses': 0}
        generator = np.random.default_rng(seed('lidar', relative_path, severity))
        if reference:
            self.lidar['RNG'] = generator
            result, _, count, _ = self.lidar['simulate_fog'](severity, points, 10)
        else:
            alpha, beta = ((.02, .008), (.03, .008), (.06, .05))[severity - 1]
            ranges = np.linalg.norm(points[:, :3], axis=1)
            attenuated = np.round(np.exp(-2 * alpha * ranges) * points[:, 3])
            # The reference rounds float32 radius to one decimal and uses a
            # decimal-valued dict key capped at 200 m. Index the same bins.
            bins = np.rint(np.minimum(np.round(ranges, 1).astype(np.float64), 200.) * 10).astype(np.int64)
            distance, response = self.tables[severity][bins].T
            response = response * points[:, 3].astype(np.float64) * ranges.astype(np.float64) ** 2 * beta / (1e-6 / np.pi)
            fog = response > attenuated
            result = points.astype(np.float64)
            result[:, 3] = attenuated
            generator.integers(low=1, high=20, size=1)  # Consumed even though upstream overwrites r_noise.
            count = int(fog.sum())
            if count:
                radius = ranges[fog].astype(np.float64)
                scale = distance[fog] / radius
                result[fog, :3] = points[fog, :3].astype(np.float64) * scale[:, None]
                result[fog, 3] = response[fog]
                noisy_range = generator.uniform(low=radius - 10, high=radius + 10, size=count)
                result[fog, :3] *= (radius / noisy_range)[:, None]
        result = result.astype(np.float32)
        assert np.isfinite(result).all()
        assert result.shape == points.shape and np.array_equal(result[:, 4], points[:, 4])
        return result, {'fog_responses': int(count),
            'negative_intensities': int((result[:, 3] < 0).sum()),
            'points_beyond_200m': int((np.linalg.norm(result[:, :3], axis=1) > 200).sum())}
