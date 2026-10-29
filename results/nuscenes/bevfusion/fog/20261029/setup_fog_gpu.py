"""Restore the project's previously exercised CUDA 12.1 / PyTorch 2.1.2 stack.
Adapted to an isolated Colab folder; installs toolkit only, not GPU drivers.
Actual fog-session GPU build and smoke must still be run.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

WORK = Path("/content/fog_runtime")
OUT = Path("/content/fog_runtime/setup")
STUDY = Path("/content/fog_runtime/study")

for folder in (WORK, OUT, STUDY):
    folder.mkdir(parents=True, exist_ok=True)

PYTHON = WORK / "bevfusion310/bin/python"
REPO = WORK / "mmdetection3d"
CUDA = WORK / "cuda-12.1"
COMMIT = "fe25f7a51d36e3702f961e198894580d83c4387b"

env = os.environ.copy()
env.update({
    "CUDA_VISIBLE_DEVICES": "0",
    "UV_CACHE_DIR": str(WORK / "uv_cache"),
    "UV_PYTHON_INSTALL_DIR": str(WORK / "managed_python"),
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONUNBUFFERED": "1",
    "MPLBACKEND": "Agg",
    "CC": "/usr/bin/gcc-11",
    "CXX": "/usr/bin/g++-11",
    "MAX_JOBS": "2",
})

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

log = (OUT / "all_in_one_setup.log").open("a", buffering=1)

def run(command, cwd=None):
    command = [str(value) for value in command]
    heading = "\nRUNNING: " + " ".join(command)
    print(heading, flush=True)
    log.write(heading + "\n")
    process = subprocess.Popen(
        command, cwd=cwd, env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
    code = process.wait()
    if code:
        raise RuntimeError(
            f"Command failed with exit code {code}. "
            f"Log: {OUT / 'all_in_one_setup.log'}"
        )

def stage(number, title):
    message = f"\nSTAGE {number}/5: {title}"
    print(message, flush=True)
    log.write(message + "\n")

def download(url, destination, expected_hash, expected_bytes=None):
    if not destination.exists():
        temporary = destination.with_name(destination.name + ".downloading")
        run([
            "curl", "--fail", "--location", "--retry", "3",
            "--output", temporary, url,
        ])
        assert sha256(temporary) == expected_hash, (
            "Download SHA-256 mismatch", destination.name
        )
        if expected_bytes is not None:
            assert temporary.stat().st_size == expected_bytes
        temporary.replace(destination)

    assert sha256(destination) == expected_hash, destination.name
    if expected_bytes is not None:
        assert destination.stat().st_size == expected_bytes

def run_script(name, text):
    path = OUT / name
    path.write_text(text)
    run([PYTHON, "-u", path], cwd=REPO)
    return path

try:
    stage(1, "Restore isolated Python 3.10 and pinned PyTorch")

    assert shutil.which("curl"), "curl is missing."
    assert shutil.which("git"), "git is missing."
    if not Path("/usr/bin/gcc-11").is_file() or not Path("/usr/bin/g++-11").is_file():
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "gcc-11", "g++-11"])
    assert Path("/usr/bin/gcc-11").is_file(), "GCC 11 is missing."
    assert Path("/usr/bin/g++-11").is_file(), "G++ 11 is missing."

    compiler_bin = WORK / "compiler_bin"
    compiler_bin.mkdir(exist_ok=True)
    for alias, target in {"gcc": "/usr/bin/gcc-11", "g++": "/usr/bin/g++-11", "cc": "/usr/bin/gcc-11", "c++": "/usr/bin/g++-11"}.items():
        link = compiler_bin / alias
        if not link.exists():
            link.symlink_to(target)
    env["PATH"] = str(compiler_bin) + ":" + env["PATH"]
    run([compiler_bin / "gcc", "--version"])

    uv = WORK / "uv_package/bin/uv"
    if not uv.is_file():
        run([
            sys.executable, "-m", "pip", "install",
            "--quiet", "--no-cache-dir",
            "--target", WORK / "uv_package",
            "uv==0.13.4",
        ])

    run([uv, "python", "install", "3.10.22"])
    if not PYTHON.is_file():
        run([
            uv, "venv", "--python", "3.10.22", "--seed",
            WORK / "bevfusion310",
        ])

    run([
        PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "setuptools==69.5.1", "wheel", "numpy==1.26.4",
    ])
    run([
        PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "torch==2.1.2", "torchvision==0.16.2",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
    run([PYTHON, "-u", "-c", """
import sys, numpy, torch, torchvision
from pkg_resources import packaging
from torch.utils.cpp_extension import BuildExtension
assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == "1.26.4"
assert torch.__version__ == "2.1.2+cu121"
assert torchvision.__version__ == "0.16.2+cu121"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
x = torch.tensor([1., 2.], device="cuda")
assert (x * 2).cpu().tolist() == [2., 4.]
torch.cuda.synchronize()
print("PINNED_TORCH_GPU_PASS", flush=True)
"""])

    stage(2, "Restore pinned repository and dependencies")

    if not (REPO / ".git").is_dir():
        assert not REPO.exists(), f"Unexpected directory: {REPO}"
        run([
            "git", "clone",
            "https://github.com/open-mmlab/mmdetection3d.git",
            REPO,
        ])
    run(["git", "checkout", "--detach", COMMIT], cwd=REPO)

    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip()
    assert actual_commit == COMMIT

    pins = [
        "setuptools==69.5.1", "numpy==1.26.4",
        "torch==2.1.2", "torchvision==0.16.2",
        "mmengine==0.10.7", "mmcv==2.1.0", "mmdet==3.2.0",
        "nuscenes-devkit==1.2.0", "scipy==1.15.3",
        "matplotlib==3.8.4", "scikit-learn==1.7.2",
        "shapely==2.0.7", "numba==0.61.2",
        "spconv-cu120==2.3.6", "cumm-cu120==0.4.11",
        "ninja==1.13.2", "pillow==12.3.0", "opencv-python==4.11.0.86",
        "opencv-python-headless==4.11.0.86",
    ]
    constraints = OUT / "constraints.txt"
    constraints.write_text("\n".join(pins) + "\n")

    run([
        PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "-c", constraints, "mmengine==0.10.7",
    ])
    run([
        PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "-c", constraints,
        "https://download.openmmlab.com/mmcv/dist/cu121/"
        "torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl",
    ])
    run([
        PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "-c", constraints,
        *pins[6:],
    ])
    run([
        PYTHON, "-m", "pip", "install", "--no-cache-dir",
        "--no-build-isolation", "-c", constraints, "-e", REPO,
    ])

    stage(3, "Install CUDA 12.1 compiler toolkit")

    if not (CUDA / "bin/nvcc").is_file():
        installer = WORK / "cuda_12.1.0_530.30.02_linux.run"
        download(
            "https://developer.download.nvidia.com/compute/cuda/"
            "12.1.0/local_installers/"
            "cuda_12.1.0_530.30.02_linux.run",
            installer,
            "68699036c12d71adb9ad2799dce2ff070"
            "270fab4488b90920b9756ab3f52c41c",
        )
        staging = WORK / "cuda_installer_tmp"
        staging.mkdir(exist_ok=True)
        print(
            "Installing toolkit; this may be quiet for several minutes.",
            flush=True,
        )
        run([
            "sh", installer, "--silent", "--toolkit",
            f"--toolkitpath={CUDA}", f"--tmpdir={staging}",
        ])
        assert (CUDA / "bin/nvcc").is_file()
        assert (CUDA / "include/cuda.h").is_file()
        installer.unlink()

    env["CUDA_HOME"] = str(CUDA)
    env["PATH"] = str(CUDA / "bin") + ":" + env["PATH"]
    env["LD_LIBRARY_PATH"] = (
        str(CUDA / "lib64") + ":" +
        env.get("LD_LIBRARY_PATH", "")
    )
    env["PYTHONPATH"] = str(REPO) + ":" + env.get("PYTHONPATH", "")
    run([CUDA / "bin/nvcc", "--version"])

    stage(4, "Build BEVFusion custom operations")

    run([
        PYTHON, "projects/BEVFusion/setup.py",
        "build_ext", "--inplace",
    ], cwd=REPO)

    stage(5, "Verify dependencies and CUDA operations")

    run_script("all_in_one_operations_check.py", r'''
from pathlib import Path
import hashlib
import importlib
import importlib.metadata as metadata
import json
import torch
from torch.utils.cpp_extension import CUDA_HOME
from mmcv.ops import nms
from projects.BEVFusion.bevfusion.ops.bev_pool.bev_pool import bev_pool
from projects.BEVFusion.bevfusion.ops.voxel.voxelize import Voxelization

expected = {
    "torch": "2.1.2+cu121",
    "torchvision": "0.16.2+cu121",
    "numpy": "1.26.4",
    "setuptools": "69.5.1",
    "mmengine": "0.10.7",
    "mmcv": "2.1.0",
    "mmdet": "3.2.0",
    "nuscenes-devkit": "1.2.0",
    "spconv-cu120": "2.3.6",
    "cumm-cu120": "0.4.11",
}
for name, version in expected.items():
    assert metadata.version(name) == version, name

assert Path(CUDA_HOME).resolve() == Path(
    "/content/fog_runtime/cuda-12.1"
).resolve()
assert torch.cuda.device_count() == 1

boxes = torch.tensor(
    [[0., 0., 1., 1.], [0., 0., 1., 1.]], device="cuda"
)
scores = torch.tensor([.9, .8], device="cuda")
_, keep = nms(boxes, scores, .5)
assert keep.cpu().tolist() == [0]

features = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
coordinates = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
pooled = bev_pool(features, coordinates, 1, 1, 1, 1)
torch.testing.assert_close(
    pooled.flatten(), torch.tensor([4., 6.], device="cuda"),
    rtol=0, atol=0,
)

points = torch.tensor([
    [.1, .1, .1, 1.],
    [.2, .2, .2, 2.],
    [1.1, .1, .1, 3.],
], device="cuda")
layer = Voxelization(
    voxel_size=[1., 1., 1.],
    point_cloud_range=[0., 0., 0., 2., 2., 2.],
    max_num_points=5, max_voxels=10, deterministic=True,
).eval()
voxels, coords, counts = layer(points)
voxels, coords, counts = voxels.cpu(), coords.cpu(), counts.cpu()
wanted = {
    (0, 0, 0): points[:2].cpu(),
    (1, 0, 0): points[2:].cpu(),
}
assert len(coords) == 2
assert {tuple(row) for row in coords.tolist()} == set(wanted)
for index, row in enumerate(coords.tolist()):
    expected_points = wanted[tuple(row)]
    assert counts[index].item() == len(expected_points)
    torch.testing.assert_close(
        voxels[index, :len(expected_points)], expected_points,
        rtol=0, atol=0,
    )
torch.cuda.synchronize()

extensions = {}
for name in (
    "projects.BEVFusion.bevfusion.ops.bev_pool.bev_pool_ext",
    "projects.BEVFusion.bevfusion.ops.voxel.voxel_layer",
):
    path = Path(importlib.import_module(name).__file__)
    extensions[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

audit = {
    "status": "KAGGLE_BEVFUSION_CUDA_OPERATIONS_PASS",
    "source_commit": "fe25f7a51d36e3702f961e198894580d83c4387b",
    "versions": expected,
    "gpu": torch.cuda.get_device_name(0),
    "bev_pool_output": pooled.flatten().cpu().tolist(),
    "voxel_coordinates": coords.tolist(),
    "voxel_point_counts": counts.tolist(),
    "extensions": extensions,
    "scope": "Synthetic operations passed; model smoke pending.",
}
out = Path("/content/fog_runtime/setup")
(out / "cuda_operations_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)
(out / "environment.txt").write_text(
    "\n".join(sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in metadata.distributions()
        if d.metadata.get("Name")
    )) + "\n"
)
print(json.dumps(audit, indent=2), flush=True)
''')


finally:
    log.close()
