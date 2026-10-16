
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
    "/tmp/camera_challenges/cuda-12.1"
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
out = Path("/kaggle/working/camera_challenges_setup")
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
