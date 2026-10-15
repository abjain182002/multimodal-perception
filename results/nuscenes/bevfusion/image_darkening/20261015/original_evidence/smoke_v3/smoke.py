import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

base = Path("/content/drive/MyDrive/Major_Project/nuscenes_full_validation")
local = Path("/content/image_darkening600")
out = local / "smoke_v3"
destination = base / "image_darkening600_20261015/smoke_v3"
root = Path("/content/datasets/nuscenes_full")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def canonical(value):
    """Compare metadata values independently of dictionary insertion order."""
    import numpy as np

    def encode(item):

        if isinstance(item, type):
            return {
                "__class__": item.__module__ + "." + item.__qualname__
            }
        if isinstance(item, (set, frozenset)):
            encoded = [encode(value) for value in item]
            return {
                "__set__": sorted(
                    encoded,
                    key=lambda value: json.dumps(value, sort_keys=True)
                )
            }
        if isinstance(item, dict):
            assert all(isinstance(key, str) for key in item)
            return {
                key: encode(item[key])
                for key in sorted(item)
            }
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            return {
                "__tensor__": str(item.dtype),
                "shape": list(item.shape),
                "bytes": array.tobytes().hex(),
            }
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                return {
                    "__object_array__": encode(item.tolist()),
                    "shape": list(item.shape),
                }
            return {
                "__array__": str(item.dtype),
                "shape": list(item.shape),
                "bytes": np.ascontiguousarray(item).tobytes().hex(),
            }
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, tuple):
            return {"__tuple__": [encode(v) for v in item]}
        if isinstance(item, list):
            return [encode(v) for v in item]
        if isinstance(item, Path):
            return {"__path__": str(item)}
        if item is None or isinstance(item, (str, bool, int, float)):
            return item
        raise TypeError(
            f"Unsupported metadata type: {type(item).__name__}"
        )

    return json.dumps(
        encode(value), sort_keys=True, separators=(",", ":")
    )


protocol_path = local / "protocol.json"
assert sha256(protocol_path) == (
    "e4f63cc095c4bb76d7355e7bc1adbbdfb1005e5a630d6538f8b4e4425588a391"
)
protocol = json.loads(protocol_path.read_text())
tokens = (local / "sample_tokens.txt").read_text().splitlines()
assert len(tokens) == len(set(tokens)) == 600

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd="/content/mmdetection3d", text=True
).strip()
assert commit == protocol["source_commit"]
assert torch.__version__ == "2.1.2+cu121"
assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
set_random_seed(protocol["seed"])

ann = Path(
    "/content/runtime_benchmark150/"
    "nuscenes_camera_availability600_infos_val.pkl"
)
assert sha256(ann) == (
    "ff6dd3c63e8c3de23b9754947db326d0549bb376271cb7e235a3d54b0c365c44"
)

checkpoint = Path(
    "/content/bevfusion_checkpoints/"
    "bevfusion_lidar-cam_voxel0075_second_secfpn_"
    "8xb4-cyclic-20e_nus-3d-5239b1af.pth"
)
assert sha256(checkpoint) == protocol["checkpoint_sha256"]

cfg = Config.fromfile(
    "/content/mmdetection3d/projects/BEVFusion/configs/"
    "bevfusion_lidar-cam_voxel0075_second_secfpn_"
    "8xb4-cyclic-20e_nus-3d.py"
)
dataset_cfg = cfg.test_dataloader.dataset
dataset_cfg.data_root = str(root) + "/"
dataset_cfg.ann_file = str(ann)
dataset_cfg.metainfo["version"] = "v1.0-trainval"

print("LOADING FUSION MODEL...", flush=True)
model = init_model(cfg, str(checkpoint), device="cuda:0")
load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
model.eval()
cfg.dump(str(out / "resolved_config.py"))

dataset = DATASETS.build(dataset_cfg)
assert len(dataset) == 600
assert [dataset.get_data_info(i)["token"] for i in range(600)] == tokens

original = dataset[0]
image_keys = [k for k in ("img", "imgs") if k in original["inputs"]]
assert len(image_keys) == 1, list(original["inputs"])
image_key = image_keys[0]
raw = original["inputs"][image_key]
assert isinstance(raw, torch.Tensor), type(raw)
assert raw.ndim == 4 and raw.shape[0] == 6, raw.shape
assert raw.shape[1] == 3, raw.shape
assert torch.isfinite(raw).all()
assert raw.min().item() >= 0 and raw.max().item() <= 255

original_image_hash = tensor_hash(raw)
original_points_hash = tensor_hash(original["inputs"]["points"])
original_metadata = canonical(original["data_samples"].metainfo)
records = []
reference_processed = {}

for condition in protocol["conditions"]:
    name = condition["name"]
    gain = float(condition["image_value_multiplier"])
    print("\nCHECKING CONDITION:", name, flush=True)
    set_random_seed(protocol["seed"])

    sample = copy.deepcopy(original)
    expected = raw.to(torch.float32) * gain
    sample["inputs"][image_key] = expected.clone()

    assert sample["inputs"][image_key].dtype == torch.float32
    assert torch.equal(sample["inputs"][image_key], expected)
    assert tensor_hash(sample["inputs"]["points"]) == original_points_hash
    assert canonical(sample["data_samples"].metainfo) == original_metadata
    assert tensor_hash(original["inputs"][image_key]) == original_image_hash

    captured = {}

    def check_before(module, args):
        incoming = args[0]["inputs"][image_key]
        assert isinstance(incoming, (list, tuple)) and len(incoming) == 1
        assert torch.equal(incoming[0].cpu(), expected.cpu())
        captured["pre_normalization_image_sha256"] = tensor_hash(incoming[0])
        captured["pre_normalization_check"] = True

    def check_after(module, args, processed):
        inputs = processed["inputs"]
        images = inputs["imgs"]
        assert images.ndim == 5 and images.shape[:2] == (1, 6)
        assert torch.isfinite(images).all()
        meta = processed["data_samples"][0].metainfo

        points = [tensor_hash(p) for p in inputs["points"]]
        geometry = canonical({
            key: meta[key] for key in
            ("cam2img", "lidar2cam", "cam2lidar", "img_aug_matrix")
        })
        if not reference_processed:
            reference_processed.update(points=points, geometry=geometry)
        assert points == reference_processed["points"]
        assert geometry == reference_processed["geometry"]

        captured.update({
            "processed_image_shape": list(images.shape),
            "processed_image_sha256": tensor_hash(images),
            "points_sha256": points,
            "geometry_sha256": hashlib.sha256(geometry.encode()).hexdigest(),
            "lidar_and_calibration_unchanged": True,
        })

    before_handle = model.data_preprocessor.register_forward_pre_hook(
        check_before
    )
    after_handle = model.data_preprocessor.register_forward_hook(check_after)
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            predictions = model.test_step(pseudo_collate([sample]))
        torch.cuda.synchronize()
    finally:
        before_handle.remove()
        after_handle.remove()

    assert len(predictions) == 1 and captured["pre_normalization_check"]
    instances = predictions[0].pred_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu()
    scores = instances.scores_3d.detach().cpu()
    labels = instances.labels_3d.detach().cpu()
    assert len(boxes) == len(scores) == len(labels)
    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert ((labels >= 0) & (labels < 10)).all()

    payload = {
        "sample_token": tokens[0],
        "boxes_3d": boxes.tolist(),
        "scores_3d": scores.tolist(),
        "labels_3d": labels.tolist(),
    }
    prediction_path = out / f"{name}_prediction.json"
    prediction_path.write_text(json.dumps(payload, indent=2) + "\n")

    record = {
        "condition": name,
        "image_value_multiplier": gain,
        "detections": len(scores),
        "finite_predictions": True,
        "input_image_key": image_key,
        "raw_image_shape": list(raw.shape),
        "transformed_value_range": [
            expected.min().item(), expected.max().item()
        ],
        "prediction_sha256": sha256(prediction_path),
        "input_checks": captured,
    }
    records.append(record)
    print(json.dumps(record, indent=2), flush=True)
    del predictions, instances, sample

assert len({r["input_checks"]["pre_normalization_image_sha256"]
            for r in records}) == 3
assert len({r["input_checks"]["processed_image_sha256"]
            for r in records}) == 3

audit = {
    "status": "IMAGE_DARKENING_THREE_CONDITION_SMOKE_PASS",
    "sample_token": tokens[0],
    "dataset_samples": 600,
    "protocol_sha256": sha256(protocol_path),
    "annotation_sha256": sha256(ann),
    "checkpoint_sha256": sha256(checkpoint),
    "source_commit": commit,
    "strict_checkpoint_load": True,
    "autocast": False,
    "tf32": False,
    "conditions": records,
    "scope": (
        "One shared frame tested under three conditions. Float32 image "
        "multiplication verified before data-preprocessor normalization. "
        "LiDAR and calibration unchanged. Accuracy evaluation pending."
    ),
}
(out / "smoke_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

for source in out.iterdir():
    if source.is_file() and source.suffix in (".py", ".json"):
        target = destination / source.name
        if target.exists():
            assert sha256(target) == sha256(source), (
                f"Existing evidence differs: {target}"
            )
        else:
            shutil.copy2(source, target)
        assert sha256(source) == sha256(target)

print("\nTHREE-CONDITION DARKENING SMOKE AND DRIVE BACKUP PASS", flush=True)
