
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import pickle
import subprocess

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

study = Path("/kaggle/working/camera_challenges600")
out = study / "smoke_v2"
repo = Path("/tmp/camera_challenges/mmdetection3d")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

def encode(value):
    if isinstance(value, dict):
        return {str(k): encode(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, torch.Tensor):
        return {
            "tensor": str(value.dtype),
            "shape": list(value.shape),
            "hash": tensor_hash(value),
        }
    if isinstance(value, np.ndarray):
        return {
            "array": str(value.dtype),
            "shape": list(value.shape),
            "hash": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, type):
        return {"class": value.__module__ + "." + value.__qualname__}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported metadata type: {type(value)}")

def canonical(value):
    return json.dumps(encode(value), sort_keys=True, separators=(",", ":"))

protocol_path = study / "protocol.json"
protocol_hash = (
    "ab01b793d13d7b2b06dee318cd99ee3d3"
    "c39b6d7f04206538052d90c54533d31"
)
assert sha256(protocol_path) == protocol_hash
protocol = json.loads(protocol_path.read_text())

cpu_audit = json.loads(
    (study / "cpu_validation/cpu_validation_audit.json").read_text()
)
assert cpu_audit["status"] == "CAMERA_CHALLENGES_CPU_TRANSFORMS_PASS"
module_path = study / "cpu_validation/corruptions.py"
assert sha256(module_path) == cpu_audit["module_sha256"]

spec = importlib.util.spec_from_file_location("checked_corruptions", module_path)
transforms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transforms)

# Locate condition records within the frozen protocol.
wanted_names = {
    "clean", "blur_sigma1", "blur_sigma2",
    "noise_std10", "noise_std25",
    "calibration_yaw1", "calibration_yaw3",
}
found = {}

def find_conditions(value):
    if isinstance(value, dict):
        if value.get("name") in wanted_names:
            name = value["name"]
            assert name not in found, f"Duplicate condition: {name}"
            found[name] = value
        for child in value.values():
            find_conditions(child)
    elif isinstance(value, list):
        for child in value:
            find_conditions(child)

find_conditions(protocol)
assert set(found) == wanted_names, list(found)

path_audit = json.loads((study / "annotation_path_audit.json").read_text())
assert path_audit["status"] == "KAGGLE_ANNOTATION_PATH_RESOLUTION_PASS"
annotation = Path(path_audit["annotation_path"])
assert sha256(annotation) == path_audit["annotation_sha256"]

receipt = json.loads((study / "checkpoint_receipt.json").read_text())
checkpoint = Path(receipt["path"])
assert sha256(checkpoint) == receipt["sha256"]

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
).strip()
assert commit == "fe25f7a51d36e3702f961e198894580d83c4387b"

set_random_seed(20261002, deterministic=False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
assert torch.__version__ == "2.1.2+cu121"
assert torch.cuda.device_count() == 1

config_path = repo / (
    "projects/BEVFusion/configs/"
    "bevfusion_lidar-cam_voxel0075_second_secfpn_"
    "8xb4-cyclic-20e_nus-3d.py"
)
cfg = Config.fromfile(str(config_path))

# Full strict checkpoint loading makes separate backbone pretraining unnecessary.
cfg.model.img_backbone.init_cfg = None
cfg.load_from = None
cfg.resume = False

dataset_cfg = copy.deepcopy(cfg.test_dataloader.dataset)
dataset_cfg.data_root = path_audit["sensor_root"]
dataset_cfg.ann_file = str(annotation)
dataset_cfg.test_mode = True
dataset_cfg.data_prefix = dict(
    pts="samples/LIDAR_TOP",
    sweeps="sweeps/LIDAR_TOP",
    img="",
)
cfg.test_dataloader.dataset = dataset_cfg
cfg.dump(str(out / "resolved_config.py"))

print("LOADING FUSION MODEL...", flush=True)
model = init_model(cfg, checkpoint=None, device="cuda:0")
load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
model.eval()
print("STRICT CHECKPOINT LOAD PASS", flush=True)

dataset = DATASETS.build(dataset_cfg)
tokens = (study / "sample_tokens.txt").read_text().splitlines()
assert len(dataset) == len(tokens) == 600
assert [
    dataset.get_data_info(i)["token"] for i in range(len(dataset))
] == tokens

print("CHECKING ALL PARSED SENSOR PATHS...", flush=True)
parsed_paths = set()
for index in range(len(dataset)):
    info = dataset.get_data_info(index)
    paths = [info["lidar_points"]["lidar_path"]]
    paths.extend(
        camera["img_path"] for camera in info["images"].values()
    )
    paths.extend(
        sweep["lidar_points"]["lidar_path"]
        for sweep in info.get("lidar_sweeps", [])
    )
    for value in paths:
        path = Path(value)
        assert path.is_file(), f"Parsed sensor missing: {path}"
        parsed_paths.add(str(path))
    if (index + 1) % 100 == 0:
        print(f"Parsed paths checked: {index + 1}/600", flush=True)

assert len(parsed_paths) == 9600, len(parsed_paths)
(out / "parsed_sensor_paths.json").write_text(
    json.dumps(sorted(parsed_paths), indent=2) + "\n"
)
print("ALL PARSED SENSOR PATHS PASS", flush=True)

set_random_seed(20261002, deterministic=False)
original = dataset[0]
assert isinstance(original["inputs"]["img"], torch.Tensor)
assert tuple(original["inputs"]["img"].shape) == (6, 3, 256, 704)

token = tokens[0]
original_image = original["inputs"]["img"].clone()
original_points_hash = tensor_hash(original["inputs"]["points"])
original_metadata = canonical(original["data_samples"].metainfo)

records = []
clean_processed_points = None

for name in (
    "clean", "blur_sigma1", "blur_sigma2",
    "noise_std10", "noise_std25",
    "calibration_yaw1", "calibration_yaw3",
):
    print("\nCHECKING CONDITION:", name, flush=True)
    condition = found[name]
    sample = copy.deepcopy(original)

    raw_image = transforms.corrupt_images(
        original_image.cpu().numpy(), condition, token
    )
    sample["inputs"]["img"] = torch.from_numpy(
        np.ascontiguousarray(raw_image)
    ).to(dtype=original_image.dtype)

    changed_keys = {"cam2lidar", "lidar2cam", "lidar2img"}
    if condition["family"] == "calibration":
        before = sample["data_samples"].metainfo
        after = transforms.corrupt_calibration(
            before, condition["yaw_degrees"]
        )
        assert canonical({
            k: v for k, v in before.items() if k not in changed_keys
        }) == canonical({
            k: v for k, v in after.items() if k not in changed_keys
        })
        sample["data_samples"].set_metainfo(after)
        assert tensor_hash(sample["inputs"]["img"]) == tensor_hash(
            original_image
        )
    else:
        assert canonical(sample["data_samples"].metainfo) == original_metadata

    assert tensor_hash(sample["inputs"]["points"]) == original_points_hash
    assert torch.isfinite(sample["inputs"]["img"]).all()

    if name == "clean":
        assert torch.equal(sample["inputs"]["img"], original_image)
    elif condition["family"] in ("blur", "noise"):
        assert not torch.equal(sample["inputs"]["img"], original_image)

    captured = {}

    def inspect_preprocessed(module, args, result):
        processed = result["inputs"]
        images = processed["imgs"]
        points = processed["points"]
        assert tuple(images.shape) == (1, 6, 3, 256, 704)
        assert torch.isfinite(images).all()
        captured["processed_image_sha256"] = tensor_hash(images)
        captured["processed_points_sha256"] = [
            tensor_hash(value) for value in points
        ]
        captured["processed_metadata"] = canonical(
            result["data_samples"][0].metainfo
        )

    handle = model.data_preprocessor.register_forward_hook(
        inspect_preprocessed
    )
    set_random_seed(20261002, deterministic=False)
    try:
        with torch.inference_mode():
            predictions = model.test_step(pseudo_collate([sample]))
        torch.cuda.synchronize()
    finally:
        handle.remove()

    assert len(predictions) == 1 and captured
    if clean_processed_points is None:
        clean_processed_points = captured["processed_points_sha256"]
    assert captured["processed_points_sha256"] == clean_processed_points

    instances = predictions[0].pred_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu()
    scores = instances.scores_3d.detach().cpu()
    labels = instances.labels_3d.detach().cpu()

    assert torch.isfinite(boxes).all()
    assert torch.isfinite(scores).all()
    assert len(boxes) == len(scores) == len(labels)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert ((labels >= 0) & (labels < 10)).all()

    prediction_path = out / f"{name}_prediction.pt"
    torch.save({
        "sample_token": token,
        "condition": name,
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }, prediction_path)

    record = {
        "condition": name,
        "sample_token": token,
        "detections": len(scores),
        "finite_predictions": True,
        "raw_image_sha256": tensor_hash(sample["inputs"]["img"]),
        "processed_image_sha256": captured["processed_image_sha256"],
        "processed_points_sha256": captured["processed_points_sha256"],
        "metadata_sha256": hashlib.sha256(
            canonical(sample["data_samples"].metainfo).encode()
        ).hexdigest(),
        "prediction_sha256": sha256(prediction_path),
        "lidar_points_unchanged": True,
    }
    records.append(record)
    print(json.dumps(record, indent=2), flush=True)

assert tensor_hash(original["inputs"]["points"]) == original_points_hash
assert canonical(original["data_samples"].metainfo) == original_metadata

audit = {
    "status": "KAGGLE_SEVEN_CONDITION_MODEL_SMOKE_PASS",
    "samples_tested_per_condition": 1,
    "planned_samples_per_condition": 600,
    "protocol_sha256": protocol_hash,
    "annotation_sha256": sha256(annotation),
    "checkpoint_sha256": receipt["sha256"],
    "transform_module_sha256": sha256(module_path),
    "source_commit": commit,
    "gpu": torch.cuda.get_device_name(0),
    "conditions": records,
    "backbone_initialization": (
        "Separate pretrained initialization disabled; full checkpoint "
        "loaded strictly before prediction."
    ),
    "scope": (
        "One shared sample checked under all seven conditions. "
        "Full inference and accuracy evaluation remain pending."
    ),
}
(out / "smoke_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)
print("\nKAGGLE_SEVEN_CONDITION_MODEL_SMOKE_PASS", flush=True)
