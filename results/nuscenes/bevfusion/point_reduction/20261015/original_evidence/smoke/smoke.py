
import copy
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

local = Path("/content/lidar_point_reduction600")
out = local / "smoke"
root = Path("/content/datasets/nuscenes_full")
base = Path("/content/drive/MyDrive/Major_Project/nuscenes_full_validation")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
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

def select_points(points, token, fraction):
    assert points.ndim == 2 and len(points) > 0
    if fraction == 1.0:
        indices = np.arange(len(points), dtype=np.int64)
    else:
        seed = int.from_bytes(
            hashlib.sha256(
                ("20261002:" + token).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        generator = np.random.Generator(np.random.PCG64(seed))
        permutation = generator.permutation(len(points))
        count = max(1, int(np.floor(fraction * len(points))))
        indices = np.sort(permutation[:count])
    selected = points[
        torch.as_tensor(indices, dtype=torch.long, device=points.device)
    ].clone()
    return selected, indices

protocol_path = local / "protocol.json"
assert sha256(protocol_path) == (
    "f6037b05188e2eb8c0b51ef6c3ac76d99085831866f1ad034b348277cb2f43f1"
)
protocol = json.loads(protocol_path.read_text())
tokens = (local / "sample_tokens.txt").read_text().splitlines()
assert len(tokens) == len(set(tokens)) == 600
assert sha256(local / "sample_tokens.txt") == protocol["source_tokens_sha256"]
assert np.__version__ == "1.26.4"
assert torch.__version__ == "2.1.2+cu121"
assert torch.cuda.is_available()

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd="/content/mmdetection3d", text=True
).strip()
assert commit == protocol["source_commit"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

ann = Path(
    "/content/runtime_benchmark150/"
    "nuscenes_camera_availability600_infos_val.pkl"
)
assert sha256(ann) == (
    "ff6dd3c63e8c3de23b9754947db326d0549bb376271cb7e235a3d54b0c365c44"
)
receipt = json.loads((
    base / "camera_availability_600_20261014/checkpoint_receipt.json"
).read_text())

filenames = {
    "lidar":
        "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera":
        "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}
reference_raw = None
reference_selected = {}
records = []

for model_name, filename in filenames.items():
    print("\nLOADING:", model_name, flush=True)
    set_random_seed(protocol["seed"])
    cfg = Config.fromfile(str(
        Path("/content/mmdetection3d/projects/BEVFusion/configs") / filename
    ))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset_cfg.data_root = str(root) + "/"
    dataset_cfg.ann_file = str(ann)
    dataset_cfg.metainfo["version"] = "v1.0-trainval"

    checkpoint = Path(receipt[model_name]["path"])
    assert sha256(checkpoint) == protocol["checkpoint_sha256"][model_name]
    model = init_model(cfg, str(checkpoint), device="cuda:0")
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model.eval()
    cfg.dump(str(out / f"{model_name}_resolved_config.py"))

    dataset = DATASETS.build(dataset_cfg)
    assert len(dataset) == 600
    assert [dataset.get_data_info(i)["token"] for i in range(600)] == tokens
    original = dataset[0]
    raw_points = original["inputs"]["points"]
    raw_hash = tensor_hash(raw_points)

    if reference_raw is None:
        reference_raw = raw_hash
    assert raw_hash == reference_raw, (
        "Models do not start with identical points; stop before inference."
    )

    original_metadata = canonical(original["data_samples"].metainfo)
    original_images = (
        tensor_hash(original["inputs"]["img"])
        if model_name == "lidar_camera" else None
    )
    processed_image_reference = []
    selected_indices = {}

    for condition in protocol["conditions"]:
        if condition["model"] != model_name:
            continue
        name = condition["name"]
        fraction = condition["retained_fraction"]
        print("CHECKING:", name, flush=True)
        set_random_seed(protocol["seed"])

        expected, indices = select_points(raw_points, tokens[0], fraction)
        selected_indices[fraction] = indices
        selected_hash = tensor_hash(expected)
        if fraction not in reference_selected:
            reference_selected[fraction] = selected_hash
        assert selected_hash == reference_selected[fraction]
        assert len(expected) == max(1, int(np.floor(fraction*len(raw_points))))

        sample = copy.deepcopy(original)
        assert canonical(sample["data_samples"].metainfo) == original_metadata
        captured = {}

        def reduce_before(module, args):
            inputs = args[0]["inputs"]
            points = inputs["points"]
            assert len(points) == 1
            assert tensor_hash(points[0]) == raw_hash
            if model_name == "lidar_camera":
                assert tensor_hash(inputs["img"][0]) == original_images
            changed, chosen = select_points(points[0], tokens[0], fraction)
            assert np.array_equal(chosen, indices)
            assert torch.equal(changed.cpu(), expected.cpu())
            inputs["points"] = [changed]
            captured["preprocessing_input_check"] = True

        def check_after(module, args, processed):
            points = processed["inputs"]["points"]
            assert len(points) == 1
            assert tensor_hash(points[0]) == selected_hash
            if model_name == "lidar_camera":
                image_hash = tensor_hash(processed["inputs"]["imgs"])
                if not processed_image_reference:
                    processed_image_reference.append(image_hash)
                assert image_hash == processed_image_reference[0]
                captured["processed_images_unchanged"] = True
            captured["processed_points_sha256"] = tensor_hash(points[0])

        h1 = model.data_preprocessor.register_forward_pre_hook(reduce_before)
        h2 = model.data_preprocessor.register_forward_hook(check_after)
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                prediction = model.test_step(pseudo_collate([sample]))[0]
            torch.cuda.synchronize()
        finally:
            h1.remove()
            h2.remove()

        assert captured["preprocessing_input_check"]
        assert tensor_hash(original["inputs"]["points"]) == raw_hash
        assert canonical(original["data_samples"].metainfo) == original_metadata
        if model_name == "lidar_camera":
            assert tensor_hash(original["inputs"]["img"]) == original_images

        instances = prediction.pred_instances_3d
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
        path = out / f"{name}_prediction.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        record = {
            "condition": name,
            "retained_fraction": fraction,
            "original_points": len(raw_points),
            "retained_points": len(expected),
            "original_points_sha256": raw_hash,
            "selected_points_sha256": selected_hash,
            "selected_indices_sha256": hashlib.sha256(
                indices.astype("<i8").tobytes()
            ).hexdigest(),
            "detections": len(scores),
            "prediction_sha256": sha256(path),
            "finite_predictions": True,
            "calibration_metadata_unchanged": True,
            "input_checks": captured,
        }
        records.append(record)
        print(json.dumps(record, indent=2), flush=True)
        del prediction, instances, sample

    assert set(selected_indices[0.25]) <= set(selected_indices[0.5])
    assert set(selected_indices[0.5]) <= set(selected_indices[1.0])
    del model, dataset, original
    gc.collect()
    torch.cuda.empty_cache()

assert len(records) == 6
audit = {
    "status": "PAIRED_POINT_REDUCTION_SIX_CONDITION_SMOKE_PASS",
    "sample_token": tokens[0],
    "protocol_sha256": sha256(protocol_path),
    "annotation_sha256": sha256(ann),
    "source_commit": commit,
    "numpy": np.__version__,
    "torch": torch.__version__,
    "identical_original_points_across_models": True,
    "identical_retained_points_across_models": True,
    "nested_point_subsets": True,
    "conditions": records,
    "scope": "One shared frame; full 600-frame inference and accuracy pending.",
}
(out / "smoke_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print("\nSIX-CONDITION POINT-REDUCTION SMOKE PASS", flush=True)
