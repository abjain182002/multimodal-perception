
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
out = study / "inference"
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


import time
import shutil
from mmdet3d.evaluation.metrics import NuScenesMetric

smoke = json.loads((study / "smoke_v2/smoke_audit.json").read_text())
assert smoke["status"] == "KAGGLE_SEVEN_CONDITION_MODEL_SMOKE_PASS"
assert smoke["protocol_sha256"] == protocol_hash
assert smoke["checkpoint_sha256"] == receipt["sha256"]
assert smoke["annotation_sha256"] == sha256(annotation)
assert smoke["transform_module_sha256"] == sha256(module_path)
assert shutil.disk_usage(out).free > 2_000_000_000

identity = {
    "protocol_sha256": protocol_hash,
    "annotation_sha256": sha256(annotation),
    "checkpoint_sha256": receipt["sha256"],
    "transform_module_sha256": sha256(module_path),
    "source_commit": commit,
}

with annotation.open("rb") as stream:
    data_infos = pickle.load(stream)["data_list"]

classes = dataset.metainfo["classes"]
summary = []

def atomic_json(path, value):
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)

for name in (
    "clean", "blur_sigma1", "blur_sigma2",
    "noise_std10", "noise_std25",
    "calibration_yaw1", "calibration_yaw3",
):
    condition = found[name]
    folder = out / name
    predictions_folder = folder / "predictions"
    predictions_folder.mkdir(parents=True, exist_ok=True)

    atomic_json(folder / "protocol.json", {
        **identity,
        "condition": condition,
        "seed": 20261002,
        "samples": 600,
        "autocast": False,
        "tf32": False,
        "cudnn_benchmark": False,
    })

    print("\nSTARTING CONDITION:", name, flush=True)
    started = time.monotonic()
    created = reused = 0
    formatted_results = []
    ledger = []

    for index, token in enumerate(tokens):
        destination = predictions_folder / f"{index:04d}_{token}.pt"

        if destination.exists():
            saved = torch.load(destination, map_location="cpu")
            assert saved["identity"] == identity
            assert saved["condition"] == name
            assert saved["sample_token"] == token
            assert saved["index"] == index
            reused += 1

        else:
            set_random_seed(20261002, deterministic=False)
            sample = dataset[index]
            assert tuple(sample["inputs"]["img"].shape) == (6, 3, 256, 704)

            original_image = sample["inputs"]["img"].clone()
            original_points = tensor_hash(sample["inputs"]["points"])
            before_metadata = canonical(sample["data_samples"].metainfo)

            image = transforms.corrupt_images(
                original_image.cpu().numpy(), condition, token
            )
            sample["inputs"]["img"] = torch.from_numpy(
                np.ascontiguousarray(image)
            ).to(dtype=original_image.dtype)

            if condition["family"] == "calibration":
                before = sample["data_samples"].metainfo
                after = transforms.corrupt_calibration(
                    before, condition["yaw_degrees"]
                )
                changed = {"cam2lidar", "lidar2cam", "lidar2img"}
                assert canonical({
                    k: v for k, v in before.items() if k not in changed
                }) == canonical({
                    k: v for k, v in after.items() if k not in changed
                })
                sample["data_samples"].set_metainfo(after)
                assert torch.equal(sample["inputs"]["img"], original_image)
            else:
                assert canonical(
                    sample["data_samples"].metainfo
                ) == before_metadata

            assert tensor_hash(sample["inputs"]["points"]) == original_points
            assert torch.isfinite(sample["inputs"]["img"]).all()
            if name == "clean":
                assert torch.equal(sample["inputs"]["img"], original_image)

            expected_geometry = canonical({
                key: sample["data_samples"].metainfo[key]
                for key in ("cam2img", "cam2lidar", "lidar2cam", "lidar2img")
            })
            captured = {}

            def inspect_processed(module, args, result):
                images = result["inputs"]["imgs"]
                assert tuple(images.shape) == (1, 6, 3, 256, 704)
                assert torch.isfinite(images).all()

                captured["processed_image_sha256"] = tensor_hash(images)
                captured["processed_points_sha256"] = [
                    tensor_hash(value)
                    for value in result["inputs"]["points"]
                ]
                assert captured["processed_points_sha256"] == [
                    original_points
                ]
                metadata = result["data_samples"][0].metainfo
                assert canonical({
                    key: metadata[key]
                    for key in ("cam2img", "cam2lidar", "lidar2cam", "lidar2img")
                }) == expected_geometry

            handle = model.data_preprocessor.register_forward_hook(
                inspect_processed
            )
            set_random_seed(20261002, deterministic=False)
            try:
                with torch.inference_mode():
                    prediction = model.test_step(
                        pseudo_collate([sample])
                    )[0]
                torch.cuda.synchronize()
            finally:
                handle.remove()

            assert captured
            instances = prediction.pred_instances_3d
            cpu_prediction = {
                "bboxes_3d": instances.bboxes_3d.to("cpu"),
                "scores_3d": instances.scores_3d.detach().cpu(),
                "labels_3d": instances.labels_3d.detach().cpu(),
            }

            saved = {
                "identity": identity,
                "condition": name,
                "sample_token": token,
                "index": index,
                "prediction": cpu_prediction,
                "input_checks": {
                    "original_image_sha256": tensor_hash(original_image),
                    "original_points_sha256": original_points,
                    "original_metadata_sha256": hashlib.sha256(
                        before_metadata.encode()
                    ).hexdigest(),
                    "transformed_image_sha256": tensor_hash(
                        sample["inputs"]["img"]
                    ),
                    "transformed_metadata_sha256": hashlib.sha256(
                        canonical(sample["data_samples"].metainfo).encode()
                    ).hexdigest(),
                    **captured,
                },
            }
            temporary = destination.with_suffix(".pt.writing")
            torch.save(saved, temporary)
            # Verify the serialized CPU tensors before accepting the file.
            check = torch.load(temporary, map_location="cpu")
            assert check["identity"] == identity
            assert check["sample_token"] == token
            for field in ("scores_3d", "labels_3d"):
                assert torch.equal(
                    check["prediction"][field], cpu_prediction[field]
                )
            assert torch.equal(
                check["prediction"]["bboxes_3d"].tensor,
                cpu_prediction["bboxes_3d"].tensor,
            )
            temporary.replace(destination)
            created += 1

        prediction = saved["prediction"]
        assert torch.isfinite(prediction["bboxes_3d"].tensor).all()
        assert torch.isfinite(prediction["scores_3d"]).all()
        assert (
            (prediction["scores_3d"] >= 0) &
            (prediction["scores_3d"] <= 1)
        ).all()
        assert (
            (prediction["labels_3d"] >= 0) &
            (prediction["labels_3d"] < len(classes))
        ).all()

        # Match every corruption's original inputs to the fresh clean run.
        if name != "clean":
            clean_file = (
                out / "clean/predictions" /
                f"{index:04d}_{token}.pt"
            )
            clean_saved = torch.load(clean_file, map_location="cpu")
            assert clean_saved["identity"] == identity
            for field in (
                "original_image_sha256",
                "original_points_sha256",
                "original_metadata_sha256",
                "processed_points_sha256",
            ):
                assert saved["input_checks"][field] == (
                    clean_saved["input_checks"][field]
                ), (name, index, field)

        formatted_results.append({
            "sample_idx": index,
            "pred_instances_3d": prediction,
        })
        ledger.append({
            "index": index,
            "sample_token": token,
            "prediction_sha256": sha256(destination),
            **saved["input_checks"],
        })

        if (index + 1) % 50 == 0 or index + 1 == len(tokens):
            atomic_json(folder / "progress.json", {
                **identity,
                "condition": name,
                "verified_saved_samples": index + 1,
                "total": len(tokens),
                "created": created,
                "reused": reused,
            })
            print(json.dumps({
                "condition": name,
                "index": index + 1,
                "total": len(tokens),
                "created": created,
                "reused": reused,
                "elapsed_seconds": round(time.monotonic() - started, 1),
            }), flush=True)

    print("FORMATTING NUSCENES PREDICTIONS:", name, flush=True)
    metric = NuScenesMetric(
        data_root=path_audit["sensor_root"],
        ann_file=str(annotation),
        modality={"use_camera": True, "use_lidar": True},
        format_only=True,
        jsonfile_prefix=str(folder / "formatted"),
    )
    metric.dataset_meta = dataset.metainfo
    metric.data_infos = data_infos
    metric.version = "v1.0-trainval"

    result_paths, _ = metric.format_results(
        formatted_results,
        classes=classes,
        jsonfile_prefix=str(folder / "formatted"),
    )
    generated = Path(result_paths["pred_instances_3d"])
    result_path = folder / "results_nusc.json"
    shutil.copy2(generated, result_path)

    result_data = json.loads(result_path.read_text())
    assert set(result_data["results"]) == set(tokens)
    assert len(result_data["results"]) == 600
    for token, detections in result_data["results"].items():
        assert all(item["sample_token"] == token for item in detections)

    atomic_json(folder / "input_checks.json", ledger)
    report = {
        "status": "CAMERA_CHALLENGE_CONDITION_INFERENCE_PASS",
        **identity,
        "condition": name,
        "samples": 600,
        "exact_frozen_token_match": True,
        "results_path": str(result_path),
        "results_sha256": sha256(result_path),
        "input_checks_sha256": sha256(folder / "input_checks.json"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "scope": "Inference and saved records checked; accuracy pending.",
    }
    atomic_json(folder / "inference_audit.json", report)
    summary.append(report)
    atomic_json(out / "completed_conditions.json", summary)
    print(json.dumps(report, indent=2), flush=True)

    del formatted_results, ledger, result_data, metric

atomic_json(out / "seven_condition_inference.json", {
    "status": "KAGGLE_SEVEN_CONDITION_INFERENCE_PASS",
    **identity,
    "samples_per_condition": 600,
    "conditions": summary,
    "scope": (
        "Seven prediction sets saved and checked in Kaggle working storage. "
        "External preservation and accuracy evaluation remain pending."
    ),
})
print("\nKAGGLE_SEVEN_CONDITION_INFERENCE_PASS", flush=True)
