import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import mmengine
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS, METRICS

root = Path("/content/datasets/nuscenes_full")
local = Path("/content/lidar_point_reduction600")
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/lidar_point_reduction600_20261015"
)
out = local / "inference"
destination = study / "inference"
configs = Path("/content/mmdetection3d/projects/BEVFusion/configs")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256(source)
    if target.exists():
        assert sha256(target) == expected, f"Existing mismatch: {target}"
        return
    temporary = target.with_name(target.name + ".uploading")
    shutil.copy2(source, temporary)
    assert sha256(temporary) == expected
    temporary.replace(target)


import numpy as np
import subprocess

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

def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

protocol_path = local / "protocol.json"
assert sha256(protocol_path) == (
    "f6037b05188e2eb8c0b51ef6c3ac76d99085831866f1ad034b348277cb2f43f1"
)
protocol = json.loads(protocol_path.read_text())
smoke = json.loads((study / "smoke/smoke_audit.json").read_text())
assert smoke["status"] == "PAIRED_POINT_REDUCTION_SIX_CONDITION_SMOKE_PASS"
assert smoke["protocol_sha256"] == sha256(protocol_path)
tokens = (local / "sample_tokens.txt").read_text().splitlines()
assert len(tokens) == len(set(tokens)) == 600
assert sha256(local / "sample_tokens.txt") == protocol["source_tokens_sha256"]

ann = Path(
    "/content/runtime_benchmark150/"
    "nuscenes_camera_availability600_infos_val.pkl"
)
assert sha256(ann) == smoke["annotation_sha256"]
receipt = json.loads((
    Path("/content/drive/MyDrive/Major_Project/nuscenes_full_validation")
    / "camera_availability_600_20261014/checkpoint_receipt.json"
).read_text())
for model_name in ("lidar", "lidar_camera"):
    assert receipt[model_name]["sha256"] == \
        protocol["checkpoint_sha256"][model_name]

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd="/content/mmdetection3d", text=True
).strip()
assert commit == protocol["source_commit"]
assert np.__version__ == "1.26.4"
assert torch.__version__ == "2.1.2+cu121"
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

raw_reference = {}
selected_reference = {}
unchanged_reference = {}

def verify_pairing(token, record):
    raw = (record["original_points"], record["original_points_sha256"])
    if token not in raw_reference:
        raw_reference[token] = raw
    assert raw_reference[token] == raw, (
        "Original points changed across conditions/models", token
    )

    key = (token, float(condition["retained_fraction"]))
    selected = (
        record["retained_points"],
        record["selected_points_sha256"],
        record["selected_indices_sha256"],
    )
    if key not in selected_reference:
        selected_reference[key] = selected
    assert selected_reference[key] == selected, (
        "Retained points differ across models", token
    )

    key = (model_name, token)
    unchanged = (record["images_sha256"], record["metadata_sha256"])
    if key not in unchanged_reference:
        unchanged_reference[key] = unchanged
    assert unchanged_reference[key] == unchanged, (
        "Images or metadata changed across retention levels", token
    )

print("SIX-CONDITION INFERENCE PREFLIGHT PASS", flush=True)

filenames = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}
summaries = []

for condition in protocol["conditions"]:
    name = condition["name"]
    model_name = condition["model"]
    print("\nSTARTING CONDITION:", name, flush=True)
    set_random_seed(protocol["seed"])

    model_out = out / name
    drive_out = destination / name
    frames = model_out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    drive_out.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(configs / filenames[model_name]))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset_cfg.data_root = str(root) + "/"
    dataset_cfg.ann_file = str(ann)
    dataset_cfg.metainfo["version"] = "v1.0-trainval"

    checkpoint = Path(receipt[model_name]["path"])
    assert sha256(checkpoint) == receipt[model_name]["sha256"]

    run_protocol = {
        "condition": condition,
        "frozen_protocol_sha256": sha256(local / "protocol.json"),
        "annotation_sha256": sha256(ann),
        "checkpoint_sha256": receipt[model_name]["sha256"],
        "torch": torch.__version__,
        "seed": protocol["seed"],
        "source_commit": protocol["source_commit"],
        "script_sha256": sha256(__file__),
        "transformation": protocol["transformation"],
        "autocast": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "numpy": np.__version__,
    }
    protocol_path = model_out / "protocol.json"
    protocol_path.write_text(json.dumps(run_protocol, indent=2) + "\n")
    protocol_hash = sha256(protocol_path)
    copy_verified(protocol_path, drive_out / "protocol.json")

    model = init_model(cfg, str(checkpoint), device="cuda:0")
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model.eval()
    dataset = DATASETS.build(dataset_cfg)
    assert [dataset.get_data_info(i)["token"] for i in range(600)] == tokens
    cfg.dump(str(model_out / "resolved_config.py"))
    copy_verified(
        model_out / "resolved_config.py", drive_out / "resolved_config.py"
    )


    hook_calls = [0]
    fraction = float(condition["retained_fraction"])
    assert fraction in (1.0, 0.5, 0.25)
    current_input = {}
    input_records = []

    def reduce_before_normalization(module, args):
        batch = args[0]
        inputs = batch["inputs"]
        assert len(inputs["points"]) == 1
        raw = inputs["points"][0]
        assert raw.ndim == 2 and len(raw) > 0
        assert torch.isfinite(raw).all()

        changed, indices = select_points(raw, token, fraction)
        assert len(changed) == max(1, int(np.floor(fraction * len(raw))))
        assert torch.isfinite(changed).all()

        image_hash = None
        if model_name == "lidar_camera":
            assert len(inputs["img"]) == 1
            image_hash = tensor_hash(inputs["img"][0])

        current_input.update({
            "original_points": len(raw),
            "retained_points": len(changed),
            "original_points_sha256": tensor_hash(raw),
            "selected_points_sha256": tensor_hash(changed),
            "selected_indices_sha256": hashlib.sha256(
                indices.astype("<i8").tobytes()
            ).hexdigest(),
            "images_sha256": image_hash,
            "metadata_sha256": hashlib.sha256(
                canonical(batch["data_samples"][0].metainfo).encode()
            ).hexdigest(),
        })
        # Only replace the point tensor. Images and metadata are untouched.
        inputs["points"] = [changed]
        hook_calls[0] += 1

    handle = model.data_preprocessor.register_forward_pre_hook(
        reduce_before_normalization
    )
    results, pending = [], []
    created = reused = 0
    started = time.perf_counter()

    try:
        for index, token in enumerate(tokens):
            current_input.clear()
            path = frames / f"{token}.pt"
            drive_path = drive_out / "frames" / path.name

            if not path.exists() and drive_path.exists():
                copy_verified(drive_path, path)

            if path.exists():
                saved = torch.load(path, map_location="cpu")
                assert saved["protocol_sha256"] == protocol_hash
                assert saved["sample_token"] == token
                result = saved["result"]
                assert result["sample_idx"] == index
                input_check = saved["input_check"]
                reused += 1
            else:
                with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                    prediction = model.test_step(
                        pseudo_collate([dataset[index]])
                    )[0]
                torch.cuda.synchronize()
                assert prediction.metainfo["sample_idx"] == index

                instances = prediction.pred_instances_3d.cpu()
                for field in list(instances.keys()):
                    instances[field] = instances[field].to("cpu")
                assert instances.bboxes_3d.tensor.device.type == "cpu"
                assert torch.isfinite(instances.bboxes_3d.tensor).all()
                assert torch.isfinite(instances.scores_3d).all()
                assert (
                    (instances.scores_3d >= 0)
                    & (instances.scores_3d <= 1)
                ).all()
                assert (
                    (instances.labels_3d >= 0)
                    & (instances.labels_3d < 10)
                ).all()
                assert len(instances.bboxes_3d) == len(instances.scores_3d)
                result = {
                    "sample_idx": index,
                    "pred_instances_3d": instances,
                }
                assert current_input
                input_check = dict(current_input)
                temporary = path.with_suffix(".saving")
                torch.save({
                    "protocol_sha256": protocol_hash,
                    "sample_token": token,
                    "result": result,
                    "input_check": input_check,
                }, temporary)
                temporary.replace(path)
                created += 1
                del prediction

            verify_pairing(token, input_check)
            input_records.append({
                "sample_token": token,
                **input_check,
            })
            results.append(result)
            pending.append((path, drive_path))
            if (index + 1) % 50 == 0:
                for source, target in pending:
                    copy_verified(source, target)
                pending.clear()
                print(json.dumps({
                    "condition": name,
                    "index": index + 1,
                    "total": 600,
                    "created": created,
                    "reused": reused,
                    "backed_up": index + 1,
                    "elapsed_seconds": round(
                        time.perf_counter() - started, 1
                    ),
                }), flush=True)
    finally:
        handle.remove()

    assert hook_calls[0] == created
    assert len(input_records) == 600
    input_audit = model_out / "input_checks.json"
    input_audit.write_text(json.dumps({
        "condition": name,
        "retained_fraction": fraction,
        "protocol_sha256": protocol_hash,
        "records": input_records,
    }, indent=2) + "\n")
    copy_verified(input_audit, drive_out / input_audit.name)

    print("FORMATTING:", name, flush=True)
    metric_cfg = dict(cfg.test_evaluator)
    metric_cfg.update(
        data_root=str(root) + "/",
        ann_file=str(ann),
        modality=dict(dataset_cfg.modality),
        format_only=True,
        jsonfile_prefix=str(model_out / "official_results"),
    )
    formatter = METRICS.build(metric_cfg)
    formatter.dataset_meta = dataset.metainfo
    formatter.version = "v1.0-trainval"
    formatter.data_infos = mmengine.load(str(ann))["data_list"]
    files, temporary_dir = formatter.format_results(
        results,
        classes=dataset.metainfo["classes"],
        jsonfile_prefix=str(model_out / "official_results"),
    )
    result_path = Path(files["pred_instances_3d"])
    data = json.loads(result_path.read_text())
    assert set(data["results"]) == set(tokens)

    summary = {
        "status": "CONDITION_INFERENCE_AND_FORMATTING_PASS",
        "condition": name,
        "samples": 600,
        "created": created,
        "reused": reused,
        "results_sha256": sha256(result_path),
        "protocol_sha256": protocol_hash,
        "input_checks_sha256": sha256(input_audit),
        "scope": "Predictions complete; accuracy evaluation pending.",
    }
    audit_path = model_out / "inference_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2) + "\n")
    copy_verified(result_path, drive_out / "results_nusc.json")
    copy_verified(audit_path, drive_out / audit_path.name)
    summaries.append(summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("CONDITION RESULTS VERIFIED ON DRIVE:", name, flush=True)

    del model, dataset, formatter, results, result, data
    gc.collect()
    torch.cuda.empty_cache()

assert len(raw_reference) == 600
assert len(selected_reference) == 1800
assert len(unchanged_reference) == 1200

report = {
    "all_600_original_point_inputs_matched": True,
    "all_retained_point_subsets_matched_across_models": True,
    "images_and_metadata_matched_across_retention_levels": True,
    "status": "PAIRED_POINT_REDUCTION_SIX_CONDITION_INFERENCE_PASS",
    "samples_per_condition": 600,
    "scenes": 150,
    "conditions": summaries,
    "scope": "Inference and formatting complete; accuracy pending.",
}
report_path = out / "six_condition_inference.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
copy_verified(report_path, destination / report_path.name)
print(json.dumps(report, indent=2), flush=True)
print("ALL SIX POINT-REDUCTION CONDITIONS PRESERVED ON DRIVE", flush=True)
