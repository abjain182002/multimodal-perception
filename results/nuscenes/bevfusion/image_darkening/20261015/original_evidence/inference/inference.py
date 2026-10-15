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
local = Path("/content/image_darkening600")
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/image_darkening600_20261015"
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


protocol_path = local / "protocol.json"
assert sha256(protocol_path) == (
    "e4f63cc095c4bb76d7355e7bc1adbbdfb1005e5a630d6538f8b4e4425588a391"
)
protocol = json.loads(protocol_path.read_text())
smoke = json.loads((study / "smoke_v3/smoke_audit.json").read_text())
assert smoke["status"] == "IMAGE_DARKENING_THREE_CONDITION_SMOKE_PASS"
assert sha256(protocol_path) == smoke["protocol_sha256"]

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
assert receipt["lidar_camera"]["sha256"] == protocol["checkpoint_sha256"]

import subprocess
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

print("DARKENING INFERENCE PREFLIGHT PASS", flush=True)

filenames = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}
summaries = []

for condition in protocol["conditions"]:
    name = condition["name"]
    model_name = protocol["model"]
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
    gain = float(condition["image_value_multiplier"])
    assert gain in (1.0, 0.5, 0.25)

    def darken_before_normalization(module, args):
        batch = args[0]
        inputs = batch["inputs"]
        images = inputs["img"]
        assert isinstance(images, (list, tuple)) and len(images) == 1
        raw = images[0]
        assert isinstance(raw, torch.Tensor)
        assert raw.ndim == 4 and raw.shape[:2] == (6, 3)
        assert torch.isfinite(raw).all()
        assert raw.min().item() >= 0 and raw.max().item() <= 255

        # Replace only images; leave points, geometry and data samples intact.
        transformed = raw.to(torch.float32) * gain
        assert transformed.shape == raw.shape
        assert transformed.dtype == torch.float32
        assert torch.isfinite(transformed).all()
        inputs["img"] = [transformed]
        hook_calls[0] += 1

    handle = model.data_preprocessor.register_forward_pre_hook(
        darken_before_normalization
    )
    results, pending = [], []
    created = reused = 0
    started = time.perf_counter()

    try:
        for index, token in enumerate(tokens):
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
                temporary = path.with_suffix(".saving")
                torch.save({
                    "protocol_sha256": protocol_hash,
                    "sample_token": token,
                    "result": result,
                }, temporary)
                temporary.replace(path)
                created += 1
                del prediction

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

report = {
    "status": "IMAGE_DARKENING_THREE_CONDITION_INFERENCE_PASS",
    "samples_per_condition": 600,
    "scenes": 150,
    "conditions": summaries,
    "scope": "Inference and formatting complete; accuracy pending.",
}
report_path = out / "three_condition_inference.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
copy_verified(report_path, destination / report_path.name)
print(json.dumps(report, indent=2), flush=True)
print("ALL THREE DARKENING CONDITIONS PRESERVED ON DRIVE", flush=True)
