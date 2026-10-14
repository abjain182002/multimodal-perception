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
local = root / "camera_availability_600"
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/camera_availability_600_20261014"
)
out = Path("/content/camera_availability_inference")
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

protocol = json.loads((local / "protocol.json").read_text())
metadata = json.loads((local / "metadata_audit.json").read_text())
receipt = json.loads((study / "checkpoint_receipt.json").read_text())
smoke = json.loads((study / "smoke/smoke_audit.json").read_text())
tokens = (local / "sample_tokens.txt").read_text().splitlines()
ann = Path(metadata["annotation_path"])

assert smoke["status"] == "FOUR_CONDITION_MODEL_AND_MASK_SMOKE_PASS"
assert sha256(local / "protocol.json") == smoke["protocol_sha256"]
assert sha256(ann) == metadata["annotation_sha256"]
assert len(tokens) == len(set(tokens)) == 600

filenames = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}
summaries = []

for condition in protocol["conditions"]:
    name = condition["name"]
    model_name = condition["model"]
    print("\nSTARTING CONDITION:", name, flush=True)
    set_random_seed(protocol["controls"]["seed"])

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
        "seed": protocol["controls"]["seed"],
        "source_commit": protocol["controls"]["source_commit"],
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
    def mask_views(module, args, processed):
        if model_name == "lidar_camera":
            images = processed["inputs"]["imgs"]
            assert images.ndim == 5 and images.shape[:2] == (1, 6)
            paths = processed["data_samples"][0].metainfo["img_path"]
            channels = []
            for path in paths:
                matches = [
                    part for part in Path(path).parts
                    if part.startswith("CAM_")
                ]
                assert len(matches) == 1
                channels.append(matches[0])
            assert len(channels) == len(set(channels)) == 6
            assert set(channels) == {
                "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
            }
            masked = [
                i for i, channel in enumerate(channels)
                if channel not in condition["available_cameras"]
            ]
            if masked:
                changed = images.clone()
                for index in masked:
                    changed[:, index] = 0
                assert torch.count_nonzero(changed[:, masked]).item() == 0
                for index in set(range(6)) - set(masked):
                    assert torch.equal(changed[:, index], images[:, index])
                processed["inputs"]["imgs"] = changed
        hook_calls[0] += 1
        return processed

    handle = model.data_preprocessor.register_forward_hook(mask_views)
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
                with torch.inference_mode():
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
    "status": "FOUR_CONDITION_SUBSET_INFERENCE_PASS",
    "samples_per_condition": 600,
    "scenes": 150,
    "conditions": summaries,
    "scope": "Inference and formatting complete; accuracy pending.",
}
report_path = out / "four_condition_inference.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
copy_verified(report_path, destination / report_path.name)
print(json.dumps(report, indent=2), flush=True)
print("ALL FOUR CONDITIONS PRESERVED ON DRIVE", flush=True)
