import gc
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import mmengine
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS, METRICS

root = Path("/content/datasets/nuscenes_full")
frozen = root / "frozen_parts_01_02_subset"
out = Path("/content/bevfusion_subset_validation")
drive_out = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/bevfusion_subset_validation"
)
drive_out.mkdir(parents=True, exist_ok=True)

smoke = json.loads(
    Path("/content/bevfusion_subset_smoke/smoke_audit.json").read_text()
)
assert smoke["status"] == "PAIRED_SUBSET_SMOKE_PASS"

tokens = (frozen / "sample_tokens.txt").read_text().splitlines()
metadata = json.loads((frozen / "metadata_audit.json").read_text())
receipt = json.loads(
    Path("/content/bevfusion_checkpoints/download_receipt.json").read_text()
)
ann = Path(metadata["annotation_path"])
assert len(tokens) == len(set(tokens)) == 1231
assert hashlib.sha256(ann.read_bytes()).hexdigest() == metadata["annotation_sha256"]

configs = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}
config_dir = Path("/content/mmdetection3d/projects/BEVFusion/configs")

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    shutil.copyfile(source, temporary)
    assert sha256(temporary) == sha256(source)
    temporary.replace(target)

summaries = []

for name, filename in configs.items():
    random.seed(20261002)
    np.random.seed(20261002)
    torch.manual_seed(20261002)
    torch.cuda.manual_seed_all(20261002)

    model_out = out / name
    frames_dir = model_out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    drive_model = drive_out / name

    cfg = Config.fromfile(str(config_dir / filename))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset_cfg.data_root = str(root) + "/"
    dataset_cfg.ann_file = str(ann)
    dataset_cfg.metainfo["version"] = "v1.0-trainval"

    checkpoint = Path(receipt[name]["path"])
    assert sha256(checkpoint) == receipt[name]["sha256"]

    protocol = {
        "model": name,
        "checkpoint_sha256": receipt[name]["sha256"],
        "annotation_sha256": metadata["annotation_sha256"],
        "tokens_sha256": sha256(frozen / "sample_tokens.txt"),
        "resolved_config_sha256": hashlib.sha256(
            cfg.pretty_text.encode()
        ).hexdigest(),
        "source_commit": "fe25f7a51d36e3702f961e198894580d83c4387b",
        "seed": 20261002,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "scope": "Frozen parts 01–02 validation subset; 1231 samples.",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    protocol_path = model_out / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    cfg.dump(str(model_out / "resolved_config.py"))
    copy_verified(protocol_path, drive_model / protocol_path.name)

    model = init_model(cfg, str(checkpoint), device="cuda:0")
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model.eval()

    dataset = DATASETS.build(dataset_cfg)
    assert len(dataset) == 1231
    assert [
        dataset.get_data_info(i)["token"] for i in range(len(dataset))
    ] == tokens

    created = reused = 0
    results = []
    pending_backup = []
    started = time.perf_counter()

    for index, token in enumerate(tokens):
        path = frames_dir / f"{token}.pt"
        drive_path = drive_model / "frames" / path.name

        if not path.exists() and drive_path.exists():
            copy_verified(drive_path, path)

        if path.exists():
            saved = torch.load(path, map_location="cpu")
            assert saved["protocol_sha256"] == protocol_hash
            assert saved["sample_token"] == token
            assert saved["result"]["sample_idx"] == index
            result = saved["result"]
            reused += 1
        else:
            batch = pseudo_collate([dataset[index]])
            with torch.inference_mode():
                prediction = model.test_step(batch)[0]
            torch.cuda.synchronize()
            assert prediction.metainfo["sample_idx"] == index

            instances = prediction.pred_instances_3d.cpu()
            for field in list(instances.keys()):
                instances[field] = instances[field].to("cpu")
            assert instances.bboxes_3d.tensor.device.type == "cpu"
            assert instances.scores_3d.device.type == "cpu"
            assert torch.isfinite(instances.bboxes_3d.tensor).all()
            assert torch.isfinite(instances.scores_3d).all()
            assert len(instances.bboxes_3d) == len(instances.scores_3d)
            assert ((instances.scores_3d >= 0) & (instances.scores_3d <= 1)).all()
            assert ((instances.labels_3d >= 0) & (instances.labels_3d < 10)).all()

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
            del batch, prediction

        # Normalize restored predictions before NumPy-based formatting.
        cpu_instances = result["pred_instances_3d"]
        for field in list(cpu_instances.keys()):
            cpu_instances[field] = cpu_instances[field].to("cpu")
        assert cpu_instances.bboxes_3d.tensor.device.type == "cpu"
        results.append(result)
        pending_backup.append((path, drive_path))

        if (index + 1) % 100 == 0 or index + 1 == len(tokens):
            for source, target in pending_backup:
                copy_verified(source, target)
            pending_backup.clear()
            print(json.dumps({
                "model": name,
                "index": index + 1,
                "total": len(tokens),
                "created": created,
                "reused": reused,
                "backed_up": index + 1,
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            }), flush=True)

    # Use the pinned MMDetection3D formatter, without invoking full-val evaluation.
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
    formatted = json.loads(result_path.read_text())
    assert set(formatted["results"]) == set(tokens)

    summary = {
        "status": "SUBSET_INFERENCE_AND_FORMATTING_PASS",
        "model": name,
        "samples": len(tokens),
        "created": created,
        "reused": reused,
        "result_path": str(result_path),
        "results_sha256": sha256(result_path),
        "protocol_sha256": protocol_hash,
        "scope": "Inference complete; accuracy evaluation pending.",
    }
    audit_path = model_out / "inference_audit.json"
    audit_path.write_text(json.dumps(summary, indent=2) + "\n")
    copy_verified(result_path, drive_model / "results_nusc.json")
    copy_verified(audit_path, drive_model / audit_path.name)
    copy_verified(
        model_out / "resolved_config.py",
        drive_model / "resolved_config.py",
    )
    summaries.append(summary)
    print(json.dumps(summary, indent=2), flush=True)

    del model, dataset, formatter, results, result, cpu_instances, formatted
    gc.collect()
    torch.cuda.empty_cache()

report = {
    "status": "PAIRED_SUBSET_INFERENCE_PASS",
    "models": summaries,
    "interpretation": "No accuracy metrics measured yet.",
}
report_path = out / "paired_inference.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
copy_verified(report_path, drive_out / report_path.name)
copy_verified(Path(__file__), drive_out / "inference.py")
print(json.dumps(report, indent=2))
