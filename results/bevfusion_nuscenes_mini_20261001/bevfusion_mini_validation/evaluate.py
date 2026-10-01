import gc
import hashlib
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS, METRICS

root = Path("/content/datasets/nuscenes")
out = Path("/content/bevfusion_mini_validation")
ann = root / "mmdet3d_infos/nuscenes_infos_val.pkl"
config_dir = Path("/content/mmdetection3d/projects/BEVFusion/configs")

smoke = json.loads(
    Path("/content/bevfusion_mini_smoke/smoke_audit.json").read_text()
)
assert smoke["status"] == "REAL_SAMPLE_SMOKE_PASS"

receipt = json.loads(
    Path("/content/bevfusion_checkpoints/download_receipt.json").read_text()
)

configs = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}

# Locate paths independently of the download receipt's JSON layout.
checkpoint_dir = Path("/content/bevfusion_checkpoints")
checkpoint_names = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth",
}

all_tokens = {}
reports = {}

for name, config_name in configs.items():
    folder = out / name
    predictions = folder / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_dir / checkpoint_names[name]
    expected = next(
        record["checkpoint_sha256"]
        for record in smoke["models"] if record["model"] == name
    )
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == expected

    cfg = Config.fromfile(str(config_dir / config_name))
    cfg.test_dataloader.dataset.data_root = str(root) + "/"
    cfg.test_dataloader.dataset.ann_file = str(ann)
    cfg.test_dataloader.dataset.metainfo["version"] = "v1.0-mini"

    model = init_model(cfg, str(checkpoint), device="cuda:0")
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    assert len(dataset) == 81

    metric = METRICS.build(dict(
        type="NuScenesMetric",
        data_root=str(root),
        ann_file=str(ann),
        modality=dict(cfg.input_modality),
        metric="bbox",
        jsonfile_prefix=str(folder / "official_results"),
    ))
    metric.dataset_meta = dataset.metainfo
    assert metric.dataset_meta["version"] == "v1.0-mini"

    tokens = []
    for index in range(len(dataset)):
        token = dataset.get_data_info(index)["token"]
        tokens.append(token)
        batch = pseudo_collate([dataset[index]])

        with torch.inference_mode():
            result = model.test_step(batch)[0]

        pred = result.pred_instances_3d
        boxes = pred.bboxes_3d.tensor.detach().cpu()
        scores = pred.scores_3d.detach().cpu()
        labels = pred.labels_3d.detach().cpu()
        assert torch.isfinite(boxes).all()
        assert torch.isfinite(scores).all()

        (predictions / f"{index:03d}.json").write_text(json.dumps({
            "sample_token": token,
            "sample_idx": index,
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "labels": labels.tolist(),
        }) + "\n")

        metric.process(batch, [result.to_dict()])
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"{name}: {index + 1}/{len(dataset)}", flush=True)

        del batch, result, pred, boxes, scores, labels

    assert len(set(tokens)) == 81
    all_tokens[name] = tokens
    values = metric.evaluate(len(dataset))

    report = {
        "status": "MINI_VALIDATION_PASS",
        "model": name,
        "version": "v1.0-mini",
        "eval_split": "mini_val",
        "frames": len(tokens),
        "checkpoint_sha256": checkpoint_hash,
        "metadata_sha256": hashlib.sha256(ann.read_bytes()).hexdigest(),
        "source_commit": "fe25f7a51d36e3702f961e198894580d83c4387b",
        "metrics": {key: float(value) for key, value in values.items()},
        "scope": "Pretrained-model mini pilot; not full nuScenes validation.",
    }
    (folder / "evaluation_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    reports[name] = report

    print(json.dumps({
        "model": name,
        "headline_metrics": {
            key: float(value) for key, value in values.items()
            if key.endswith("/mAP") or key.endswith("/NDS")
        },
    }, indent=2), flush=True)

    del model, dataset, metric
    gc.collect()
    torch.cuda.empty_cache()

assert all_tokens["lidar"] == all_tokens["lidar_camera"]
(out / "paired_evaluation.json").write_text(json.dumps({
    "status": "PAIRED_MINI_VALIDATION_PASS",
    "same_ordered_sample_tokens": True,
    "sample_tokens": all_tokens["lidar"],
    "models": reports,
}, indent=2) + "\n")
print("PAIRED MINI VALIDATION PASS", flush=True)
