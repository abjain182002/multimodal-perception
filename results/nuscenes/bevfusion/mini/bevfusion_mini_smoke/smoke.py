import gc
import hashlib
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

root = Path("/content/datasets/nuscenes")
out = Path("/content/bevfusion_mini_smoke")
configs = Path("/content/mmdetection3d/projects/BEVFusion/configs")
weights = Path("/content/bevfusion_checkpoints")

models = {
    "lidar": (
        "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
        "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth",
        "2628f933277774e194932815ea1a5d66a0bd9f69ebc8d9040e8c92760bb6652e",
    ),
    "lidar_camera": (
        "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
        "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth",
        "e68c6c084d8971f4f879751b0ed3981f575c24ee3cf385a1e39197c26a9bb3ab",
    ),
}

records = []
sample_tokens = []

for name, (config_name, checkpoint_name, expected_hash) in models.items():
    checkpoint = weights / checkpoint_name
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == expected_hash, name

    cfg = Config.fromfile(str(configs / config_name))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset_cfg.data_root = str(root) + "/"
    dataset_cfg.ann_file = str(
        root / "mmdet3d_infos/nuscenes_infos_val.pkl"
    )

    print(f"\nLOADING MODEL: {name}", flush=True)
    model = init_model(cfg, str(checkpoint), device="cuda:0")

    # Fail if the official checkpoint cannot fully load.
    load_checkpoint(
        model, str(checkpoint), map_location="cpu", strict=True
    )
    print(f"STRICT CHECKPOINT LOAD PASS: {name}", flush=True)

    dataset = DATASETS.build(dataset_cfg)
    assert len(dataset) == 81, len(dataset)
    info = dataset.get_data_info(0)
    token = info["token"]
    sample_tokens.append(token)

    batch = pseudo_collate([dataset[0]])
    with torch.inference_mode():
        prediction = model.test_step(batch)[0].pred_instances_3d
    torch.cuda.synchronize()

    boxes = prediction.bboxes_3d.tensor.detach().cpu()
    scores = prediction.scores_3d.detach().cpu()
    labels = prediction.labels_3d.detach().cpu()

    assert len(boxes) == len(scores) == len(labels)
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert ((labels >= 0) & (labels < 10)).all()

    prediction_file = out / f"{name}_prediction.json"
    prediction_file.write_text(json.dumps({
        "sample_token": token,
        "classes": list(cfg.class_names),
        "boxes": boxes.tolist(),
        "scores": scores.tolist(),
        "labels": labels.tolist(),
    }, indent=2) + "\n")

    record = {
        "model": name,
        "sample_token": token,
        "validation_samples": len(dataset),
        "strict_checkpoint_load": "PASS",
        "detections": len(scores),
        "checkpoint_sha256": digest,
        "prediction_sha256": hashlib.sha256(
            prediction_file.read_bytes()
        ).hexdigest(),
    }
    records.append(record)
    print(json.dumps(record, indent=2), flush=True)

    del model, dataset, batch, prediction, boxes, scores, labels
    gc.collect()
    torch.cuda.empty_cache()

assert len(set(sample_tokens)) == 1

report = {
    "status": "REAL_SAMPLE_SMOKE_PASS",
    "version": "v1.0-mini",
    "models": records,
    "scope": "One shared validation sample; accuracy not yet evaluated.",
}
(out / "smoke_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
