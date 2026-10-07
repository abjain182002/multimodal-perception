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

root = Path("/content/datasets/nuscenes_full")
frozen = root / "frozen_parts_01_02_subset"
out = Path("/content/bevfusion_subset_smoke")
config_dir = Path("/content/mmdetection3d/projects/BEVFusion/configs")
checkpoint_dir = Path("/content/bevfusion_checkpoints")

tokens = (frozen / "sample_tokens.txt").read_text().splitlines()
metadata = json.loads((frozen / "metadata_audit.json").read_text())
receipt = json.loads((checkpoint_dir / "download_receipt.json").read_text())

assert metadata["status"] == "SUBSET_METADATA_PASS"
assert len(tokens) == 1231
assert torch.cuda.is_available()

configs = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}

audits = []
for name, filename in configs.items():
    print("LOADING MODEL:", name, flush=True)

    cfg = Config.fromfile(str(config_dir / filename))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset_cfg.data_root = str(root) + "/"
    dataset_cfg.ann_file = metadata["annotation_path"]
    dataset_cfg.metainfo["version"] = "v1.0-trainval"

    checkpoint = Path(receipt[name]["path"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        receipt[name]["sha256"]
    )

    model = init_model(cfg, str(checkpoint), device="cuda:0")
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model.eval()

    dataset = DATASETS.build(dataset_cfg)
    assert len(dataset) == 1231
    dataset_tokens = [
        dataset.get_data_info(i)["token"] for i in range(len(dataset))
    ]
    assert dataset_tokens == tokens, "Dataset token order mismatch"
    print("ORDERED DATASET TOKENS PASS", flush=True)

    batch = pseudo_collate([dataset[0]])
    with torch.inference_mode():
        predictions = model.test_step(batch)
    torch.cuda.synchronize()

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.metainfo["sample_idx"] == 0, prediction.metainfo
    assert dataset.get_data_info(0)["token"] == tokens[0]

    instances = prediction.pred_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu()
    scores = instances.scores_3d.detach().cpu()
    labels = instances.labels_3d.detach().cpu()

    assert len(boxes) == len(scores) == len(labels)
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(scores).all()
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

    audit = {
        "model": name,
        "sample_token": tokens[0],
        "validation_samples": len(dataset),
        "strict_checkpoint_load": "PASS",
        "finite_predictions": "PASS",
        "detections": len(scores),
        "checkpoint_sha256": receipt[name]["sha256"],
        "prediction_sha256": hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest(),
    }
    audits.append(audit)
    print(json.dumps(audit, indent=2), flush=True)

    del model, dataset, batch, predictions, prediction, instances
    gc.collect()
    torch.cuda.empty_cache()

report = {
    "status": "PAIRED_SUBSET_SMOKE_PASS",
    "samples": 1231,
    "scenes": 31,
    "models": audits,
    "scope": "One shared frozen-subset sample; accuracy not yet evaluated.",
}
(out / "smoke_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
