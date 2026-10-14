import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

root = Path("/content/datasets/nuscenes_full")
local = root / "camera_availability_600"
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/camera_availability_600_20261014"
)
out = Path("/content/camera_availability_smoke")
configs = Path("/content/mmdetection3d/projects/BEVFusion/configs")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

protocol = json.loads((local / "protocol.json").read_text())
metadata = json.loads((local / "metadata_audit.json").read_text())
receipt = json.loads((study / "checkpoint_receipt.json").read_text())
tokens = (local / "sample_tokens.txt").read_text().splitlines()
assert metadata["status"] == "CAMERA_AVAILABILITY_METADATA_PASS"
assert sha256(metadata["annotation_path"]) == metadata["annotation_sha256"]
assert sha256(local / "protocol.json") == metadata["protocol_sha256"]
assert len(tokens) == 600

filenames = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}
audits = []
reference_points = None
reference_geometry = None

for model_name, filename in filenames.items():
    print("LOADING MODEL:", model_name, flush=True)
    set_random_seed(protocol["controls"]["seed"])
    cfg = Config.fromfile(str(configs / filename))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset_cfg.data_root = str(root) + "/"
    dataset_cfg.ann_file = metadata["annotation_path"]
    dataset_cfg.metainfo["version"] = "v1.0-trainval"

    checkpoint = Path(receipt[model_name]["path"])
    assert sha256(checkpoint) == receipt[model_name]["sha256"]
    model = init_model(cfg, str(checkpoint), device="cuda:0")
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
    model.eval()
    print("STRICT CHECKPOINT LOAD PASS:", model_name, flush=True)

    dataset = DATASETS.build(dataset_cfg)
    assert len(dataset) == 600
    assert [dataset.get_data_info(i)["token"] for i in range(600)] == tokens
    original = dataset[0]
    cfg.dump(str(out / f"{model_name}_resolved_config.py"))

    conditions = [
        c for c in protocol["conditions"] if c["model"] == model_name
    ]
    for condition in conditions:
        name = condition["name"]
        hook_record = {}

        def check_and_mask(module, args, processed):
            global reference_points, reference_geometry
            inputs = processed["inputs"]
            points_hashes = [tensor_hash(p) for p in inputs["points"]]
            if model_name == "lidar_camera":
                images = inputs["imgs"]
                assert images.ndim == 5 and images.shape[:2] == (1, 6)
                meta = processed["data_samples"][0].metainfo
                paths = meta["img_path"]
                assert len(paths) == 6

                channels = []
                for path in paths:
                    matches = [
                        part for part in Path(path).parts
                        if part.startswith("CAM_")
                    ]
                    assert len(matches) == 1, path
                    channels.append(matches[0])
                assert len(set(channels)) == 6
                assert set(channels) == {
                    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
                }

                geometry = repr({
                    key: meta[key] for key in (
                        "cam2img", "lidar2cam", "cam2lidar", "img_aug_matrix"
                    )
                })
                if reference_points is None:
                    reference_points = points_hashes
                    reference_geometry = geometry
                assert points_hashes == reference_points
                assert geometry == reference_geometry

                before = images.clone()
                masked = [
                    i for i, channel in enumerate(channels)
                    if channel not in condition["available_cameras"]
                ]
                images = images.clone()
                for index in masked:
                    images[:, index] = 0
                    assert torch.count_nonzero(images[:, index]).item() == 0
                for index in set(range(6)) - set(masked):
                    assert torch.equal(images[:, index], before[:, index])
                inputs["imgs"] = images

                hook_record.update({
                    "camera_order": channels,
                    "masked_channels": [channels[i] for i in masked],
                    "image_shape": list(images.shape),
                    "unmasked_views_unchanged": True,
                    "lidar_and_geometry_unchanged_across_fusion_conditions": True,
                })
            hook_record["points_sha256"] = points_hashes
            return processed

        handle = model.data_preprocessor.register_forward_hook(check_and_mask)
        try:
            with torch.inference_mode():
                predictions = model.test_step(
                    pseudo_collate([copy.deepcopy(original)])
                )
            torch.cuda.synchronize()
        finally:
            handle.remove()

        assert len(predictions) == 1 and hook_record
        instances = predictions[0].pred_instances_3d
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
        audit = {
            "condition": name,
            "sample_token": tokens[0],
            "strict_checkpoint_load": True,
            "finite_predictions": True,
            "detections": len(scores),
            "prediction_sha256": sha256(path),
            "input_check": hook_record,
        }
        audits.append(audit)
        print(json.dumps(audit, indent=2), flush=True)
        del predictions, instances

    del model, dataset, original
    gc.collect()
    torch.cuda.empty_cache()

report = {
    "status": "FOUR_CONDITION_MODEL_AND_MASK_SMOKE_PASS",
    "samples_in_frozen_subset": 600,
    "smoke_sample_token": tokens[0],
    "protocol_sha256": sha256(local / "protocol.json"),
    "conditions": audits,
    "scope": "One shared sample checked; subset accuracy pending.",
}
(out / "smoke_audit.json").write_text(json.dumps(report, indent=2) + "\n")
destination = study / "smoke"
for source in out.iterdir():
    if source.is_file() and source.suffix in (".py", ".json"):
        target = destination / source.name
        shutil.copy2(source, target)
        assert sha256(source) == sha256(target)

print(json.dumps(report, indent=2), flush=True)
print("SMOKE SCRIPT, CONFIGURATIONS AND PREDICTIONS VERIFIED ON DRIVE")
