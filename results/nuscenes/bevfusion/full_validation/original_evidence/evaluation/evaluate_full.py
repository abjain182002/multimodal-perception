from pathlib import Path
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil

import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval

root = Path("/content/datasets/nuscenes_full")
merged = Path("/content/bevfusion_full_validation/merged")
out = Path("/content/bevfusion_full_validation/full_evaluation")
base = Path(
    "/content/drive/MyDrive/Major_Project/nuscenes_full_validation"
)
drive_out = base / "full_validation_evaluation_20261010"
out.mkdir(parents=True, exist_ok=True)
drive_out.mkdir(parents=True, exist_ok=True)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".uploading")
    shutil.copy2(source, temporary)
    assert sha256(source) == sha256(temporary)
    temporary.replace(target)
    assert sha256(source) == sha256(target)

assert importlib.metadata.version("nuscenes-devkit") == "1.2.0"

# Confirm the same evaluator source files used in the earlier replay.
sdk = Path(nuscenes.__file__).parent
expected_sources = {
    "eval/detection/evaluate.py":
        "1547188122d60ce0bc8600cb27861b10f624e3d38dbfb08e5569ffebb6cbba6e",
    "eval/detection/algo.py":
        "460e1691c1391e01fe7c6f4576add6054b29cc280687513a9f9737b7363f4f38",
    "eval/common/loaders.py":
        "9853cb47dd898c85e2032507e6bb419460f173e6b9a3e069675efe0839e72fe7",
    "utils/splits.py":
        "eab6fa5e2536a2a85bd9451fb35771833e262b4b96319a6b26fee1dce8f4e2cd",
}
for relative, expected in expected_sources.items():
    assert sha256(sdk / relative) == expected, relative
print("PINNED EVALUATOR SOURCE HASHES PASS", flush=True)

merge_audit = json.loads((merged / "merge_audit.json").read_text())
assert merge_audit["status"] == "PAIRED_FULL_VALIDATION_MERGE_PASS"
assert merge_audit["samples"] == 6019
assert merge_audit["scenes"] == 150

plan_path = root / "full_validation_batch_plan.json"
assert sha256(plan_path) == merge_audit["plan_sha256"]
plan = json.loads(plan_path.read_text())
tokens = {
    token for batch in plan["batches"]
    for token in batch["sample_tokens"]
}
assert len(tokens) == 6019

environment = {
    "python": platform.python_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in (
            "nuscenes-devkit", "numpy", "scipy",
            "matplotlib", "scikit-learn", "shapely"
        )
    },
    "evaluator_source_sha256": expected_sources,
}
environment_path = out / "environment_audit.json"
environment_path.write_text(json.dumps(environment, indent=2) + "\n")
copy_verified(environment_path, drive_out / environment_path.name)

cfg = config_factory("detection_cvpr_2019")
config_path = out / "detection_config.json"
config_path.write_text(json.dumps(cfg.serialize(), indent=2) + "\n")
copy_verified(config_path, drive_out / config_path.name)

nusc = NuScenes(
    version="v1.0-trainval", dataroot=str(root), verbose=True
)
records = {}
reference_counts = None

for merged_record in merge_audit["models"]:
    model = merged_record["model"]
    prediction = Path(merged_record["result_path"])
    assert sha256(prediction) == merged_record["results_sha256"]

    model_out = out / model
    print("\nEVALUATING FULL VALIDATION:", model, flush=True)
    evaluator = NuScenesEval(
        nusc=nusc,
        config=cfg,
        result_path=str(prediction),
        eval_set="val",
        output_dir=str(model_out),
        verbose=True,
    )

    assert set(evaluator.pred_boxes.sample_tokens) == tokens
    assert set(evaluator.gt_boxes.sample_tokens) == tokens
    print("EXACT FULL PREDICTION / GT TOKEN MATCH PASS", flush=True)

    counts = dict(Counter(
        box.detection_name
        for token in evaluator.gt_boxes.sample_tokens
        for box in evaluator.gt_boxes[token]
    ))
    if reference_counts is None:
        reference_counts = counts
    else:
        assert counts == reference_counts

    evaluator.main(plot_examples=0, render_curves=True)

    summary_path = model_out / "metrics_summary.json"
    details_path = model_out / "metrics_details.json"
    metrics = json.loads(summary_path.read_text())
    assert math.isfinite(metrics["mean_ap"])
    assert math.isfinite(metrics["nd_score"])
    assert 0 <= metrics["mean_ap"] <= 1
    assert 0 <= metrics["nd_score"] <= 1

    audit = {
        "status": "FULL_VALIDATION_EVALUATION_PASS",
        "model": model,
        "samples": 6019,
        "scenes": 150,
        "eval_set": "val",
        "prediction_gt_token_equality": True,
        "predictions_sha256": sha256(prediction),
        "metrics_summary_sha256": sha256(summary_path),
        "metrics_details_sha256": sha256(details_path),
        "filtered_gt_counts": counts,
        "mAP": metrics["mean_ap"],
        "NDS": metrics["nd_score"],
        "per_class_AP": metrics["mean_dist_aps"],
        "tp_errors": metrics["tp_errors"],
        "evaluator_source_sha256": expected_sources,
        "scope": (
            "Official full nuScenes validation split; "
            "two pretrained pipelines, not a controlled camera ablation."
        ),
    }
    (model_out / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    # Preserve metrics, audit and all evaluator-generated curves.
    for source in sorted(model_out.rglob("*")):
        if source.is_file():
            copy_verified(
                source, drive_out / model / source.relative_to(model_out)
            )

    records[model] = audit
    print(json.dumps(audit, indent=2), flush=True)
    print("METRICS AND CURVES VERIFIED ON DRIVE:", model, flush=True)
    del evaluator, metrics
    gc.collect()

assert set(records) == {"lidar", "lidar_camera"}
report = {
    "status": "PAIRED_FULL_VALIDATION_EVALUATION_PASS",
    "samples": 6019,
    "scenes": 150,
    "models": records,
    "camera_lidar_minus_lidar_points": {
        metric: 100 * (
            records["lidar_camera"][metric] - records["lidar"][metric]
        ) for metric in ("mAP", "NDS")
    },
    "limitations": (
        "Comparison of two official pretrained pipelines; "
        "not a controlled camera ablation or independent training reproduction."
    ),
}
report_path = out / "paired_evaluation.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
copy_verified(report_path, drive_out / report_path.name)
copy_verified(Path(__file__), drive_out / "evaluate_full.py")
print(json.dumps(report, indent=2), flush=True)
print("FULL VALIDATION METRICS AND CURVES PRESERVED", flush=True)
