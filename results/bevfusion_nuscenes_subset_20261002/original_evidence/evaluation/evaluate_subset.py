import gc
import hashlib
import inspect
import json
import shutil
from collections import Counter
from pathlib import Path

import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

root = Path("/content/datasets/nuscenes_full")
out = Path("/content/bevfusion_subset_validation")
backup = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/bevfusion_subset_validation"
)
frozen = root / "frozen_parts_01_02_subset"

tokens = set((frozen / "sample_tokens.txt").read_text().splitlines())
assert len(tokens) == 1231
split_name = "val_parts01_02_frozen"

inference = json.loads((backup / "paired_inference.json").read_text())
assert inference["status"] == "PAIRED_SUBSET_INFERENCE_PASS"

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha256(source) == sha256(target)

# Check both result files before loading the large metadata tables.
for model in inference["models"]:
    path = backup / model["model"] / "results_nusc.json"
    assert sha256(path) == model["results_sha256"]
    assert set(json.loads(path.read_text())["results"]) == tokens

print("PAIRED PREDICTION INPUTS PASS", flush=True)

nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=str(root),
    verbose=True,
)
config = config_factory("detection_cvpr_2019")
summaries = {}
source_hashes = {}

sdk_root = Path(nuscenes.__file__).parent
for relative in [
    "eval/detection/evaluate.py",
    "eval/detection/algo.py",
    "eval/common/loaders.py",
    "utils/splits.py",
]:
    source_hashes[relative] = sha256(sdk_root / relative)

for model in inference["models"]:
    name = model["model"]
    prediction_path = backup / name / "results_nusc.json"
    evaluation_dir = out / name / "subset_evaluation"

    print("EVALUATING:", name, flush=True)
    evaluator = DetectionEval(
        nusc=nusc,
        config=config,
        result_path=str(prediction_path),
        eval_set=split_name,
        output_dir=str(evaluation_dir),
        verbose=True,
    )

    assert set(evaluator.gt_boxes.sample_tokens) == tokens
    assert set(evaluator.pred_boxes.sample_tokens) == tokens
    print("EXACT PREDICTION / GT TOKEN MATCH PASS", flush=True)

    gt_counts = Counter(
        box.detection_name
        for token in evaluator.gt_boxes.sample_tokens
        for box in evaluator.gt_boxes[token]
    )

    # Official metric calculation and curve rendering.
    evaluator.main(plot_examples=0, render_curves=True)
    metrics_path = evaluation_dir / "metrics_summary.json"
    metrics = json.loads(metrics_path.read_text())

    assert 0 <= metrics["mean_ap"] <= 1
    assert 0 <= metrics["nd_score"] <= 1

    audit = {
        "status": "SUBSET_EVALUATION_PASS",
        "model": name,
        "samples": 1231,
        "scenes": 31,
        "eval_set": split_name,
        "prediction_gt_token_equality": True,
        "predictions_sha256": model["results_sha256"],
        "metrics_summary_sha256": sha256(metrics_path),
        "filtered_gt_counts": {
            cls: gt_counts[cls] for cls in config.class_names
        },
        "evaluator_source_sha256": source_hashes,
        "mAP": metrics["mean_ap"],
        "NDS": metrics["nd_score"],
        "scope": (
            "Availability-selected parts 01–02 validation subset; "
            "not full nuScenes validation."
        ),
    }
    (evaluation_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    for source in evaluation_dir.rglob("*"):
        if source.is_file():
            target = (
                backup / name / "subset_evaluation"
                / source.relative_to(evaluation_dir)
            )
            copy_verified(source, target)

    summaries[name] = audit
    print(json.dumps(audit, indent=2), flush=True)

    del evaluator, metrics
    gc.collect()

paired = {
    "status": "PAIRED_SUBSET_EVALUATION_PASS",
    "samples": 1231,
    "scenes": 31,
    "models": summaries,
    "camera_lidar_minus_lidar_points": {
        "mAP": 100 * (
            summaries["lidar_camera"]["mAP"] - summaries["lidar"]["mAP"]
        ),
        "NDS": 100 * (
            summaries["lidar_camera"]["NDS"] - summaries["lidar"]["NDS"]
        ),
    },
    "limitations": (
        "Availability-selected subset, not random and not full validation. "
        "Comparison of two pretrained pipelines, not a controlled camera ablation."
    ),
}

paired_path = out / "paired_evaluation.json"
paired_path.write_text(json.dumps(paired, indent=2) + "\n")
copy_verified(paired_path, backup / paired_path.name)
copy_verified(Path(__file__), backup / "evaluate_subset.py")

print(json.dumps(paired, indent=2))
print("METRICS AND CURVES VERIFIED ON DRIVE")
