
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import inspect
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
import nuscenes.eval.detection.evaluate as evaluator_module

local = Path("/content/camera_challenges600_evaluation")
study = local / "study"
out = local / "evaluation"
destination = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/camera_challenges600_20261016/evaluation"
)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text)
    temporary.replace(path)

def preserve_folder(source, target):
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in sorted(source.rglob("*")):
        if path.is_file():
            relative = path.relative_to(source)
            saved = target / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
            assert sha256(path) == sha256(saved), str(saved)
            records[relative.as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return records

assert importlib.metadata.version("nuscenes-devkit") == "1.2.0"
assert np.__version__ == "1.26.4"

metadata = json.loads(
    (local / "evaluation_setup/metadata_audit.json").read_text()
)
assert metadata["status"] == "CAMERA_CHALLENGES_CPU_EVALUATION_SETUP_PASS"

inference = json.loads(
    (study / "inference/seven_condition_inference.json").read_text()
)
assert inference["status"] == "KAGGLE_SEVEN_CONDITION_INFERENCE_PASS"
assert sha256(study / "protocol.json") == inference["protocol_sha256"]

tokens = (study / "sample_tokens.txt").read_text().splitlines()
assert len(tokens) == len(set(tokens)) == 600
token_set = set(tokens)

names = [
    "clean", "blur_sigma1", "blur_sigma2",
    "noise_std10", "noise_std25",
    "calibration_yaw1", "calibration_yaw3",
]
records = {record["condition"]: record for record in inference["conditions"]}
assert set(records) == set(names)

for name in names:
    path = study / "inference" / name / "results_nusc.json"
    assert sha256(path) == records[name]["results_sha256"], name
    with path.open() as stream:
        results = json.load(stream)
    assert set(results["results"]) == token_set, name
    del results

evaluator_path = Path(inspect.getfile(evaluator_module))
evaluator_hash = sha256(evaluator_path)
print("Official evaluator source SHA-256:", evaluator_hash, flush=True)

print("\nLOADING GROUND-TRUTH METADATA...", flush=True)
nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=metadata["data_root"],
    verbose=True,
)

validation_scenes = set(create_splits_scenes()["val"])
selected_scenes = {
    nusc.get("sample", token)["scene_token"] for token in tokens
}
assert len(selected_scenes) == 150
assert all(
    nusc.get("scene", scene)["name"] in validation_scenes
    for scene in selected_scenes
)

# Limit GT iteration to the frozen subset; retain full SDK table lookups.
class SelectedSampleView:
    def __init__(self, sdk, selected_tokens):
        self._sdk = sdk
        self.sample = [sdk.get("sample", token) for token in selected_tokens]

    def __getattr__(self, name):
        return getattr(self._sdk, name)

view = SelectedSampleView(nusc, tokens)
original_load_gt = evaluator_module.load_gt

def selected_load_gt(sdk, eval_split, box_cls, verbose=False):
    assert eval_split == "val"
    assert box_cls is DetectionBox
    return original_load_gt(view, eval_split, box_cls, verbose=verbose)

evaluator_module.load_gt = selected_load_gt
config = config_factory("detection_cvpr_2019")

expected_gt_counts = {
    "car": 5327,
    "truck": 922,
    "bus": 208,
    "trailer": 237,
    "construction_vehicle": 155,
    "pedestrian": 2427,
    "motorcycle": 184,
    "bicycle": 195,
    "traffic_cone": 1059,
    "barrier": 1604,
}
assert sum(expected_gt_counts.values()) == 12318

all_reports = []
clean_metrics = None

try:
    for name in names:
        print("\nEVALUATING CONDITION:", name, flush=True)
        condition_folder = out / name
        condition_folder.mkdir(exist_ok=True)

        prediction_path = study / "inference" / name / "results_nusc.json"
        evaluator = evaluator_module.DetectionEval(
            nusc=nusc,
            config=config,
            result_path=str(prediction_path),
            eval_set="val",
            output_dir=str(condition_folder),
            verbose=True,
        )

        assert set(evaluator.gt_boxes.sample_tokens) == token_set
        assert set(evaluator.pred_boxes.sample_tokens) == token_set

        counts = {key: 0 for key in expected_gt_counts}
        for box in evaluator.gt_boxes.all:
            counts[box.detection_name] += 1
        assert counts == expected_gt_counts, (name, counts)

        evaluator.main(plot_examples=0, render_curves=True)

        metric_path = condition_folder / "metrics_summary.json"
        metrics = json.loads(metric_path.read_text())

        class_means = metrics["mean_dist_aps"]
        recomputed_map = float(np.mean(list(class_means.values())))
        assert abs(recomputed_map - metrics["mean_ap"]) < 1e-12

        recomputed_nds = (
            5 * metrics["mean_ap"] +
            sum(
                max(0.0, 1.0 - float(error))
                for error in metrics["tp_errors"].values()
            )
        ) / 10.0
        assert abs(recomputed_nds - metrics["nd_score"]) < 1e-12

        if name == "clean":
            clean_metrics = metrics

        report = {
            "status": "CAMERA_CHALLENGE_CONDITION_EVALUATION_PASS",
            "condition": name,
            "samples": 600,
            "scenes": 150,
            "filtered_gt_objects": 12318,
            "filtered_gt_counts": counts,
            "exact_prediction_gt_token_match": True,
            "protocol_sha256": inference["protocol_sha256"],
            "prediction_sha256": records[name]["results_sha256"],
            "evaluator_source_sha256": evaluator_hash,
            "nuscenes_devkit": "1.2.0",
            "eval_config": "detection_cvpr_2019",
            "mean_ap": metrics["mean_ap"],
            "nd_score": metrics["nd_score"],
            "delta_map_points_vs_clean": (
                metrics["mean_ap"] - clean_metrics["mean_ap"]
            ) * 100,
            "delta_nds_points_vs_clean": (
                metrics["nd_score"] - clean_metrics["nd_score"]
            ) * 100,
            "class_ap": class_means,
            "tp_errors": metrics["tp_errors"],
            "metrics_summary_sha256": sha256(metric_path),
            "metrics_details_sha256": sha256(
                condition_folder / "metrics_details.json"
            ),
            "scope": (
                "nuScenes detection evaluation on the frozen 600-frame "
                "subset. Synthetic conditions; not full-validation "
                "or physical-system robustness."
            ),
        }
        write_json(condition_folder / "evaluation_audit.json", report)
        preserved = preserve_folder(
            condition_folder, destination / name
        )
        report["verified_drive_files"] = len(preserved)
        all_reports.append(report)

        write_json(out / "completed_conditions.json", all_reports)
        shutil.copy2(
            out / "completed_conditions.json",
            destination / "completed_conditions.json",
        )
        assert sha256(out / "completed_conditions.json") == sha256(
            destination / "completed_conditions.json"
        )

        print(json.dumps({
            "condition": name,
            "mAP_x100": metrics["mean_ap"] * 100,
            "NDS_x100": metrics["nd_score"] * 100,
            "delta_mAP_points": report["delta_map_points_vs_clean"],
            "delta_NDS_points": report["delta_nds_points_vs_clean"],
            "drive_metrics_verified": True,
        }, indent=2), flush=True)

        del evaluator, metrics
        plt.close("all")
        gc.collect()

finally:
    evaluator_module.load_gt = original_load_gt

summary = {
    "status": "SEVEN_CONDITION_SUBSET_EVALUATION_AND_DRIVE_COPY_PASS",
    "samples_per_condition": 600,
    "scenes": 150,
    "conditions": all_reports,
    "protocol_sha256": inference["protocol_sha256"],
    "evaluator_source_sha256": evaluator_hash,
    "scope": (
        "Seven-condition subset accuracy evaluated and metric files "
        "verified on Drive. Compact final evidence archive, figures "
        "and GitHub import remain pending."
    ),
}
write_json(out / "seven_condition_evaluation.json", summary)
shutil.copy2(
    out / "seven_condition_evaluation.json",
    destination / "seven_condition_evaluation.json",
)
assert sha256(out / "seven_condition_evaluation.json") == sha256(
    destination / "seven_condition_evaluation.json"
)

print("\nSEVEN_CONDITION_SUBSET_EVALUATION_AND_DRIVE_COPY_PASS", flush=True)
