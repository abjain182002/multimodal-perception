
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import math
import sys

import nuscenes
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.utils.splits import create_splits_scenes

model = sys.argv[1]
output = Path(sys.argv[2]) / model
output.mkdir(parents=True, exist_ok=True)

work = Path("/content/full_validation_replay_20261020")
root = Path("/content/datasets/nuscenes_full")
base = Path(
    "/content/drive/MyDrive/Major_Project/nuscenes_full_validation"
)
original = work / "original_evidence"
reference = original / "evaluation" / model
prediction = base / "full_validation_merged" / model / "results_nusc.json"
tolerance = 1e-12

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

environment = json.loads((work / "environment_audit.json").read_text())
assert environment["status"] == "FULL_REPLAY_CPU_EVALUATOR_SOURCE_PASS"
for name, version in environment["packages"].items():
    assert metadata.version(name) == version, name

sdk = Path(nuscenes.__file__).parent
for relative, expected in environment["evaluator_source_sha256"].items():
    assert sha256(sdk / relative) == expected, relative

restoration = json.loads(
    (work / "metadata_restoration_audit.json").read_text()
)
print("VERIFYING RESTORED METADATA:", model, flush=True)
for name, record in restoration["metadata_files"].items():
    path = root / name
    assert path.stat().st_size == record["bytes"], name
    assert sha256(path) == record["sha256"], name

original_audit = json.loads(
    (reference / "evaluation_audit.json").read_text()
)
assert original_audit["samples"] == 6019
assert original_audit["scenes"] == 150
assert original_audit["evaluator_source_sha256"] == (
    environment["evaluator_source_sha256"]
)

for filename, key in (
    ("metrics_summary.json", "metrics_summary_sha256"),
    ("metrics_details.json", "metrics_details_sha256"),
):
    assert sha256(reference / filename) == original_audit[key], filename

print("VERIFYING SAVED PREDICTIONS:", model, flush=True)
assert sha256(prediction) == original_audit["predictions_sha256"]

merge = json.loads((original / "merge/merge_audit.json").read_text())
plan_path = original / "batch_plan.json"
assert sha256(plan_path) == merge["plan_sha256"]
plan = json.loads(plan_path.read_text())
ordered_tokens = [
    token for batch in plan["batches"]
    for token in batch["sample_tokens"]
]
assert len(ordered_tokens) == len(set(ordered_tokens)) == 6019
tokens = set(ordered_tokens)
del plan, ordered_tokens

cfg = config_factory("detection_cvpr_2019")
saved_config = json.loads(
    (original / "evaluation/detection_config.json").read_text()
)
assert cfg.serialize() == saved_config
(output / "detection_config.json").write_text(
    json.dumps(cfg.serialize(), indent=2) + "\n"
)

print("LOADING OFFICIAL METADATA:", model, flush=True)
nusc = NuScenes(
    version="v1.0-trainval", dataroot=str(root), verbose=True
)
official_names = set(create_splits_scenes()["val"])
official_scenes = {
    scene["token"] for scene in nusc.scene
    if scene["name"] in official_names
}
assert len(official_scenes) == 150
official_tokens = {
    sample["token"] for sample in nusc.sample
    if sample["scene_token"] in official_scenes
}
assert official_tokens == tokens

print("STARTING FULL CPU EVALUATION:", model, flush=True)
evaluator = NuScenesEval(
    nusc=nusc,
    config=cfg,
    result_path=str(prediction),
    eval_set="val",
    output_dir=str(output),
    verbose=True,
)
assert set(evaluator.pred_boxes.sample_tokens) == tokens
assert set(evaluator.gt_boxes.sample_tokens) == tokens

counts = dict(Counter(
    box.detection_name
    for token in evaluator.gt_boxes.sample_tokens
    for box in evaluator.gt_boxes[token]
))
assert counts == original_audit["filtered_gt_counts"]
print("OFFICIAL TOKENS AND FILTERED GT COUNTS PASS", flush=True)

# Curves are already preserved from the original evaluation.
# This replay recomputes and saves the underlying metric JSON files.
evaluator.main(plot_examples=0, render_curves=False)

differences = []
maximum_error = 0.0
numeric_values_checked = 0

def compare(actual, expected, path):
    global maximum_error, numeric_values_checked

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            differences.append(path + ": dictionary keys/type differ")
            return
        for key in expected:
            if key != "eval_time":
                compare(actual[key], expected[key], path + "." + key)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            differences.append(path + ": list length/type differs")
            return
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}[{index}]")
        return

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            differences.append(path + ": numeric type differs")
            return
        numeric_values_checked += 1
        if math.isnan(expected):
            if not math.isnan(actual):
                differences.append(path + ": NaN differs")
        elif not math.isfinite(expected):
            if actual != expected:
                differences.append(path + ": nonfinite value differs")
        elif not math.isfinite(actual):
            differences.append(path + ": replay value is nonfinite")
        else:
            error = abs(actual - expected)
            maximum_error = max(maximum_error, error)
            if error > tolerance:
                differences.append(f"{path}: absolute error {error}")
        return

    if actual != expected:
        differences.append(path + ": value differs")

for filename in ("metrics_summary.json", "metrics_details.json"):
    compare(
        json.loads((output / filename).read_text()),
        json.loads((reference / filename).read_text()),
        filename,
    )

summary = json.loads((output / "metrics_summary.json").read_text())
report = {
    "status": (
        "FULL_VALIDATION_SAME_EVALUATOR_REPLAY_PASS"
        if not differences else "FULL_VALIDATION_REPLAY_DIFFERENCES_FOUND"
    ),
    "model": model,
    "samples": 6019,
    "scenes": 150,
    "official_prediction_gt_token_match": True,
    "filtered_gt_counts": counts,
    "prediction_sha256": original_audit["predictions_sha256"],
    "evaluator_source_sha256": environment["evaluator_source_sha256"],
    "mAP": summary["mean_ap"],
    "NDS": summary["nd_score"],
    "comparison_tolerance": tolerance,
    "maximum_absolute_error": maximum_error,
    "numeric_values_checked": numeric_values_checked,
    "difference_count": len(differences),
    "first_differences": differences[:30],
    "excluded_from_comparison": ["eval_time"],
    "scope": (
        "Fresh ground-truth evaluation of the preserved full-validation "
        "predictions using the same evaluator. "
        "Not independent cross-implementation verification."
    ),
}
(output / "replay_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2), flush=True)
assert not differences, "Metric differences found; outputs preserved."
