
import gc
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import nuscenes
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

root = Path("/content/datasets/nuscenes_full")
base = Path(
    "/content/drive/MyDrive/Major_Project/nuscenes_full_validation"
)
original = base / "bevfusion_subset_validation"
frozen = base / "frozen_parts_01_02_subset"
out = Path("/content/nuscenes_subset_replay")
backup = base / "evaluation_replay_20261008"
split_name = "val_parts01_02_frozen"
tolerance = 1e-12

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha256(source) == sha256(target), target

def compare_values(actual, expected, path="root"):
    """Compare stored values; evaluation wall-clock time is excluded."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        errors = [
            compare_values(actual[key], value, path + "." + key)
            for key, value in expected.items()
            if key != "eval_time"
        ]
        return max(errors, default=0.0)

    if isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), path
        return max(
            (compare_values(a, b, f"{path}[{i}]")
             for i, (a, b) in enumerate(zip(actual, expected))),
            default=0.0,
        )

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert isinstance(actual, (int, float)), path
        if math.isnan(expected):
            assert math.isnan(actual), path
            return 0.0
        if not math.isfinite(expected):
            assert actual == expected, path
            return 0.0
        error = abs(actual - expected)
        assert error <= tolerance, f"{path}: error={error}"
        return error

    assert actual == expected, path
    return 0.0

tokens = set((frozen / "sample_tokens.txt").read_text().splitlines())
assert len(tokens) == 1231

environment = json.loads((out / "environment_audit.json").read_text())
assert environment["status"] == "CPU_EVALUATOR_SOURCE_PASS"

sdk = Path(nuscenes.__file__).parent
for relative, expected in environment["evaluator_source_sha256"].items():
    assert sha256(sdk / relative) == expected, relative

split_path = root / "v1.0-trainval/splits.json"
selected_scenes = set(json.loads(split_path.read_text())[split_name])
assert len(selected_scenes) == 31
assert selected_scenes <= set(create_splits_scenes()["val"])

# Verify predictions and original summaries before evaluation.
audits = {}
for name in ["lidar", "lidar_camera"]:
    folder = original / name
    audit = json.loads(
        (folder / "subset_evaluation/evaluation_audit.json").read_text()
    )
    assert audit["evaluator_source_sha256"] == environment["evaluator_source_sha256"]
    assert sha256(folder / "results_nusc.json") == audit["predictions_sha256"]
    assert sha256(folder / "subset_evaluation/metrics_summary.json") == \
        audit["metrics_summary_sha256"]

    predictions = json.loads((folder / "results_nusc.json").read_text())
    assert set(predictions["results"]) == tokens
    audits[name] = audit
    del predictions

print("PAIRED REPLAY INPUTS PASS", flush=True)

nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=str(root),
    verbose=True,
)

selected_scene_tokens = {
    scene["token"] for scene in nusc.scene
    if scene["name"] in selected_scenes
}
metadata_tokens = {
    sample["token"] for sample in nusc.sample
    if sample["scene_token"] in selected_scene_tokens
}
assert metadata_tokens == tokens
print("EXACT FROZEN SCENE / METADATA TOKEN MATCH PASS", flush=True)

results = {}
for name in ["lidar", "lidar_camera"]:
    print("REPLAYING:", name, flush=True)
    folder = out / name

    evaluator = DetectionEval(
        nusc=nusc,
        config=config_factory("detection_cvpr_2019"),
        result_path=str(original / name / "results_nusc.json"),
        eval_set=split_name,
        output_dir=str(folder),
        verbose=True,
    )

    assert set(evaluator.pred_boxes.sample_tokens) == tokens
    assert set(evaluator.gt_boxes.sample_tokens) == tokens

    counts = dict(Counter(
        box.detection_name
        for token in evaluator.gt_boxes.sample_tokens
        for box in evaluator.gt_boxes[token]
    ))
    assert counts == audits[name]["filtered_gt_counts"]

    evaluator.main(plot_examples=0, render_curves=True)

    summary = json.loads((folder / "metrics_summary.json").read_text())
    recorded = original / name / "subset_evaluation"

    errors = {}
    for filename in ["metrics_summary.json", "metrics_details.json"]:
        errors[filename] = compare_values(
            json.loads((folder / filename).read_text()),
            json.loads((recorded / filename).read_text()),
            filename,
        )

    results[name] = {
        "status": "GROUND_TRUTH_REPLAY_PASS",
        "samples": len(tokens),
        "scenes": len(selected_scenes),
        "prediction_gt_token_equality": True,
        "filtered_gt_counts": counts,
        "mAP": summary["mean_ap"],
        "NDS": summary["nd_score"],
        "maximum_absolute_errors": errors,
        "comparison_tolerance": tolerance,
        "excluded_from_comparison": ["eval_time"],
        "predictions_sha256": audits[name]["predictions_sha256"],
    }

    (folder / "replay_audit.json").write_text(
        json.dumps(results[name], indent=2) + "\n"
    )
    print(json.dumps(results[name], indent=2), flush=True)

    del evaluator, summary
    gc.collect()

report = {
    "status": "PAIRED_GROUND_TRUTH_REPLAY_PASS",
    "models": results,
    "evaluator_source_sha256": environment["evaluator_source_sha256"],
    "custom_split_sha256": sha256(split_path),
    "frozen_tokens_sha256": sha256(frozen / "sample_tokens.txt"),
    "scope": (
        "Recomputed accuracy on the same availability-selected "
        "1,231 samples / 31 scenes. Not full nuScenes validation "
        "or an independent evaluator cross-check."
    ),
}
(out / "paired_replay_audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

# Preserve and hash-check every replay artifact on Drive.
manifest = {}
for source in sorted(out.rglob("*")):
    if source.is_file():
        relative = source.relative_to(out)
        copy_verified(source, backup / relative)
        manifest[str(relative)] = {
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }

receipt = {
    "status": "REPLAY_DRIVE_BACKUP_VERIFIED",
    "files": manifest,
}
receipt_path = out / "backup_receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
copy_verified(receipt_path, backup / receipt_path.name)

print("PAIRED GROUND-TRUTH REPLAY PASS", flush=True)
print("REPLAY DRIVE BACKUP VERIFIED", flush=True)
print("Drive folder:", backup, flush=True)
