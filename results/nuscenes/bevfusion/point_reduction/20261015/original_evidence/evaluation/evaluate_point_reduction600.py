import gc
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import nuscenes.eval.detection.evaluate as evaluator_module
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory

root = Path("/content/datasets/nuscenes_full")
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/lidar_point_reduction600_20261015"
)
out = Path("/content/lidar_point_reduction600/evaluation")
saved = study / "evaluation"

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def backup_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".uploading")
    shutil.copy2(source, temporary)
    assert sha256(source) == sha256(temporary)
    temporary.replace(target)

tokens = (study / "sample_tokens.txt").read_text().splitlines()
token_set = set(tokens)
assert len(tokens) == len(token_set) == 600
protocol = json.loads((study / "protocol.json").read_text())
inference = json.loads(
    (study / "inference/six_condition_inference.json").read_text()
)
assert inference["status"] == "PAIRED_POINT_REDUCTION_SIX_CONDITION_INFERENCE_PASS"
assert {r["condition"] for r in inference["conditions"]} == {
    c["name"] for c in protocol["conditions"]
}

for record in inference["conditions"]:
    folder = study / "inference" / record["condition"]
    prediction = folder / "results_nusc.json"
    assert sha256(prediction) == record["results_sha256"]
    assert sha256(folder / "protocol.json") == record["protocol_sha256"]
    data = json.loads(prediction.read_text())
    assert set(data["results"]) == token_set
    del data
    print("PREDICTION HASH AND TOKENS PASS:",
          record["condition"], flush=True)


import importlib.metadata
assert importlib.metadata.version("nuscenes-devkit") == "1.2.0"
assert sha256(study / "protocol.json") == (
    "f6037b05188e2eb8c0b51ef6c3ac76d99085831866f1ad034b348277cb2f43f1"
)
condition_specs = {c["name"]: c for c in protocol["conditions"]}
input_records = {}

for record in inference["conditions"]:
    name = record["condition"]
    path = study / "inference" / name / "input_checks.json"
    assert sha256(path) == record["input_checks_sha256"]
    audit = json.loads(path.read_text())
    assert audit["protocol_sha256"] == record["protocol_sha256"]
    rows = audit["records"]
    assert [r["sample_token"] for r in rows] == tokens
    fraction = condition_specs[name]["retained_fraction"]
    assert audit["retained_fraction"] == fraction
    for row in rows:
        assert row["original_points"] > 0
        assert row["retained_points"] == max(
            1, int(fraction * row["original_points"])
        )
    input_records[name] = rows

for index, token in enumerate(tokens):
    rows = {
        name: records[index]
        for name, records in input_records.items()
    }
    assert len({
        (r["original_points"], r["original_points_sha256"])
        for r in rows.values()
    }) == 1, ("Original point mismatch", token)

    for suffix in ("full", "keep050", "keep025"):
        a, b = rows["lidar_" + suffix], rows["lidar_camera_" + suffix]
        for key in (
            "retained_points", "selected_points_sha256",
            "selected_indices_sha256"
        ):
            assert a[key] == b[key], (token, suffix, key)

    for model in ("lidar", "lidar_camera"):
        model_rows = [rows[model + "_" + s]
                      for s in ("full", "keep050", "keep025")]
        assert len({
            (r["images_sha256"], r["metadata_sha256"])
            for r in model_rows
        }) == 1, ("Image/metadata mismatch", token)

del input_records
print("ALL 600 PAIRED INPUT RECORDS VERIFIED", flush=True)

nusc = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=True)
sample_by_token = {row["token"]: row for row in nusc.sample}
assert token_set <= set(sample_by_token)
assert len({sample_by_token[t]["scene_token"] for t in tokens}) == 150

# Restrict only the sample iteration used by the official GT loader.
# Full SDK lookup tables remain available through the wrapped object.
class SelectedSampleView:
    def __init__(self, sdk):
        self.sdk = sdk
        self.sample = [sample_by_token[t] for t in tokens]

    def __getattr__(self, name):
        return getattr(self.sdk, name)

original_load_gt = evaluator_module.load_gt
def load_frozen_gt(sdk, eval_split, box_cls, verbose=False):
    assert eval_split == "val"
    gt = original_load_gt(
        SelectedSampleView(sdk), eval_split, box_cls, verbose=verbose
    )
    assert set(gt.sample_tokens) == token_set
    return gt

evaluator_module.load_gt = load_frozen_gt
config = config_factory("detection_cvpr_2019")
(out / "detection_config.json").write_text(
    json.dumps(config.serialize(), indent=2) + "\n"
)

summaries = {}
shared_gt_counts = None

for record in inference["conditions"]:
    name = record["condition"]
    print("\nEVALUATING:", name, flush=True)
    prediction = study / "inference" / name / "results_nusc.json"
    folder = out / name
    folder.mkdir(exist_ok=True)

    evaluator = evaluator_module.DetectionEval(
        nusc,
        config=config,
        result_path=str(prediction),
        eval_set="val",
        output_dir=str(folder),
        verbose=True,
    )
    assert set(evaluator.pred_boxes.sample_tokens) == token_set
    assert set(evaluator.gt_boxes.sample_tokens) == token_set

    counts = {label: 0 for label in config.class_names}
    for token in evaluator.gt_boxes.sample_tokens:
        for box in evaluator.gt_boxes[token]:
            counts[box.detection_name] += 1
    if shared_gt_counts is None:
        shared_gt_counts = counts
    assert counts == shared_gt_counts
    assert counts == {
        "car": 5327, "truck": 922, "bus": 208, "trailer": 237,
        "construction_vehicle": 155, "pedestrian": 2427,
        "motorcycle": 184, "bicycle": 195,
        "traffic_cone": 1059, "barrier": 1604,
    }, ("Unexpected filtered ground truth", counts)
    print("EXACT PAIRED PREDICTION / GT TOKEN MATCH PASS", flush=True)

    evaluator.main(plot_examples=0, render_curves=True)
    metrics = json.loads((folder / "metrics_summary.json").read_text())
    audit = {
        "status": "CONDITION_SUBSET_EVALUATION_PASS",
        "condition": name,
        "samples": 600,
        "scenes": 150,
        "prediction_gt_token_equality": True,
        "filtered_gt_counts": counts,
        "predictions_sha256": record["results_sha256"],
        "metrics_summary_sha256": sha256(folder / "metrics_summary.json"),
        "mAP": metrics["mean_ap"],
        "NDS": metrics["nd_score"],
        "scope": "Four selected frames per validation scene; subset evaluation.",
    }
    (folder / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    for source in folder.rglob("*"):
        if source.is_file():
            backup_file(source, saved / name / source.relative_to(folder))

    summaries[name] = audit
    print(json.dumps(audit, indent=2), flush=True)
    print("METRICS AND CURVES VERIFIED ON DRIVE:", name, flush=True)
    del evaluator, metrics
    gc.collect()


own_baseline = {}
for condition in protocol["conditions"]:
    name = condition["name"]
    reference = summaries[condition["model"] + "_full"]
    row = summaries[name]
    own_baseline[name] = {
        "mAP": 100 * (row["mAP"] - reference["mAP"]),
        "NDS": 100 * (row["NDS"] - reference["NDS"]),
    }

fusion_difference = {}
for suffix in ("full", "keep050", "keep025"):
    lidar = summaries["lidar_" + suffix]
    fusion = summaries["lidar_camera_" + suffix]
    fusion_difference[suffix] = {
        "mAP": 100 * (fusion["mAP"] - lidar["mAP"]),
        "NDS": 100 * (fusion["NDS"] - lidar["NDS"]),
    }

report = {
    "status": "PAIRED_POINT_REDUCTION_SIX_CONDITION_EVALUATION_PASS",
    "samples_per_condition": 600,
    "scenes": 150,
    "protocol_sha256": sha256(study / "protocol.json"),
    "paired_input_records_verified": True,
    "conditions": summaries,
    "difference_from_own_full_baseline_points": own_baseline,
    "fusion_minus_lidar_at_same_retention_points": fusion_difference,
    "official_evaluator_source_sha256": sha256(
        inspect.getfile(evaluator_module)
    ),
    "limitations": protocol["limitations"],
}
report_path = out / "six_condition_evaluation.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
for source in (
    report_path, out / "detection_config.json", Path(__file__)
):
    backup_file(source, saved / source.name)

print(json.dumps(report, indent=2), flush=True)
print("ALL SIX POINT-REDUCTION EVALUATIONS AND DRIVE BACKUPS VERIFIED", flush=True)
