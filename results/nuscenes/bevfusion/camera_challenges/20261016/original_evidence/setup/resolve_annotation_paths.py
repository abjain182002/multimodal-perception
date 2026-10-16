
from pathlib import Path, PurePosixPath
import copy
import hashlib
import json
import pickle
import numpy as np

study = Path("/kaggle/working/camera_challenges600")
audit = json.loads((study / "annotation_input_audit.json").read_text())
assert audit["status"] == "KAGGLE_ORIGINAL_ANNOTATION_AND_TOKENS_PASS"

source = Path(audit["annotation_path"])
root = Path(audit["sensor_root"])
required = set(
    (study / "required_sensor_files.txt").read_text().splitlines()
)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert sha256(source) == audit["annotation_sha256"]

with source.open("rb") as stream:
    original = pickle.load(stream)
updated = copy.deepcopy(original)

changes = []
referenced = set()
old_root = "/content/datasets/nuscenes_full/"

def resolve(value, default_prefix):
    value = str(value)

    if value.startswith(old_root):
        relative = value[len(old_root):]
    elif value.startswith(str(root) + "/"):
        relative = value[len(str(root)) + 1:]
    elif value.startswith(("samples/", "sweeps/")):
        relative = value
    else:
        assert "/" not in value, (
            "Unrecognised sensor path; stopping rather than guessing",
            value,
        )
        relative = default_prefix + "/" + value

    path = PurePosixPath(relative)
    assert not path.is_absolute() and ".." not in path.parts
    assert relative in required, relative

    destination = root / relative
    assert destination.is_file(), f"Missing sensor: {destination}"
    referenced.add(relative)
    return str(destination)

def replace(container, key, prefix):
    before = container[key]
    after = resolve(before, prefix)
    container[key] = after
    changes.append({"original": before, "resolved": after})

rows = updated["data_list"]
for index, row in enumerate(rows, 1):
    replace(
        row["lidar_points"], "lidar_path",
        "samples/LIDAR_TOP",
    )
    for camera, record in row["images"].items():
        replace(record, "img_path", "samples/" + camera)

    for sweep in row.get("lidar_sweeps", []):
        replace(
            sweep["lidar_points"], "lidar_path",
            "sweeps/LIDAR_TOP",
        )

    if index % 100 == 0:
        print(f"Resolved sensor paths: {index}/600 frames", flush=True)

# Restore the old path values in a comparison copy, then compare everything.
comparison = copy.deepcopy(updated)
for old_row, new_row in zip(
    original["data_list"], comparison["data_list"]
):
    new_row["lidar_points"]["lidar_path"] = (
        old_row["lidar_points"]["lidar_path"]
    )
    for camera in old_row["images"]:
        new_row["images"][camera]["img_path"] = (
            old_row["images"][camera]["img_path"]
        )
    for old_sweep, new_sweep in zip(
        old_row.get("lidar_sweeps", []),
        new_row.get("lidar_sweeps", []),
    ):
        new_sweep["lidar_points"]["lidar_path"] = (
            old_sweep["lidar_points"]["lidar_path"]
        )

def assert_equal(left, right, location="root"):
    assert type(left) is type(right), location
    if isinstance(left, dict):
        assert left.keys() == right.keys(), location
        for key in left:
            assert_equal(left[key], right[key], location + "." + str(key))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), location
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, location + f"[{index}]")
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype, location
        assert np.array_equal(left, right, equal_nan=True), location
    elif isinstance(left, (float, np.floating)):
        assert left == right or (
            np.isnan(left) and np.isnan(right)
        ), location
    else:
        assert left == right, location

assert_equal(original, comparison)

tokens = (study / "sample_tokens.txt").read_text().splitlines()
assert [row["token"] for row in rows] == tokens
assert len(rows) == 600

destination = study / "annotations_kaggle.pkl"
temporary = destination.with_suffix(".pkl.writing")
with temporary.open("wb") as stream:
    pickle.dump(updated, stream, protocol=4)

with temporary.open("rb") as stream:
    restored = pickle.load(stream)
assert_equal(updated, restored)
temporary.replace(destination)

record = {
    "status": "KAGGLE_ANNOTATION_PATH_RESOLUTION_PASS",
    "samples": 600,
    "same_ordered_frozen_tokens": True,
    "original_annotation_sha256": sha256(source),
    "annotation_path": str(destination),
    "annotation_sha256": sha256(destination),
    "sensor_root": str(root),
    "resolved_path_references": len(changes),
    "unique_referenced_sensor_files": len(referenced),
    "non_path_annotation_content_unchanged": True,
    "scope": (
        "All referenced sensors exist. Only sensor path fields changed; "
        "sensor contents were verified in the preceding transfer audit. "
        "Actual model/corruption smoke remains pending."
    ),
}
(study / "annotation_path_audit.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
(study / "annotation_path_changes.json").write_text(
    json.dumps(changes, indent=2) + "\n"
)
print(json.dumps(record, indent=2), flush=True)
