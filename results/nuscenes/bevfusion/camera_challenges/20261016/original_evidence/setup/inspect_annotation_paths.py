
from pathlib import Path
import hashlib
import json
import pickle
import sys

inputs = Path(sys.argv[1])
study = Path("/kaggle/working/camera_challenges600")
annotation = (
    inputs / "annotations/"
    "nuscenes_camera_availability600_infos_val.pkl"
)
expected_hash = (
    "ff6dd3c63e8c3de23b9754947db326d05"
    "49bb376271cb7e235a3d54b0c365c44"
)
assert hashlib.sha256(annotation.read_bytes()).hexdigest() == expected_hash

with annotation.open("rb") as stream:
    data = pickle.load(stream)

rows = data["data_list"]
tokens = (study / "sample_tokens.txt").read_text().splitlines()
assert len(rows) == len(tokens) == 600
assert [row["token"] for row in rows] == tokens

first = rows[0]
print("\nFIRST SAMPLE KEYS:", list(first), flush=True)
print("FIRST LIDAR RECORD:", json.dumps(
    first.get("lidar_points"), default=str, indent=2
), flush=True)
print("FIRST CAMERA PATHS:", json.dumps({
    name: record.get("img_path")
    for name, record in first["images"].items()
}, indent=2), flush=True)

sweeps = first.get("lidar_sweeps", [])
print("PRECEDING SWEEPS:", len(sweeps), flush=True)
if sweeps:
    print("FIRST SWEEP:", json.dumps(
        sweeps[0], default=str, indent=2
    ), flush=True)

audit = {
    "status": "KAGGLE_ORIGINAL_ANNOTATION_AND_TOKENS_PASS",
    "annotation_path": str(annotation),
    "annotation_sha256": expected_hash,
    "samples": 600,
    "same_ordered_frozen_tokens": True,
    "sensor_root": str(inputs / "dataset"),
    "scope": "Path resolution and actual model smoke remain pending.",
}
(study / "annotation_input_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n"
)
print(json.dumps(audit, indent=2), flush=True)
