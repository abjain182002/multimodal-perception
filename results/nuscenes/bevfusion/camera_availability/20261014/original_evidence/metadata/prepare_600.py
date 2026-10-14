import gc
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from tools.dataset_converters.nuscenes_converter import _fill_trainval_infos
from tools.dataset_converters.update_infos_to_v2 import update_nuscenes_infos

root = Path("/content/datasets/nuscenes_full")
local = root / "camera_availability_600"
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/camera_availability_600_20261014"
)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

selection = json.loads((local / "selection_audit.json").read_text())
for name, expected in selection["record_sha256"].items():
    assert sha256(local / name) == expected, name
    assert sha256(study / name) == expected, name

tokens = (local / "sample_tokens.txt").read_text().splitlines()
restoration = json.loads(
    (local / "sensor_restoration_audit.json").read_text()
)
assert restoration["status"] == "CAMERA_AVAILABILITY_SENSOR_RESTORATION_PASS"
assert sha256(local / "sensor_restoration_audit.json") == sha256(
    study / "sensor_restoration_audit.json"
)
assert len(tokens) == len(set(tokens)) == 600

for relative, record in restoration["members"].items():
    path = root / relative
    assert path.is_file() and path.stat().st_size == record["bytes"], relative

link = Path("/content/mmdetection3d/data/nuscenes")
link.parent.mkdir(exist_ok=True)
if link.is_symlink():
    assert link.resolve() == root.resolve()
elif link.exists():
    raise RuntimeError(f"Unexpected existing directory: {link}")
else:
    link.symlink_to(root, target_is_directory=True)

nusc = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=True)
sample_by_token = {row["token"]: row for row in nusc.sample}
selected = [sample_by_token[token] for token in tokens]
scene_tokens = {row["scene_token"] for row in selected}
assert len(scene_tokens) == 150

official_val = set(create_splits_scenes()["val"])
assert {
    nusc.get("scene", token)["name"] for token in scene_tokens
} == official_val

# Iterate selected frames while preserving SDK lookups for all tables.
class ConversionView:
    def __init__(self, sdk, samples):
        self.sdk = sdk
        self.sample = samples

    def __getattr__(self, name):
        return getattr(self.sdk, name)

view = ConversionView(nusc, selected)
print("SELECTED TOKENS AND OFFICIAL VALIDATION SCENES PASS", flush=True)

train_infos, val_infos = _fill_trainval_infos(
    view, set(), scene_tokens, test=False, max_sweeps=9
)
assert not train_infos
assert [row["token"] for row in val_infos] == tokens

for row in val_infos:
    assert Path(row["lidar_path"]).is_file()
    assert len(row["cams"]) == 6
    assert len(row["sweeps"]) <= 9
    for cam in row["cams"].values():
        assert Path(cam["data_path"]).is_file(), cam["data_path"]
    for sweep in row["sweeps"]:
        assert Path(sweep["data_path"]).is_file(), sweep["data_path"]

print("CONVERTED SENSOR PATHS PASS", flush=True)
filename = "nuscenes_camera_availability600_infos_val.pkl"
raw = root / filename
mmengine.dump({
    "infos": val_infos,
    "metadata": {"version": "v1.0-trainval"},
}, str(raw))

del view, nusc, selected, sample_by_token, train_infos, val_infos
gc.collect()

output = root / "mmdet3d_infos"
output.mkdir(exist_ok=True)
update_nuscenes_infos(str(raw), str(output))

annotation = output / filename
converted = mmengine.load(str(annotation))
assert converted["metainfo"]["version"] == "v1.0-trainval"
assert [row["token"] for row in converted["data_list"]] == tokens

commit = subprocess.check_output(
    ["git", "-C", "/content/mmdetection3d", "rev-parse", "HEAD"],
    text=True
).strip()
assert commit == "fe25f7a51d36e3702f961e198894580d83c4387b"

audit = {
    "status": "CAMERA_AVAILABILITY_METADATA_PASS",
    "samples": 600,
    "scenes": 150,
    "maximum_preceding_lidar_sweeps": 9,
    "same_ordered_frozen_tokens": True,
    "source_commit": commit,
    "annotation_path": str(annotation),
    "annotation_sha256": sha256(annotation),
    "protocol_sha256": sha256(local / "protocol.json"),
    "scope": "Annotations prepared; model and mask checks pending.",
}
audit_path = local / "metadata_audit.json"
audit_path.write_text(json.dumps(audit, indent=2) + "\n")

destination = study / "metadata"
destination.mkdir(exist_ok=True)
for source in (annotation, audit_path, Path(__file__)):
    target = destination / source.name
    shutil.copy2(source, target)
    assert sha256(source) == sha256(target)

print(json.dumps(audit, indent=2), flush=True)
print("ANNOTATIONS, CONVERTER AND AUDIT VERIFIED ON DRIVE", flush=True)
