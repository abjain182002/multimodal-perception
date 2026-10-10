import gc
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

repo = Path("/content/mmdetection3d")
sys.path.insert(0, str(repo))
from tools.dataset_converters.nuscenes_converter import _fill_trainval_infos
from tools.dataset_converters.update_infos_to_v2 import update_nuscenes_infos

root = Path("/content/datasets/nuscenes_full")
base = Path(
    "/content/drive/MyDrive/Major_Project/nuscenes_full_validation"
)
plan_path = root / "full_validation_batch_plan.json"
plan = json.loads(plan_path.read_text())
batch = plan["batches"][1]
tokens = batch["sample_tokens"]
scene_tokens = set(batch["scene_tokens"])
assert len(tokens) == len(set(tokens)) == 2368

restoration = json.loads(
    (root / "batch_restoration_audits/batch_02.json").read_text()
)
assert restoration["status"] == "BATCH_SENSOR_RESTORATION_PASS"
assert restoration["plan_sha256"] == hashlib.sha256(
    plan_path.read_bytes()
).hexdigest()

for relative in batch["required_files"]:
    assert (root / relative).is_file(), relative

link = Path("data/nuscenes")
if link.is_symlink():
    assert link.resolve() == root.resolve()
elif link.exists():
    raise RuntimeError(f"Unexpected existing data directory: {link}")
else:
    link.symlink_to(root, target_is_directory=True)

nusc = NuScenes(
    version="v1.0-trainval", dataroot=str(root), verbose=True
)
sample_by_token = {s["token"]: s for s in nusc.sample}
selected = [sample_by_token[token] for token in tokens]
assert {s["scene_token"] for s in selected} == scene_tokens

official_val_names = set(create_splits_scenes()["val"])
selected_names = {
    nusc.get("scene", token)["name"] for token in scene_tokens
}
assert selected_names <= official_val_names
assert {
    s["token"] for s in nusc.sample
    if s["scene_token"] in scene_tokens
} == set(tokens), "Batch must contain complete scenes."

# Restrict converter iteration while SDK lookups retain their
# original tables and token indexes.
class ConversionView:
    def __init__(self, sdk, samples):
        self.sdk = sdk
        self.sample = samples

    def __getattr__(self, name):
        return getattr(self.sdk, name)

view = ConversionView(nusc, selected)
for token in tokens:
    assert view.get("sample", token)["token"] == token
print("BATCH ITERATION AND SDK LOOKUPS PASS", flush=True)

# Inputs include the keyframe plus up to nine preceding LiDAR sweeps.
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

filename = "nuscenes_batch02_infos_val.pkl"
raw = root / filename
mmengine.dump({
    "infos": val_infos,
    "metadata": {"version": "v1.0-trainval"},
}, str(raw))

del view, nusc, selected, sample_by_token, train_infos, val_infos
gc.collect()

output_dir = root / "mmdet3d_infos"
output_dir.mkdir(exist_ok=True)
update_nuscenes_infos(str(raw), str(output_dir))

annotation = output_dir / filename
converted = mmengine.load(str(annotation))
assert converted["metainfo"]["version"] == "v1.0-trainval"
assert [row["token"] for row in converted["data_list"]] == tokens

audit = {
    "status": "BATCH_METADATA_PASS",
    "batch": 2,
    "version": "v1.0-trainval",
    "samples": len(tokens),
    "scenes": len(scene_tokens),
    "maximum_preceding_lidar_sweeps": 9,
    "same_ordered_batch_tokens": True,
    "source_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "annotation_path": str(annotation),
    "annotation_sha256": hashlib.sha256(
        annotation.read_bytes()
    ).hexdigest(),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "scope": "Batch 2 of full-validation inference; accuracy pending.",
}
audit_path = output_dir / "batch02_metadata_audit.json"
audit_path.write_text(json.dumps(audit, indent=2) + "\n")

destination = base / "full_validation_metadata/batch_02"
destination.mkdir(parents=True, exist_ok=True)
for source in (annotation, audit_path, Path(__file__)):
    target = destination / source.name
    shutil.copy2(source, target)
    assert hashlib.sha256(source.read_bytes()).digest() == \
           hashlib.sha256(target.read_bytes()).digest()

print(json.dumps(audit, indent=2))
print("BATCH 2 METADATA AND DRIVE BACKUP PASS", flush=True)
