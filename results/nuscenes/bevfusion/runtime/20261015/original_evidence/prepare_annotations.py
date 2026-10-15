
from pathlib import Path
import copy
import hashlib
import json
import mmengine

local = Path("/content/runtime_benchmark150")
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/runtime_benchmark150_20261015"
)
source = local / "nuscenes_camera_availability600_infos_val.pkl"
data = mmengine.load(str(source))
tokens = (local / "sample_tokens.txt").read_text().splitlines()
by_token = {row["token"]: row for row in data["data_list"]}
assert len(by_token) == 600
assert len(tokens) == len(set(tokens)) == 150

selected = []
for index, token in enumerate(tokens):
    row = copy.deepcopy(by_token[token])
    row["sample_idx"] = index
    selected.append(row)

output = local / "nuscenes_runtime150_infos_val.pkl"
mmengine.dump({
    "metainfo": copy.deepcopy(data["metainfo"]),
    "data_list": selected,
}, str(output))
check = mmengine.load(str(output))
assert [row["token"] for row in check["data_list"]] == tokens
assert [row["sample_idx"] for row in check["data_list"]] == list(range(150))

digest = hashlib.sha256(output.read_bytes()).hexdigest()
report = {
    "status": "RUNTIME_METADATA_PASS",
    "samples": 150,
    "scenes": 150,
    "annotation_path": str(output),
    "annotation_sha256": digest,
    "source_annotation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "protocol_sha256": hashlib.sha256(
        (local / "protocol.json").read_bytes()
    ).hexdigest(),
    "same_ordered_frozen_tokens": True,
    "scope": "Subset of verified 600-frame annotations; no timing yet.",
}
text = json.dumps(report, indent=2) + "\n"
(local / "metadata_audit.json").write_text(text)
(study / "metadata_audit.json").write_text(text)

target = study / output.name
target.write_bytes(output.read_bytes())
assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
print(text, flush=True)
print("RUNTIME ANNOTATIONS VERIFIED ON DRIVE", flush=True)
