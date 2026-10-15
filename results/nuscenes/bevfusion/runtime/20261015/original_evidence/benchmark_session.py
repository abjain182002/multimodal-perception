
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.runner import load_checkpoint, set_random_seed
from mmdet3d.apis import init_model
from mmdet3d.registry import DATASETS

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["lidar", "lidar_camera"], required=True)
parser.add_argument("--repeat", type=int, required=True)
args = parser.parse_args()

local = Path("/content/runtime_benchmark150")
study = Path(
    "/content/drive/MyDrive/Major_Project/"
    "nuscenes_full_validation/runtime_benchmark150_20261015"
)
root = Path("/content/datasets/nuscenes_full")
repo = Path("/content/mmdetection3d")
session_name = f"repeat_{args.repeat:02d}_{args.model}"
out = local / "sessions" / session_name
destination = study / "sessions" / session_name
out.mkdir(parents=True, exist_ok=True)
destination.mkdir(parents=True, exist_ok=True)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def backup(source, target):
    temporary = target.with_name(target.name + ".uploading")
    temporary.write_bytes(source.read_bytes())
    assert sha256(source) == sha256(temporary)
    temporary.replace(target)

protocol = json.loads((local / "protocol.json").read_text())
metadata = json.loads((local / "metadata_audit.json").read_text())
receipt = json.loads((study / "checkpoint_receipt.json").read_text())
tokens = (local / "sample_tokens.txt").read_text().splitlines()

assert len(tokens) == len(set(tokens)) == 150
assert sha256(metadata["annotation_path"]) == metadata["annotation_sha256"]
assert sha256(local / "protocol.json") == metadata["protocol_sha256"]
assert torch.__version__ == protocol["torch"]
assert torch.cuda.get_device_name(0) == protocol["gpu"]
commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()
assert commit == protocol["source_commit"]

torch.set_num_threads(protocol["torch_cpu_threads"])
torch.set_num_interop_threads(1)
set_random_seed(protocol["seed"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

filenames = {
    "lidar": "bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
    "lidar_camera": "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
}
cfg = Config.fromfile(str(
    repo / "projects/BEVFusion/configs" / filenames[args.model]
))
dataset_cfg = cfg.test_dataloader.dataset
dataset_cfg.data_root = str(root) + "/"
dataset_cfg.ann_file = metadata["annotation_path"]
dataset_cfg.metainfo["version"] = "v1.0-trainval"

checkpoint = Path(receipt[args.model]["path"])
assert sha256(checkpoint) == receipt[args.model]["sha256"]

print("LOADING:", session_name, flush=True)
model = init_model(cfg, str(checkpoint), device="cuda:0")
load_checkpoint(model, str(checkpoint), map_location="cpu", strict=True)
model.eval()
assert all(
    parameter.dtype == torch.float32
    for parameter in model.parameters() if parameter.is_floating_point()
)

dataset = DATASETS.build(dataset_cfg)
assert len(dataset) == 150
assert [dataset.get_data_info(i)["token"] for i in range(150)] == tokens
cfg.dump(str(out / "resolved_config.py"))
print("STRICT MODEL LOAD AND ORDERED TOKENS PASS", flush=True)

def validate(predictions):
    assert len(predictions) == 1
    instances = predictions[0].pred_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu()
    scores = instances.scores_3d.detach().cpu()
    labels = instances.labels_3d.detach().cpu()
    assert len(boxes) == len(scores) == len(labels)
    assert torch.isfinite(boxes).all()
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    assert ((labels >= 0) & (labels < 10)).all()
    return len(scores)

print("WARM-UP: 20 samples", flush=True)
with torch.inference_mode(), torch.autocast("cuda", enabled=False):
    for index in protocol["warmup_sample_indices"]:
        batch = pseudo_collate([dataset[index]])
        predictions = model.test_step(batch)
        torch.cuda.synchronize()
        validate(predictions)
        del predictions, batch

gc.collect()
torch.cuda.synchronize()
baseline_allocated = torch.cuda.memory_allocated()
baseline_reserved = torch.cuda.memory_reserved()
torch.cuda.reset_peak_memory_stats()
print("WARM-UP PASSED — STARTING TIMED PASS", flush=True)

rows = []
with torch.inference_mode(), torch.autocast("cuda", enabled=False):
    for index, token in enumerate(tokens):
        load_start = time.perf_counter()
        batch = pseudo_collate([dataset[index]])
        loading_ms = 1000 * (time.perf_counter() - load_start)

        input_points = int(batch["inputs"]["points"][0].shape[0])

        # Synchronize before and after the measured test_step.
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = model.test_step(batch)
        torch.cuda.synchronize()
        latency_ms = 1000 * (time.perf_counter() - started)

        # CPU copying and validation occur outside the timed interval.
        detections = validate(predictions)
        rows.append({
            "sample_index": index,
            "sample_token": token,
            "model": args.model,
            "repeat": args.repeat,
            "loading_and_collation_ms": loading_ms,
            "test_step_latency_ms": latency_ms,
            "input_points": input_points,
            "detections": detections,
        })
        del predictions, batch

        if (index + 1) % 25 == 0:
            print(json.dumps({
                "session": session_name,
                "completed": index + 1,
                "total": 150,
                "median_test_step_ms_so_far": round(float(np.median(
                    [r["test_step_latency_ms"] for r in rows]
                )), 2),
            }), flush=True)

torch.cuda.synchronize()
latencies = np.array([row["test_step_latency_ms"] for row in rows])
loading = np.array([row["loading_and_collation_ms"] for row in rows])
assert np.isfinite(latencies).all() and (latencies > 0).all()

raw = out / "measurements.json"
raw.write_text(json.dumps(rows, indent=2) + "\n")
environment = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "cpu_threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "nvidia_smi": subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=name,driver_version,temperature.gpu,pstate,"
        "clocks.sm,clocks.mem,memory.total",
        "--format=csv,noheader",
    ], text=True).strip(),
}
(out / "environment.json").write_text(
    json.dumps(environment, indent=2) + "\n"
)

summary = {
    "status": "RUNTIME_SESSION_PASS",
    "session": session_name,
    "model": args.model,
    "repeat": args.repeat,
    "samples": 150,
    "warmup_iterations": 20,
    "latency_ms": {
        "mean": float(latencies.mean()),
        "median": float(np.median(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "minimum": float(latencies.min()),
        "maximum": float(latencies.max()),
    },
    "loading_and_collation_ms": {
        "mean": float(loading.mean()),
        "median": float(np.median(loading)),
        "p95": float(np.percentile(loading, 95)),
    },
    "memory_bytes": {
        "allocated_after_warmup": baseline_allocated,
        "reserved_after_warmup": baseline_reserved,
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    },
    "protocol_sha256": sha256(local / "protocol.json"),
    "annotation_sha256": metadata["annotation_sha256"],
    "checkpoint_sha256": receipt[args.model]["sha256"],
    "benchmark_script_sha256": sha256(__file__),
    "measurements_sha256": sha256(raw),
    "scope": "Serial synchronized test_step latency; no new accuracy evaluation.",
}
summary_path = out / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

for source in out.iterdir():
    if source.is_file():
        backup(source, destination / source.name)

print(json.dumps(summary, indent=2), flush=True)
print("SESSION RESULTS VERIFIED ON DRIVE:", session_name, flush=True)
