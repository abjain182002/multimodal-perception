
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

root = Path("/kaggle/working/openpcdet_kitti_validation")
out = Path("/kaggle/working/pv_rcnn_runtime")
out.mkdir(exist_ok=True)
weights = Path(
    "/kaggle/working/openpcdet_checkpoints/pv_rcnn_official_kitti.pth"
)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

assert sha(weights) == (
    "08552932026855bcc10f66dbc45409f139d3208fd0095c2aab1f459f055fe463"
)
metadata = json.loads((root / "metadata_audit.json").read_text())
assert metadata["status"] == "PASS"
assert sha(root / "kitti_infos_val.pkl") == metadata["metadata_sha256"]
assert torch.cuda.is_available() and torch.cuda.device_count() == 1

seed = 20260930
torch.manual_seed(seed)
np.random.seed(seed)
rng = np.random.default_rng(seed)

cfg_from_yaml_file("cfgs/kitti_models/pv_rcnn.yaml", cfg)
logger = common_utils.create_logger(
    log_file=str(out / "model_loading.log")
)
dataset = KittiDataset(
    cfg.DATA_CONFIG, cfg.CLASS_NAMES,
    training=False, root_path=root, logger=logger
)
assert len(dataset) == 3769

indices = np.linspace(0, 3768, 32, dtype=int)
model = build_network(
    cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
)
model.load_params_from_file(str(weights), logger=logger, to_cpu=True)
model.cuda().eval()

def measure(index):
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch = dataset.collate_batch([dataset[index]])
    prepared = time.perf_counter()

    load_data_to_gpu(batch)
    torch.cuda.synchronize()
    transferred = time.perf_counter()

    predictions, _ = model(batch)
    torch.cuda.synchronize()
    finished = time.perf_counter()

    return {
        "frame_id": dataset.kitti_infos[index]["point_cloud"]["lidar_idx"],
        "preprocessing_ms": (prepared-start)*1000,
        "transfer_ms": (transferred-prepared)*1000,
        "inference_ms": (finished-transferred)*1000,
        "pipeline_ms": (finished-start)*1000,
        "detections": len(predictions[0]["pred_scores"]),
    }

rows = []
with torch.no_grad():
    print("Warm-up: 10 calls", flush=True)
    for index in indices[:10]:
        measure(int(index))
    torch.cuda.reset_peak_memory_stats()

    for repeat in range(3):
        for index in rng.permutation(indices):
            record = measure(int(index))
            record["repeat"] = repeat + 1
            rows.append(record)
        print(f"Measured pass {repeat+1}/3", flush=True)

latency = {}
for field in (
    "preprocessing_ms", "transfer_ms", "inference_ms", "pipeline_ms"
):
    values = np.array([row[field] for row in rows])
    latency[field] = {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }

report = {
    "schema_version": "pv_rcnn_runtime_v1",
    "status": "PASS",
    "model": "Official pretrained OpenPCDet PV-RCNN",
    "batch_size": 1,
    "warmup_calls": 10,
    "unique_frames": 32,
    "repeats": 3,
    "measured_calls": len(rows),
    "seed": seed,
    "frame_ids": [
        dataset.kitti_infos[int(i)]["point_cloud"]["lidar_idx"]
        for i in indices
    ],
    "machine": {
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    },
    "latency": latency,
    "serial_pipeline_calls_per_second": (
        1000 / latency["pipeline_ms"]["mean"]
    ),
    "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
    "source_commit": "233f849829b6ac19afb8af8837a0246890908755",
    "source_sha256": {
        "script": sha(Path(__file__)),
        "checkpoint": sha(weights),
        "metadata": metadata["metadata_sha256"],
        "config": sha(Path("cfgs/kitti_models/pv_rcnn.yaml")),
        "dataset_config": sha(Path("cfgs/dataset_configs/kitti_dataset.yaml")),
    },
    "scope": (
        "Synchronized serial latency: official dataset preparation/collation, "
        "GPU transfer and forward including detector postprocessing. "
        "Excludes loading, KITTI annotation conversion and output writing."
    ),
    "limitations": (
        "One session; repeated reads may be cached. Dataset preprocessing "
        "includes GT bookkeeping. Serial call rate is not asynchronous "
        "throughput. Torch memory excludes some external CUDA allocations."
    ),
}
(out / "runtime.json").write_text(json.dumps(report, indent=2) + "\n")
with (out / "latency.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)

print(json.dumps(report, indent=2), flush=True)
