"""Synchronized batch-one PV-RCNN pipeline latency on fixed KITTI frames."""
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path("/kaggle/working/openpcdet_kitti_validation")
    weights = Path("/kaggle/working/openpcdet_checkpoints/pv_rcnn_official_kitti.pth")
    out = Path("/kaggle/working/pv_rcnn_runtime")
    out.mkdir(exist_ok=True)
    config = Path("cfgs/kitti_models/pv_rcnn.yaml")
    dataset_config = Path("cfgs/dataset_configs/kitti_dataset.yaml")
    assert sha(weights) == "08552932026855bcc10f66dbc45409f139d3208fd0095c2aab1f459f055fe463"
    audit = json.loads((root / "metadata_audit.json").read_text())
    assert audit["status"] == "PASS" and audit["frames"] == 3769
    assert sha(root / "kitti_infos_val.pkl") == audit["metadata_sha256"]
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    seed = 20260930
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg_from_yaml_file(str(config), cfg)
    logger = common_utils.create_logger(log_file=str(out / "model_loading.log"))
    dataset = KittiDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False,
                           root_path=root, logger=logger)
    assert len(dataset) == 3769
    indices = np.linspace(0, len(dataset)-1, 32, dtype=int).tolist()
    frame_ids = [dataset.kitti_infos[i]["point_cloud"]["lidar_idx"] for i in indices]
    model = build_network(cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(str(weights), logger=logger, to_cpu=True)
    model.cuda().eval()
    rng = np.random.default_rng(seed)

    def measure(index):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = dataset.collate_batch([dataset[index]])
        after_cpu = time.perf_counter()
        load_data_to_gpu(batch)
        torch.cuda.synchronize()
        after_transfer = time.perf_counter()
        predictions, _ = model(batch)
        torch.cuda.synchronize()
        finish = time.perf_counter()
        return {
            "frame_id": dataset.kitti_infos[index]["point_cloud"]["lidar_idx"],
            "preprocessing_ms": (after_cpu-start)*1000,
            "transfer_ms": (after_transfer-after_cpu)*1000,
            "inference_ms": (finish-after_transfer)*1000,
            "pipeline_ms": (finish-start)*1000,
            "detections": int(len(predictions[0]["pred_scores"])),
        }

    rows = []
    with torch.no_grad():
        print("Warm-up: 10 calls", flush=True)
        for i in range(10):
            measure(indices[i % len(indices)])
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(3):
            for index in rng.permutation(indices):
                record = measure(int(index))
                record["repeat"] = repeat + 1
                rows.append(record)
            print("Measured pass", repeat+1, "/ 3", flush=True)
    summaries = {}
    for field in ("preprocessing_ms", "transfer_ms", "inference_ms", "pipeline_ms"):
        values = np.array([row[field] for row in rows])
        summaries[field] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min()),
            "max": float(values.max()),
        }
    report = {
        "schema_version": "pv_rcnn_runtime_v1",
        "status": "PASS", "model": "Official pretrained OpenPCDet PV-RCNN",
        "batch_size": 1, "warmup_calls": 10, "unique_frames": 32,
        "repeats": 3, "measured_calls": len(rows), "seed": seed,
        "frame_ids": frame_ids,
        "machine": {
            "platform": platform.platform(), "python": sys.version,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "packages": {name: version(name) for name in ("pcdet", "spconv-cu120", "numpy")},
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "source_sha256": {"script": sha(Path(__file__)), "checkpoint": sha(weights),
                           "config": sha(config), "dataset_config": sha(dataset_config),
                           "metadata": audit["metadata_sha256"]},
        "source_commit": "233f849829b6ac19afb8af8837a0246890908755",
        "latency": summaries,
        "serial_pipeline_calls_per_second": 1000/summaries["pipeline_ms"]["mean"],
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "scope": (
            "Synchronized serial latency: official KITTI dataset preparation/collation, "
            "host-to-device transfer, model forward including detector postprocessing. "
            "Excludes model loading, KITTI annotation conversion, evaluation and output writing."
        ),
        "limitations": (
            "32 frames, three passes, one Kaggle T4 session; repeated reads may be cached. "
            "Preprocessing includes evaluation-dataset GT bookkeeping. "
            "Serial call rate is not batched/asynchronous throughput; Torch peak memory "
            "does not include every external CUDA allocation. No cross-model timing claim."
        ),
    }
    (out / "runtime.json").write_text(json.dumps(report, indent=2)+"\n")
    with (out / "latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for command, name in ((["nvidia-smi"], "nvidia_smi.txt"),
                          (["nvcc", "--version"], "nvcc.txt")):
        run = subprocess.run(command, capture_output=True, text=True)
        (out / name).write_text(run.stdout + run.stderr)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
