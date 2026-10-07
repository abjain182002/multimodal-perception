"""Verify and record PV-RCNN runtime evidence using only the Python standard library."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def percentile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    expected = "23bd999fa5c64cc825d6780e6a5c4322cc701d55ae2903e21d2dac16c924efd7"
    if sha(args.archive.read_bytes()) != expected:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(args.archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or zipped.testzip() is not None:
            raise ValueError("Invalid ZIP")
        runtime = json.loads(zipped.read("runtime.json"))
        rows = list(csv.DictReader(io.StringIO(zipped.read("latency.csv").decode())))
        source = zipped.read("benchmark_pv_rcnn_local.py")
        if runtime["status"] != "PASS" or runtime["schema_version"] != "pv_rcnn_runtime_v1":
            raise ValueError("Invalid benchmark status")
        if sha(source) != runtime["source_sha256"]["script"]:
            raise ValueError("Executed benchmark source differs")
        if runtime["batch_size"] != 1 or runtime["warmup_calls"] != 10:
            raise ValueError("Unexpected benchmark protocol")
        if runtime["measured_calls"] != len(rows) or len(rows) != 96:
            raise ValueError("Expected 96 measurements")
        ids = runtime["frame_ids"]
        if len(ids) != len(set(ids)) or len(ids) != 32:
            raise ValueError("Expected 32 unique frame IDs")
        if Counter(r["frame_id"] for r in rows) != Counter({fid: 3 for fid in ids}):
            raise ValueError("Frame measurement counts differ")
        if Counter((r["frame_id"], int(r["repeat"])) for r in rows) != Counter(
                {(fid, repeat): 1 for fid in ids for repeat in (1, 2, 3)}):
            raise ValueError("Repeat assignments differ")
        for row in rows:
            values = [float(row[f]) for f in ("preprocessing_ms", "transfer_ms", "inference_ms", "pipeline_ms")]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError("Invalid latency")
            if not math.isclose(sum(values[:3]), values[3], abs_tol=1e-7):
                raise ValueError("Stage latencies do not sum to pipeline")
        for field, saved in runtime["latency"].items():
            values = [float(row[field]) for row in rows]
            recomputed = {"mean": statistics.mean(values), "median": statistics.median(values),
                          "p95": percentile(values, .95), "min": min(values), "max": max(values)}
            for key, value in recomputed.items():
                if not math.isclose(value, saved[key], rel_tol=1e-10, abs_tol=1e-8):
                    raise ValueError("Runtime statistic differs: " + field + "/" + key)
        rate = 1000 / runtime["latency"]["pipeline_ms"]["mean"]
        if not math.isclose(rate, runtime["serial_pipeline_calls_per_second"], rel_tol=1e-10):
            raise ValueError("Serial pipeline rate differs")
        root = Path(__file__).resolve().parents[1]
        out = root / "results/kitti/pv_rcnn/runtime"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("runtime.json", "latency.csv", "environment.txt", "nvidia_smi.txt",
                     "model_loading.log", "benchmark_pv_rcnn_local.py"):
            if name in names:
                (out / name).write_bytes(zipped.read(name))
        receipt = {"status": "PASS", "archive_sha256": expected,
                   "archive_bytes": args.archive.stat().st_size, "members": len(names),
                   "measured_calls_verified": 96, "unique_frames": 32,
                   "csv_statistics_recomputed": True,
                   "members_sha256": {name: sha(zipped.read(name)) for name in names}}
    (out / "archive_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    labels = {"preprocessing_ms": "Preprocessing", "transfer_ms": "GPU transfer",
              "inference_ms": "Inference + postprocessing", "pipeline_ms": "Measured pipeline"}
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["stage", "mean_ms", "median_ms", "p95_ms"])
        for field, label in labels.items():
            values = runtime["latency"][field]
            writer.writerow([label, values["mean"], values["median"], values["p95"]])
    # An editable vector chart needs no plotting dependency.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="490" viewBox="0 0 1000 490">',
           '<rect width="1000" height="490" fill="white"/>',
           '<g font-family="Arial,sans-serif" fill="#222">',
           '<text x="30" y="32" font-size="22">PV-RCNN latency: one Tesla T4, batch size 1</text>',
           '<text x="30" y="59" font-size="15">10 warm-up calls; 32 KITTI frames × 3 passes; GPU-synchronized wall time</text>']
    maximum = max(v["p95"] for v in runtime["latency"].values())
    ceiling = math.ceil(maximum / 50) * 50
    left = 275
    scale = 630 / ceiling
    for tick in range(0, ceiling+1, 50):
        x = left + tick*scale
        svg += [f'<line x1="{x}" x2="{x}" y1="85" y2="390" stroke="#dddddd"/>',
                f'<text x="{x}" y="413" text-anchor="middle" font-size="13">{tick}</text>']
    for index, (field, label) in enumerate(labels.items()):
        y = 98 + index*75
        svg.append(f'<text x="20" y="{y+29}" font-size="15">{label}</text>')
        for offset, key, color in ((0, "median", "#2864a0"), (27, "p95", "#df7926")):
            value = runtime["latency"][field][key]
            width = value*scale
            svg += [f'<rect x="{left}" y="{y+offset}" width="{width:.3f}" height="21" fill="{color}"/>',
                    f'<text x="{left+width+7:.3f}" y="{y+offset+16}" font-size="13">{value:.2f}</text>']
    svg += ['<text x="550" y="440" font-size="15">Latency (ms); axis starts at zero</text>',
            '<rect x="275" y="455" width="15" height="15" fill="#2864a0"/>',
            '<text x="299" y="468" font-size="14">Median</text>',
            '<rect x="390" y="455" width="15" height="15" fill="#df7926"/>',
            '<text x="414" y="468" font-size="14">p95</text>', '</g></svg>']
    (out / "latency_summary.svg").write_text("\n".join(svg)+"\n")
    text = """# PV-RCNN runtime result

On one Kaggle Tesla T4 at batch size 1, the measured pipeline median was 176.31 ms and p95 was 196.16 ms. Model forward including detector postprocessing had median 149.45 ms and p95 168.05 ms. The serial measured call rate was 5.54 calls/s, calculated from mean pipeline latency.

Protocol: 10 warm-up calls followed by three passes over 32 fixed KITTI validation frames (96 measurements), with CUDA synchronization at timing boundaries. CSV statistics were independently recomputed by the recorder.

Figure caption: GPU-synchronized serial wall-clock latency for official pretrained OpenPCDet PV-RCNN. Bars show median and p95 from 96 calls on one T4. The pipeline includes official evaluation-dataset preparation/collation, GPU transfer and model forward/postprocessing. It excludes checkpoint loading, KITTI annotation conversion and file writing. Stage medians/p95 values are not additive.

Limitations: one session and a small fixed subset; repeated reads may be cached. Dataset preparation includes GT bookkeeping. The serial call rate is not asynchronous throughput. Peak Torch allocations omit some external CUDA allocations. These numbers do not establish a runtime comparison with PointPillars or CLOCs; those models need the same timing protocol.

The executed benchmark source, environment, raw latency CSV, machine snapshot and archive receipt are retained alongside this result. The repository benchmark and notebook execution can be distinguished by the source SHA-256.
"""
    (out / "FINDINGS.md").write_text(text)
    print(json.dumps({"status": "PASS", "measurements": 96, "output": str(out),
                      "pipeline_median_ms": runtime["latency"]["pipeline_ms"]["median"]}, indent=2))


if __name__ == "__main__":
    main()

