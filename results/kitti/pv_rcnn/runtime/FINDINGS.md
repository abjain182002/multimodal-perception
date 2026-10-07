# PV-RCNN runtime result

On one Kaggle Tesla T4 at batch size 1, the measured pipeline median was 176.31 ms and p95 was 196.16 ms. Model forward including detector postprocessing had median 149.45 ms and p95 168.05 ms. The serial measured call rate was 5.54 calls/s, calculated from mean pipeline latency.

Protocol: 10 warm-up calls followed by three passes over 32 fixed KITTI validation frames (96 measurements), with CUDA synchronization at timing boundaries. CSV statistics were independently recomputed by the recorder.

Figure caption: GPU-synchronized serial wall-clock latency for official pretrained OpenPCDet PV-RCNN. Bars show median and p95 from 96 calls on one T4. The pipeline includes official evaluation-dataset preparation/collation, GPU transfer and model forward/postprocessing. It excludes checkpoint loading, KITTI annotation conversion and file writing. Stage medians/p95 values are not additive.

Limitations: one session and a small fixed subset; repeated reads may be cached. Dataset preparation includes GT bookkeeping. The serial call rate is not asynchronous throughput. Peak Torch allocations omit some external CUDA allocations. These numbers do not establish a runtime comparison with PointPillars or CLOCs; those models need the same timing protocol.

The executed benchmark source, environment, raw latency CSV, machine snapshot and archive receipt are retained alongside this result. The repository benchmark and notebook execution can be distinguished by the source SHA-256.

