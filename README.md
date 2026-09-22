# Multimodal Perception

MSc Major Project: Software-only deep learning pipeline for multi-modal object detection and robotic perception using camera and LiDAR fusion.

## Project Status

**Technical implementation: approximately 78% complete** (working estimate).

Latest verified milestone: **PointPillars LiDAR-only 3D baseline** on the standard 3,769-frame KITTI validation split using the pinned MMDetection3D implementation. The primary reproduced result is **Car 3D AP40 = 79.1299% for Moderate difficulty at strict IoU 0.70**, with exact metric re-evaluation from frozen predictions and a standard train/validation split-overlap audit.

The controlled **YOLOv8x vs YOLO26x + LiDAR** comparison is also complete.

Next technical priorities:

- implement and evaluate one stronger multimodal fusion method,
- extend evaluation to nuScenes / cross-dataset analysis,
- run targeted robustness experiments,
- complete a standardized runtime benchmark,
- consolidate final ablations, failure analysis and reproducibility evidence.

## Project Records

- [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) — master technical/dissertation record: project rationale, experiments, numerical results, failures, fixes, design decisions, limitations, and supported interpretations.
- [`DAILY_LOG.md`](DAILY_LOG.md) — chronological work-session record for new experiments and engineering changes.
- [`requirements/ENVIRONMENTS.md`](requirements/ENVIRONMENTS.md) — environment strategy and reproducibility notes.

## Frozen Results

- [`results/depth_method_comparison.csv`](results/depth_method_comparison.csv)
- [`results/depth_method_comparison.png`](results/depth_method_comparison.png)
- [`results/camera_detector_comparison.csv`](results/camera_detector_comparison.csv)
- [`results/camera_detector_comparison.png`](results/camera_detector_comparison.png)
- [`results/yolov8_falcon_lidar_comparison.csv`](results/yolov8_falcon_lidar_comparison.csv)
- [`results/yolov8_falcon_lidar_comparison.png`](results/yolov8_falcon_lidar_comparison.png)
- [`results/yolov8_yolo26_lidar_comparison.csv`](results/yolov8_yolo26_lidar_comparison.csv)
- [`results/yolov8_yolo26_lidar_comparison.png`](results/yolov8_yolo26_lidar_comparison.png)
- [`results/pointpillars_kitti_ap40_strict_summary.csv`](results/pointpillars_kitti_ap40_strict_summary.csv)
- [`results/pointpillars_kitti_ap40_strict.png`](results/pointpillars_kitti_ap40_strict.png)
- [`results/pointpillars_distance_diagnostic.png`](results/pointpillars_distance_diagnostic.png)
- [`results/pointpillars_evidence_manifest.json`](results/pointpillars_evidence_manifest.json)

See [`results/README.md`](results/README.md) for the artifact-storage policy.

## Storage Policy

Datasets, virtual environments, model weights, and bulk generated `model_outputs/` are intentionally excluded from Git history. Source code, evaluation scripts, compact numerical results, dissertation-ready figures, and selected qualitative examples are version-controlled.
