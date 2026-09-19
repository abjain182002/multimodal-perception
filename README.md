# Multimodal Perception

MSc Major Project: Software-only deep learning pipeline for multi-modal object detection and robotic perception using camera and LiDAR fusion.

## Project Status

**Technical implementation: approximately 65% complete** (working estimate).

Latest verified milestone: **controlled YOLOv8x + LiDAR vs Falcon + LiDAR comparison** on matched KITTI development and independent holdout splits using the frozen projected-LiDAR median-depth association method.

Next technical priorities:

- evaluate one current released YOLO-family detector if justified by time and compatibility,
- establish PointPillars as a genuine LiDAR-only 3D baseline,
- add one stronger multimodal fusion method,
- extend evaluation to nuScenes,
- run targeted robustness experiments,
- complete a standardized runtime benchmark.

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

See [`results/README.md`](results/README.md) for the artifact-storage policy.

## Storage Policy

Datasets, virtual environments, model weights, and bulk generated `model_outputs/` are intentionally excluded from Git history. Source code, evaluation scripts, compact numerical results, dissertation-ready figures, and selected qualitative examples are version-controlled.
