# Multimodal Perception

MSc Major Project: Software-only deep learning pipeline for multi-modal object detection and robotic perception using camera and LiDAR fusion.

## Project Status

Latest Mac checkpoint (25 September 2026): **100/100 tests pass** at `6b913a1`; working tree clean and synchronized with `origin/main`. The full 7,481-frame KITTI Car GT audit and the split-safe CPU CLOCs target builder are complete. Genuine dense 3D candidate boxes from Colab are still needed before learned CLOCs can be trained or evaluated. The existing structural smoke example has placeholder all-negative labels and is excluded from training.

The verified PointPillars LiDAR-only baseline evaluated on the standard 3,769-frame KITTI validation split has **Car 3D AP40 = 79.1299% for Moderate difficulty at strict IoU 0.70**. The controlled YOLOv8x versus YOLO26x + LiDAR and Falcon comparisons are also recorded. These results measure distinct tasks and are reported separately.

**Engineering target:** finish the agreed Phase 1 experiments and freeze evidence by **8–10 October 2026**. The [daily targets and exit criteria](docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md) track CLOCs, PointPainting-style fusion, BEVFusion, nuScenes, robustness, ablations and runtime. Dissertation chapter drafting starts after 10 October.

## Project Records

- [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) — master technical/dissertation record: project rationale, experiments, numerical results, failures, fixes, design decisions, limitations, and supported interpretations.
- [`DAILY_LOG.md`](DAILY_LOG.md) — chronological work-session record for new experiments and engineering changes.
- [`docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md`](docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md) — dated Phase 1 engineering targets, evidence gates, and GitHub checkpoint routine.
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
