# Multimodal Perception

MSc Major Project: Software-only deep learning pipeline for multi-modal object detection and robotic perception using camera and LiDAR fusion.

## Project Status

As of 30 September 2026, YOLO26x+CLOCs has been trained on 128 train-only frames (96 optimization / 32 internal development) and evaluated on all 3,769 KITTI SECOND validation frames. The reduced-FOV run reproduces the historical PointPillars baseline exactly across all nine AP40 metrics.

**Car strict IoU 0.70 3D AP40:** PointPillars 90.3449 / 79.1299 / 76.1566; CLOCs 90.1409 / 79.2542 / 73.6326 (Easy / Moderate / Hard). Moderate improves by 0.1243 points; Hard decreases by 2.5239 points. This is a mixed single-checkpoint result, not general fusion superiority. See [findings and limitations](docs/CLOCS_BASELINE_MATCHED_RESULT_2026-09-29.md).

Camera baselines, projected-LiDAR depth comparisons and Falcon experiments measure separate tasks. OpenPCDet/PV-RCNN feasibility is the next gate; Azitt weights, nuScenes/BEVFusion, semantic fusion, robustness and standardized runtime remain unmeasured. NVIDIA optimization is future work. Experiment targets remain subject to these gates; report drafting can proceed using frozen verified evidence.

## Project Records

- [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) — master technical/dissertation record: project rationale, experiments, numerical results, failures, fixes, design decisions, limitations, and supported interpretations.
- [`DAILY_LOG.md`](DAILY_LOG.md) — chronological work-session record for new experiments and engineering changes.
- [`docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md`](docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md) — dated Phase 1 engineering targets, evidence gates, and GitHub checkpoint routine.
- [`requirements/ENVIRONMENTS.md`](requirements/ENVIRONMENTS.md) — environment strategy and reproducibility notes.

## Frozen Results

- [Baseline-matched CLOCs AP40 figure](results/clocs_full_val_reduced/matched_ap40.png) and [CSV](results/clocs_full_val_reduced/matched_ap40.csv)
- [CLOCs range/occlusion coverage figure](results/clocs_full_val_reduced/matched_coverage.png)
- [OpenPCDet feasibility gate](docs/OPENPCDET_FEASIBILITY_2026-09-30.md)


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
