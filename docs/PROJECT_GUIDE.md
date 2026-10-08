# Multimodal Perception

MSc Major Project: Software-only deep learning pipeline for multi-modal object detection and robotic perception using camera and LiDAR fusion.

## Project Status

As of 2 October 2026, aligned YOLO26x+CLOCs training used 128 train-only KITTI frames (96 optimization / 32 internal development), with three initialization seeds and full 3,769-frame validation for each. All nine frozen PointPillars baseline AP40 metrics reproduce exactly in every run.

**Car strict IoU 0.70 Moderate 3D AP40:** PointPillars **79.1299**; epoch-12 CLOCs seeds **79.8616 / 76.2666 / 78.7616**. Only one seed improves Moderate; all three reduce Hard AP. Extending seed 20260928 to epoch 24 increases development candidate AP from 0.877935 to 0.883568 but reduces Moderate detection AP40 from 79.8616 to 79.4874. These diagnostics show inconsistent gains in this small-data implementation, not general fusion superiority or failure. Validation was inspected iteratively, not held as an untouched final test. See [seed and epoch findings](../results/kitti/clocs/seed_diagnostics/FINDINGS.md). Earlier checkpoint results remain preserved as historical experiments.

Camera baselines, projected-LiDAR depth comparisons and Falcon experiments measure separate tasks. Official pretrained OpenPCDet PV-RCNN completed all 3,769 validation frames, scoring Car 3D AP40 **92.1047 / 84.3604 / 82.4830** (Easy / Moderate / Hard). Native OpenPCDet and the pinned common evaluator agree exactly across all nine metrics. [Comparison and findings](../results/kitti/pv_rcnn/validation/FINDINGS.md) distinguish this LiDAR-only reproduction from Azitt/PV-RCNN++. Azitt weights, semantic fusion and robustness remain unmeasured. BEVFusion has completed paired pretrained-pipeline evaluation on nuScenes mini (81 samples) and an availability-selected official validation subset (1,231 samples / 31 complete scenes). Subset mAP is 64.2114 → 67.8147 and NDS is 69.2087 → 71.1543 (metrics × 100). Full nuScenes validation remains pending; this is not a controlled camera ablation. See [subset findings](../results/nuscenes/bevfusion/validation_subset/FINDINGS.md) and [mini findings](../results/nuscenes/bevfusion/mini/FINDINGS.md). A synchronized PV-RCNN T4 runtime benchmark is complete; comparable runtime measurements for other models remain pending. NVIDIA optimization is future work. Experiment targets remain subject to these gates; report drafting can proceed with evidence status stated explicitly.

### Verification scope and outstanding analysis

PointPillars baseline parity in fresh CLOCs runs checks reproduction against frozen results using the same evaluator. PV-RCNN native-versus-common agreement checks evaluator agreement on the same saved predictions. These checks answer different questions; neither establishes universal correctness or bitwise determinism, and shared evaluator ancestry has not yet been audited.

The nuScenes subset summary and initial figures were transcribed from execution output. On 7 October, all 84 original archive members were hash-verified and compact original metrics, scripts, audits, split records and summary curves were imported. Prediction tokens and stored metric arithmetic were checked; a CPU ground-truth replay on 8 October reproduced both headline metrics exactly, with summary/detail differences below 1e-12. See [replay evidence](../results/nuscenes/bevfusion/validation_subset/replay_20261008/README.md). This verifies reproduction using the same evaluator, not independent evaluator agreement. Large prediction files remain in the verified source archive. Mini/subset scene overlap remains unchecked; do not describe them as independent or the availability-selected subset as representative.

Seed 20260929's Easy-AP decline remains unexplained pending an existing-log audit. The score-blending pilot has not completed full AP40 validation. Any blend settings must be chosen on development data, with subsequent validation described as iterative. Larger-frame CLOCs training adequacy remains untested. These are open analyses, not established causes or completed experiments.

## Project Records

- [`PROJECT_HISTORY.md`](../PROJECT_HISTORY.md) — master technical/dissertation record: project rationale, experiments, numerical results, failures, fixes, design decisions, limitations, and supported interpretations.
- [`DAILY_LOG.md`](../DAILY_LOG.md) — chronological work-session record for new experiments and engineering changes.
- [`docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md`](../docs/PHASE1_DAILY_TARGETS_2026-09-25_TO_10-10.md) — dated Phase 1 engineering targets, evidence gates, and GitHub checkpoint routine.
- [`requirements/ENVIRONMENTS.md`](../requirements/ENVIRONMENTS.md) — environment strategy and reproducibility notes.

## Frozen Results

- [BEVFusion nuScenes subset findings](../results/nuscenes/bevfusion/validation_subset/FINDINGS.md), [metric comparison](../results/nuscenes/bevfusion/validation_subset/comparison.svg), and [class support](../results/nuscenes/bevfusion/validation_subset/class_support.svg)
- [BEVFusion nuScenes mini pilot](../results/nuscenes/bevfusion/mini/FINDINGS.md)

- [CLOCs seed and epoch diagnostics](../results/kitti/clocs/seed_diagnostics/FINDINGS.md), [3D AP differences](../results/kitti/clocs/seed_diagnostics/three_seed_and_epoch24_3d_deltas.svg), and [comparison CSV](../results/kitti/clocs/seed_diagnostics/ap40_comparison.csv)

- [Three-model KITTI AP40 comparison](../results/kitti/pv_rcnn/validation/comparison.png) and [CSV](../results/kitti/pv_rcnn/validation/comparison.csv)

- [Baseline-matched CLOCs AP40 figure](../results/kitti/clocs/baseline_matched_validation/matched_ap40.png) and [CSV](../results/kitti/clocs/baseline_matched_validation/matched_ap40.csv)
- [CLOCs range/occlusion coverage figure](../results/kitti/clocs/baseline_matched_validation/matched_coverage.png)
- [OpenPCDet feasibility gate](../docs/OPENPCDET_FEASIBILITY_2026-09-30.md)


- [`results/kitti/depth/method_comparison/depth_method_comparison.csv`](../results/kitti/depth/method_comparison/depth_method_comparison.csv)
- [`results/kitti/depth/method_comparison/figures/depth_method_comparison.png`](../results/kitti/depth/method_comparison/figures/depth_method_comparison.png)
- [`results/kitti/camera/detector_comparison/camera_detector_comparison.csv`](../results/kitti/camera/detector_comparison/camera_detector_comparison.csv)
- [`results/kitti/camera/detector_comparison/figures/camera_detector_comparison.png`](../results/kitti/camera/detector_comparison/figures/camera_detector_comparison.png)
- [`results/kitti/camera/lidar_depth_comparison/yolov8_falcon_lidar_comparison.csv`](../results/kitti/camera/lidar_depth_comparison/yolov8_falcon_lidar_comparison.csv)
- [`results/kitti/camera/lidar_depth_comparison/figures/yolov8_falcon_lidar_comparison.png`](../results/kitti/camera/lidar_depth_comparison/figures/yolov8_falcon_lidar_comparison.png)
- [`results/kitti/camera/lidar_depth_comparison/yolov8_yolo26_lidar_comparison.csv`](../results/kitti/camera/lidar_depth_comparison/yolov8_yolo26_lidar_comparison.csv)
- [`results/kitti/camera/lidar_depth_comparison/figures/yolov8_yolo26_lidar_comparison.png`](../results/kitti/camera/lidar_depth_comparison/figures/yolov8_yolo26_lidar_comparison.png)
- [`results/kitti/pointpillars/pointpillars_kitti_ap40_strict_summary.csv`](../results/kitti/pointpillars/pointpillars_kitti_ap40_strict_summary.csv)
- [`results/kitti/pointpillars/figures/pointpillars_kitti_ap40_strict.png`](../results/kitti/pointpillars/figures/pointpillars_kitti_ap40_strict.png)
- [`results/kitti/pointpillars/figures/pointpillars_distance_diagnostic.png`](../results/kitti/pointpillars/figures/pointpillars_distance_diagnostic.png)
- [`results/kitti/pointpillars/pointpillars_evidence_manifest.json`](../results/kitti/pointpillars/pointpillars_evidence_manifest.json)

See [`results/README.md`](../results/README.md) for the artifact-storage policy.

## Storage Policy

Datasets, virtual environments, model weights, and bulk generated `model_outputs/` are intentionally excluded from Git history. Source code, evaluation scripts, compact numerical results, dissertation-ready figures, and selected qualitative examples are version-controlled.


