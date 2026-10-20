# Multimodal Perception

## Current checkpoint — 16 October 2026

The bounded blur/noise/calibration study is complete and preserved: [one compact figure, seven conditions and original evidence](../results/nuscenes/bevfusion/camera_challenges/20261016/README.md). This supersedes earlier entries describing those experiments as planned. Existing masking, darkening, point-reduction, full-validation, runtime and architecture evidence remains unchanged. Next: critical literature/failure-handling synthesis, sourced financial scenarios, camera-only/physical multi-LiDAR scope resolution, and integrated report discussion. Not all supervisor requests are complete.

## Current analysis status — 15 October 2026

Full-validation accuracy, six annotated visual cases, a 600-frame camera-availability study, image-darkening study, matched point-reduction study and paired 150-frame/three-repeat runtime study are now complete and preserved. See the [results index](../results/README.md), [architecture](architecture/README.md), [point-reduction study](../results/nuscenes/bevfusion/point_reduction/20261015/README.md) and [report evidence/page plan](report/EVIDENCE_AND_PAGE_PLAN.md). Older dated sections below preserve earlier context. Comparable runtime is no longer pending for the evaluated BEVFusion LiDAR/fusion pair. Case studies are no longer pending.

Blur, noise and calibration challenges are planned; monetary hardware analysis remains a framework rather than a measured bill of materials. Camera-only 3D and physical multi-LiDAR scope is unresolved. Existing camera 2D/depth pilots are not a comparable camera-only nuScenes 3D benchmark. Fresh full-validation CPU replay completed on 20 October: all compared summary/detail values reproduce within 1e-12 (maximum error 8.881784197001252e-16). See [replay evidence](../results/nuscenes/bevfusion/full_validation/replay_20261020/README.md). This is same-evaluator reproduction; no new detector architecture or physical robot deployment is claimed.

MSc Major Project: Software-only deep learning pipeline for multi-modal object detection and robotic perception using camera and LiDAR fusion.

## Project Status

As of 2 October 2026, aligned YOLO26x+CLOCs training used 128 train-only KITTI frames (96 optimization / 32 internal development), with three initialization seeds and full 3,769-frame validation for each. All nine frozen PointPillars baseline AP40 metrics reproduce exactly in every run.

**Car strict IoU 0.70 Moderate 3D AP40:** PointPillars **79.1299**; epoch-12 CLOCs seeds **79.8616 / 76.2666 / 78.7616**. Only one seed improves Moderate; all three reduce Hard AP. Extending seed 20260928 to epoch 24 increases development candidate AP from 0.877935 to 0.883568 but reduces Moderate detection AP40 from 79.8616 to 79.4874. These diagnostics show inconsistent gains in this small-data implementation, not general fusion superiority or failure. Validation was inspected iteratively, not held as an untouched final test. See [seed and epoch findings](../results/kitti/clocs/seed_diagnostics/FINDINGS.md). Earlier checkpoint results remain preserved as historical experiments.

Camera baselines, projected-LiDAR depth comparisons and Falcon experiments measure separate tasks. Official pretrained OpenPCDet PV-RCNN completed all 3,769 validation frames, scoring Car 3D AP40 **92.1047 / 84.3604 / 82.4830** (Easy / Moderate / Hard). Native OpenPCDet and the pinned common evaluator agree exactly across all nine metrics. [Comparison and findings](../results/kitti/pv_rcnn/validation/FINDINGS.md) distinguish this LiDAR-only reproduction from Azitt/PV-RCNN++. Azitt weights, semantic fusion and robustness remain unmeasured. BEVFusion has completed paired pretrained-pipeline evaluation on nuScenes mini (81 samples) and an availability-selected official validation subset (1,231 samples / 31 complete scenes). Subset mAP is 64.2114 → 67.8147 and NDS is 69.2087 → 71.1543 (metrics × 100). Full nuScenes validation completed on 10 October: 6,019 samples / 150 scenes; LiDAR / camera–LiDAR mAP ×100 = 64.4301 / 68.3099 and NDS ×100 = 69.2353 / 71.1352. See [full findings](../results/nuscenes/bevfusion/full_validation/FINDINGS.md). This is not a controlled camera ablation. See [subset findings](../results/nuscenes/bevfusion/validation_subset/FINDINGS.md) and [mini findings](../results/nuscenes/bevfusion/mini/FINDINGS.md). A synchronized PV-RCNN T4 runtime benchmark is complete. A separately controlled BEVFusion LiDAR/fusion runtime study also completed on 15 October; timings across distinct protocols must not be pooled. NVIDIA optimization is future work. Experiment targets remain subject to these gates; report drafting can proceed with evidence status stated explicitly.

### Full sensor staging — 8 October 2026; evidence imported 9 October

The completed backup-index audit covers all 94,180 required files for 6,019 validation samples across all 150 validation scenes, with zero missing required files. Both part-07 archives had zero required-file matches; their audits are preserved. Full-split inference and accuracy evaluation completed on 10 October; [original full evidence](../results/nuscenes/bevfusion/full_validation/README.md) is imported. See [full staging evidence and scope](../results/nuscenes/preparation/parts01_10_20261008/README.md). Large sensor archives were not rehashed during the compact evidence import.

### Historical parts 1–6 staging — 8 October 2026

The parts 1–6 backup-index audit finds complete required inputs for 3,273 / 6,019 validation samples across 82 complete scenes, with no partial scenes. All original 1,231 frozen tokens remain covered. This is 54.38% input coverage, not new accuracy evidence. Parts 3–6 were selectively staged and backed up. This historical staging checkpoint has since been followed by full-validation inference and evaluation; see [staging evidence and scope](../results/nuscenes/preparation/parts01_06_20261008/README.md).

### Verification scope and outstanding analysis

PointPillars baseline parity in fresh CLOCs runs checks reproduction against frozen results using the same evaluator. PV-RCNN native-versus-common agreement checks evaluator agreement on the same saved predictions. These checks answer different questions; neither establishes universal correctness or bitwise determinism, and shared evaluator ancestry has not yet been audited.

The nuScenes subset summary and initial figures were transcribed from execution output. On 7 October, all 84 original archive members were hash-verified and compact original metrics, scripts, audits, split records and summary curves were imported. Prediction tokens and stored metric arithmetic were checked; a CPU ground-truth replay on 8 October reproduced both headline metrics exactly, with summary/detail differences below 1e-12. See [replay evidence](../results/nuscenes/bevfusion/validation_subset/replay_20261008/README.md). This verifies reproduction using the same evaluator, not independent evaluator agreement. Large prediction files remain in the verified source archive. Mini/subset scene overlap remains unchecked; do not describe them as independent or the availability-selected subset as representative.

The [20 October saved-record audit](../results/kitti/clocs/audit_20261020/README.md) is complete. Recorded input hashes, splits and settings agree across three seeds; checkpoint-selection/reload records and all nine baseline metrics pass. The cause of seed-20260929's Easy decline remains inconclusive. The recorded executed Kaggle trainer hash differs from the current generic repository trainer, so recover that exact source before a faithful rerun. The score blend is explicitly retained as an unfinished 16-frame candidate-ranking pilot, without full AP40 validation or a claimed seed-29 remedy. Larger-frame training adequacy and object-level failure attribution remain untested.

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








## Reference comparison audit — 20 October 2026

The [nuScenes upstream-reference audit](../results/nuscenes/bevfusion/reference_audit_20261020/README.md) verifies both live published checkpoint hashes and equality of the saved model dictionaries/test pipelines with the pinned reference for all three storage batches. Linked logs refine the score comparison and record historical configuration/runtime differences. Upstream nuScenes-devkit version, exact annotation equivalence and checkpoint metadata-to-log-row identity remain unverified. The cause of the remaining gap is unresolved; no causal attribution or further GPU rerun is claimed.

