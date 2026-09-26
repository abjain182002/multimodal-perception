# Result figure inventory — 26 September 2026

This inventory distinguishes **full-validation detection**, **small controlled pilots**, **case diagnostics**, and **implementation validation**. Each image is backed by saved numbers or a recorded selection rule. Visual layout and image legibility still require separate review before dissertation or poster use. All paths below are repository-relative.

| Figure (PNG; PDF where listed) | Backing evidence | Defensible use | Boundary |
|---|---|---|---|
| `results/pointpillars_kitti_ap40_strict.png` (.pdf) | `results/pointpillars_kitti_ap40_strict_summary.csv`, `results/pointpillars_evidence_manifest.json` | PointPillars-only Car 2D/BEV/3D strict AP40, 3,769 KITTI validation frames | Pretrained LiDAR baseline, not fusion |
| `results/pointpillars_distance_diagnostic.png` (.pdf) | `results/pointpillars_evidence_manifest.json`, `results/pointpillars_prediction_diagnostics_summary.json` | Distance-dependent best-prediction 3D-IoU coverage | Diagnostic coverage is not distance-binned KITTI AP |
| `results/pointpillars_case_distant_success_002744.png` (.pdf) | `results/pointpillars_final_case_studies.json` | Numerically selected distant success | One case; do not extrapolate accuracy |
| `results/pointpillars_case_distant_localisation_near_miss_004096.png` (.pdf) | `results/pointpillars_final_case_studies.json` | Numerically selected localisation failure | One case; not aggregate recall |
| `results/pointpillars_case_high_confidence_unmatched_001199.png` (.pdf) | `results/pointpillars_final_case_studies.json`, `results/pointpillars_high_conf_unmatched_refined.json` | High-confidence unmatched diagnostic | Do not call an official KITTI false positive without official matching |
| `results/depth_method_comparison.png` | `results/depth_method_comparison.csv` | V1/V2/V3 comparison on 100-frame development and independent 100-frame holdout | Depth error is conditional on matched camera detections |
| `results/camera_detector_comparison.png` | `results/camera_detector_comparison.csv` | YOLOv8x/Falcon camera comparison on 10 development and 10 holdout frames | Fixed operating point, not official camera AP |
| `results/yolov8_falcon_lidar_comparison.png` | `results/yolov8_falcon_lidar_comparison.csv` | Controlled camera-input pilot with fixed LiDAR depth method | Matched-object conditional depth MAE, not 3D detection AP |
| `results/yolov8_yolo26_lidar_comparison.png` | `results/yolov8_yolo26_lidar_comparison.csv` | Modern camera-detector pilot with fixed LiDAR depth method | Matched-object groups may differ |
| `results/clocs_000149_iou_scipy_parity.png` | `results/clocs_000149_iou_scipy_audit.json`, `results/clocs_real_frame_000149_audit.json` | Reproduced 258-anchor train-frame CPU 3D-IoU geometry check | Not a trained fusion or official evaluator result |

## Results still needed for the primary claim

1. A train-only multi-frame CLOCs corpus with input hashes, label coverage, and a working pinned CUDA/SECOND IoU parity gate.
2. A trained, frozen fusion model evaluated beside PointPillars on the same eligible KITTI validation protocol, with 3D and BEV AP40 and per-class difficulty counts.
3. Controlled ablations, robustness measurements, runtime measured under a named consistent protocol, and repeatable failure cases. nuScenes/BEVFusion and other methods need their own completed protocol or an explicit blocker.

Track each completed experiment's image, numerical input, script, split, checkpoint, hardware and limitation in Git before selecting it for the final narrative. An absent result is an open task, not an inferred negative finding.
