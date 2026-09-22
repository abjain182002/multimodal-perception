# Results and Artifact Policy

The `results/` directory stores **small, frozen artifacts that are useful for reproducibility, supervisor review, and dissertation writing**.

## Currently version-controlled

### Depth-estimation study

- `depth_method_comparison.csv`
- `depth_method_comparison.png`

These summarize the V1/V2/V3 development and independent-holdout comparison.

### Camera-detector pilot

- `camera_detector_comparison.csv`
- `camera_detector_comparison.png`

These summarize the controlled fixed-operating-point YOLOv8x vs Falcon+dedup pilot on the matched development and holdout frame sets.

### Camera-LiDAR detector comparison

- `yolov8_falcon_lidar_comparison.csv`
- `yolov8_falcon_lidar_comparison.png`

These summarize the controlled YOLOv8x + LiDAR vs Falcon + LiDAR comparison using the same KITTI development/holdout splits and frozen V1 median-ROI LiDAR depth-association procedure.

### Current-generation YOLO camera-LiDAR comparison

- `yolov8_yolo26_lidar_comparison.csv`
- `yolov8_yolo26_lidar_comparison.png`

These summarize the controlled YOLOv8x vs YOLO26x comparison using the same KITTI development/holdout splits, projected-LiDAR association, minimum-point rule, matching procedure and V1 median-ROI depth estimator.

The reported depth errors are conditional on successfully matched fused detections and are not a complete detector ranking.

### PointPillars LiDAR-only 3D baseline

Primary quantitative artifacts:

- `pointpillars_kitti_ap40_strict_summary.csv`
- `pointpillars_kitti_ap40_strict.png`
- `pointpillars_kitti_ap40_strict.pdf`
- `kitti_full_metrics.json`
- `kitti_full_results.txt`

Reproducibility and diagnostic artifacts:

- `pointpillars_evidence_manifest.json`
- `kitti_split_overlap_audit.json`
- `pointpillars_moderate_failure_summary.json`
- `pointpillars_prediction_diagnostics_summary.json`
- `pointpillars_high_conf_unmatched_refined.json`
- `pointpillars_final_case_studies.json`
- `pointpillars_distance_diagnostic.png`
- `pointpillars_distance_diagnostic.pdf`

Frozen qualitative examples:

- `pointpillars_case_distant_success_002744.png`
- `pointpillars_case_distant_success_002744.pdf`
- `pointpillars_case_distant_localisation_near_miss_004096.png`
- `pointpillars_case_distant_localisation_near_miss_004096.pdf`
- `pointpillars_case_high_confidence_unmatched_001199.png`
- `pointpillars_case_high_confidence_unmatched_001199.pdf`

The primary PointPillars result is the reproduced KITTI Car **3D AP40 Moderate = 79.1299% at strict IoU 0.70** on the standard 3,769-frame validation split.

The distance and prediction-centric files are diagnostic analyses and must not be presented as substitutes for official KITTI AP/recall or as an independent official false-positive calculation.

Bulk PointPillars predictions, model weights, datasets and ordinary runtime logs remain outside Git history.

## What belongs here

Commit:

- compact CSV/JSON summary tables,
- final comparison figures,
- selected qualitative examples that support a specific finding,
- important failure-case visualizations,
- small artifacts explicitly referenced in the dissertation or supervisor updates.

Do **not** commit:

- the KITTI or nuScenes datasets,
- virtual environments,
- model weights,
- hundreds or thousands of generated frames,
- duplicate screenshots,
- temporary debugging images,
- bulk contents of `model_outputs/`.

Bulk generated outputs stay local and are intentionally excluded by `.gitignore`.

## Planned qualitative examples

When a representative image is worth preserving, use a structure such as:

```text
results/qualitative/
├── yolov8_lidar_000149.png
├── falcon_lidar_000149.png
└── falcon_duplicate_failure_006648.png
```

Only selected, verified images should be added. Their purpose should be documented in `PROJECT_HISTORY.md` or `DAILY_LOG.md`.
