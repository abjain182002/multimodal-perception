# Figures and selected case studies

[Results overview](../results/README.md)

## BEVFusion full nuScenes validation

6,019 samples / 150 scenes; two official pretrained pipelines.

![Full-validation metric comparison](../results/nuscenes/bevfusion/full_validation/figures/comparison.png)

![Full-validation class AP comparison](../results/nuscenes/bevfusion/full_validation/figures/class_ap.png)

![Full-validation true-positive errors](../results/nuscenes/bevfusion/full_validation/figures/tp_errors.png)

![Filtered class support](../results/nuscenes/bevfusion/full_validation/figures/class_support.png)

[Findings and PDF/SVG downloads](../results/nuscenes/bevfusion/full_validation/README.md). [Six annotated nuScenes visual case studies](../results/nuscenes/bevfusion/case_studies/20261013/README.md) are available with captions, object records and limitations.

## PointPillars benchmark and diagnostics

### pointpillars distance diagnostic

![pointpillars_distance_diagnostic](../results/kitti/pointpillars/figures/pointpillars_distance_diagnostic.png)

[Open figure](../results/kitti/pointpillars/figures/pointpillars_distance_diagnostic.png) · [Experiment folder](../results/kitti/pointpillars/figures/)

### pointpillars kitti ap40 strict

![pointpillars_kitti_ap40_strict](../results/kitti/pointpillars/figures/pointpillars_kitti_ap40_strict.png)

[Open figure](../results/kitti/pointpillars/figures/pointpillars_kitti_ap40_strict.png) · [Experiment folder](../results/kitti/pointpillars/figures/)

## PointPillars selected case studies

### pointpillars case distant localisation near miss 004096

![pointpillars_case_distant_localisation_near_miss_004096](../results/kitti/pointpillars/case_studies/pointpillars_case_distant_localisation_near_miss_004096.png)

[Open figure](../results/kitti/pointpillars/case_studies/pointpillars_case_distant_localisation_near_miss_004096.png) · [Experiment folder](../results/kitti/pointpillars/case_studies/)

### pointpillars case distant success 002744

![pointpillars_case_distant_success_002744](../results/kitti/pointpillars/case_studies/pointpillars_case_distant_success_002744.png)

[Open figure](../results/kitti/pointpillars/case_studies/pointpillars_case_distant_success_002744.png) · [Experiment folder](../results/kitti/pointpillars/case_studies/)

### pointpillars case high confidence unmatched 001199

![pointpillars_case_high_confidence_unmatched_001199](../results/kitti/pointpillars/case_studies/pointpillars_case_high_confidence_unmatched_001199.png)

[Open figure](../results/kitti/pointpillars/case_studies/pointpillars_case_high_confidence_unmatched_001199.png) · [Experiment folder](../results/kitti/pointpillars/case_studies/)

## CLOCs seed and epoch diagnostics

### epoch12 to24 candidate ap

![epoch12_to24_candidate_ap](../results/kitti/clocs/seed_diagnostics/epoch12_to24_candidate_ap.svg)

[Open figure](../results/kitti/clocs/seed_diagnostics/epoch12_to24_candidate_ap.svg) · [Experiment folder](../results/kitti/clocs/seed_diagnostics/)

### three seed and epoch24 3d deltas

![three_seed_and_epoch24_3d_deltas](../results/kitti/clocs/seed_diagnostics/three_seed_and_epoch24_3d_deltas.svg)

[Open figure](../results/kitti/clocs/seed_diagnostics/three_seed_and_epoch24_3d_deltas.svg) · [Experiment folder](../results/kitti/clocs/seed_diagnostics/)

## PV-RCNN benchmark comparison

### comparison

![comparison](../results/kitti/pv_rcnn/validation/comparison.png)

[Open figure](../results/kitti/pv_rcnn/validation/comparison.png) · [Experiment folder](../results/kitti/pv_rcnn/validation/)

## BEVFusion nuScenes subset

### class support

![class_support](../results/nuscenes/bevfusion/validation_subset/class_support.png)

[Open figure](../results/nuscenes/bevfusion/validation_subset/class_support.png) · [Experiment folder](../results/nuscenes/bevfusion/validation_subset/)

### class support

![class_support](../results/nuscenes/bevfusion/validation_subset/class_support.svg)

[Open figure](../results/nuscenes/bevfusion/validation_subset/class_support.svg) · [Experiment folder](../results/nuscenes/bevfusion/validation_subset/)

### comparison

![comparison](../results/nuscenes/bevfusion/validation_subset/comparison.png)

[Open figure](../results/nuscenes/bevfusion/validation_subset/comparison.png) · [Experiment folder](../results/nuscenes/bevfusion/validation_subset/)

### comparison

![comparison](../results/nuscenes/bevfusion/validation_subset/comparison.svg)

[Open figure](../results/nuscenes/bevfusion/validation_subset/comparison.svg) · [Experiment folder](../results/nuscenes/bevfusion/validation_subset/)

## Camera and depth studies

### camera detector comparison

![camera_detector_comparison](../results/kitti/camera/detector_comparison/figures/camera_detector_comparison.png)

[Open figure](../results/kitti/camera/detector_comparison/figures/camera_detector_comparison.png) · [Experiment folder](../results/kitti/camera/detector_comparison/figures/)

### yolov8 falcon lidar comparison

![yolov8_falcon_lidar_comparison](../results/kitti/camera/lidar_depth_comparison/figures/yolov8_falcon_lidar_comparison.png)

[Open figure](../results/kitti/camera/lidar_depth_comparison/figures/yolov8_falcon_lidar_comparison.png) · [Experiment folder](../results/kitti/camera/lidar_depth_comparison/figures/)

### yolov8 yolo26 lidar comparison

![yolov8_yolo26_lidar_comparison](../results/kitti/camera/lidar_depth_comparison/figures/yolov8_yolo26_lidar_comparison.png)

[Open figure](../results/kitti/camera/lidar_depth_comparison/figures/yolov8_yolo26_lidar_comparison.png) · [Experiment folder](../results/kitti/camera/lidar_depth_comparison/figures/)

## Reading the figures

KITTI AP40 and nuScenes mAP/NDS are different metrics. Selected case studies illustrate individual outcomes, not their prevalence. The nuScenes subset is availability-selected; neither mini nor subset is full validation. Historical raw-cloud CLOCs plots are intentionally excluded from this review gallery. Read each experiment’s findings for the supported interpretation.


## Camera availability on 600 nuScenes frames

Same four selected frames per validation scene; existing six-camera checkpoint with specified missing-view masks.

![Camera availability metrics](../results/nuscenes/bevfusion/camera_availability/20261014/figures/metric_comparison.png)

![Per-class AP and support](../results/nuscenes/bevfusion/camera_availability/20261014/figures/class_ap.png)

![Per-class masking effects](../results/nuscenes/bevfusion/camera_availability/20261014/figures/class_mask_delta.png)

[Findings, PDF/SVG files and original evidence](../results/nuscenes/bevfusion/camera_availability/20261014/README.md). No controlled runtime or hardware-cost result is measured by these figures.



## BEVFusion runtime and memory — 15 October

![Latency sessions](../results/nuscenes/bevfusion/runtime/20261015/figures/latency_sessions.png)

![GPU memory](../results/nuscenes/bevfusion/runtime/20261015/figures/gpu_memory.png)

![Accuracy and latency, separate scopes](../results/nuscenes/bevfusion/runtime/20261015/figures/accuracy_latency.png)

[Original measurements, PDF/SVG figures and comparison tables](../results/nuscenes/bevfusion/runtime/20261015/README.md). Accuracy uses 600 frames; timing uses a nested 150 frames.

## Architecture

![Software study branches](architecture/study_overview.png)

![Pinned BEVFusion inference flow](architecture/bevfusion_inference.png)

[Sources, ownership distinctions and PDF/SVG diagrams](architecture/README.md).

