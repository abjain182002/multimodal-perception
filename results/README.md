# Results overview

## Matched LiDAR point reduction — 15 October 2026

[Compact two-panel figure, tables and original evidence](nuscenes/bevfusion/point_reduction/20261015/README.md). Six conditions / 600 frames each. At 25% retention, mAP losses relative to own fresh baseline are **−13.2695 points (LiDAR-only)** and **−7.8961 (fusion)**. This is uniform synthetic thinning of keyframe-plus-sweep points, not a physical sensor or cost comparison.

## Image-darkening robustness study — 15 October 2026

[Figures, tables and original evidence](nuscenes/bevfusion/image_darkening/20261015/README.md). Fresh clean / 0.50 / 0.25 gain on the same 600 frames: mAP ×100 **68.1300 / 67.2352 / 64.5150**, NDS ×100 **71.2886 / 70.8179 / 69.1982**. Stronger artificial darkening reduces aggregate accuracy; no real-night or runtime claim.

Results are physically grouped by dataset, method and experiment. Open an experiment's README to find its figures, tables and evidence.

## KITTI

| Study | Folder | Status |
|---|---|---|
| Camera detector comparison | [Camera](kitti/camera/detector_comparison/) | Fixed-operating-point pilot |
| Camera detections with LiDAR depth | [Camera + depth](kitti/camera/lidar_depth_comparison/) | Conditional matched-detection depth results |
| Depth estimator V1/V2/V3 | [Depth methods](kitti/depth/method_comparison/) | Development / holdout comparison |
| PointPillars | [Baseline and cases](kitti/pointpillars/) | Full 3,769-frame validation |
| PV-RCNN | [Validation](kitti/pv_rcnn/validation/), [runtime](kitti/pv_rcnn/runtime/) | Full validation and serial runtime benchmark |
| CLOCs | [Study index](kitti/clocs/README.md) | Final results separated from pilots and historical diagnostics |

## nuScenes

| Study | Folder | Status |
|---|---|---|
| BEVFusion full validation | [Full validation](nuscenes/bevfusion/full_validation/) | 6,019 samples / 150 scenes; original evidence and figures imported |
| BEVFusion mini | [Mini](nuscenes/bevfusion/mini/) | 81-sample mini_val pilot |
| BEVFusion frozen validation subset | [Subset](nuscenes/bevfusion/validation_subset/) | 1,231 samples / 31 scenes; original evidence imported; same-evaluator ground-truth replay passed |

Input preparation is recorded separately: [full validation staging audit](nuscenes/preparation/parts01_10_20261008/README.md) covers 94,180 required files, 6,019 samples and 150 scenes with zero missing required inputs. Full-split inference and accuracy evaluation completed on 10 October; see the full-validation study above. The historical [parts 1–6 staging audit](nuscenes/preparation/parts01_06_20261008/README.md) covers complete inputs for 3,273 samples / 82 scenes (54.38%). It adds no accuracy results.

## Shared records

- [Validated results](VALIDATED_RESULTS.md)
- [Figure gallery and case studies](../docs/FIGURE_GALLERY.md)
- [Split audits](audits/kitti/) and [environment inventory](audits/environment/)
- [Old-to-new path map](../docs/RESULTS_PATH_MIGRATION.json)

## Artifact policy

Commit compact metrics, selected figures and cases, source/protocol records and verification audits. Keep datasets, weights, environments, bulk predictions and generated frame collections outside Git. Archived manifests retain original bytes and historical paths; the migration map records current locations. File moves do not create new experimental evidence.

## Supervisor benchmark comparisons

[Source-checked benchmark tables](benchmarks/BENCHMARK_COMPARISONS.md) separate full nuScenes checkpoint-linked references from measured KITTI AP40 results. AP11 and unspecified upstream AP values are context only; no cross-protocol deltas are claimed. Case-study visuals remain pending.

## nuScenes case studies

[Six annotated cases](nuscenes/bevfusion/case_studies/20261013/README.md): figures, object records, plotting code and hashes from existing full-validation predictions. No new accuracy evaluation or sensor-ablation experiment.

## Camera availability and robustness

[600-frame nuScenes camera-availability study](nuscenes/bevfusion/camera_availability/20261014/README.md) includes three comparison figures, exact tables, frozen selection, model/mask checks and original evaluator records. Accuracy completed; controlled runtime and cost analysis remain pending.


## Compute and hardware-cost analysis

[BEVFusion serial runtime and GPU memory](nuscenes/bevfusion/runtime/20261015/README.md): 150 frames, three isolated sessions per model, same T4. Includes raw records, figures and an accuracy–compute matrix with separate sample scopes disclosed. [Initial sensor-cost framework](cost_performance/README.md) distinguishes measured compute from unmeasured financial cost. [Architecture diagrams](../docs/architecture/README.md) explain the study branches and pinned BEVFusion flow.



